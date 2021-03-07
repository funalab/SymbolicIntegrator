import chainer
import chainer.links as L
import chainer.functions as F
import random
#import cupy as np
import numpy as np
import codecs
from chainer.training import extensions,triggers
import pickle
import optuna
from pathlib import Path
import glob 
import os 
import time
import argparse
import sys

SIZE = 10000
EOS = 1

class EncoderDecoder(chainer.Chain):
    def __init__(self, n_layer, n_vocab, n_out, n_hidden, dropout):
        super(EncoderDecoder, self).__init__()

        with self.init_scope():
            self.embed_x = L.EmbedID(n_vocab, n_hidden)
            self.embed_y = L.EmbedID(n_out, n_hidden)

            self.encoder = L.NStepLSTM(
                n_layers=n_layer,
                in_size=n_hidden,
                out_size=n_hidden,
                dropout=dropout)
            self.decoder = L.NStepLSTM(
                n_layers=n_layer,
                in_size=n_hidden,
                out_size=n_hidden,
                dropout=dropout)

            self.W_C = L.Linear(2 * n_hidden, n_hidden)
            self.W_D = L.Linear(n_hidden, n_out)
            self.W_E = L.Linear(2 * n_hidden, n_out)

    def __call__(self, xs, ys):
        
        #add eos
        eos = self.xp.array([EOS], dtype=np.int32)
        ys_in = [F.concat((eos, y), axis=0) for y in ys]
        ys_out = [F.concat((y, eos), axis=0) for y in ys]
        
        # Both xs and ys_in are lists of arrays.
        exs = [self.embed_x(x) for x in xs]
        eys = [self.embed_y(y) for y in ys_in]
        
        # hx:dimension x batchsize
        # cx:dimension x batchsize
        # yx:batchsize x timesize x dimension
        hx, cx, yx = self.encoder(None, None, exs) 
        _, _, os = self.decoder(hx, cx, eys)

        loss = 0
        for o, y, ey in zip(os, yx, ys_out):  #batch-wise process
            op = self._contexts_vector(o,y)
            loss += F.softmax_cross_entropy(op, ey)
        loss /= len(yx)

        chainer.report({'loss': loss}, self)
        return loss

    def _contexts_vector(self, embedded_output, attention):
        a = 0 # flag
        for i in range(len(embedded_output)):
            one_hidden_vector = F.get_item(embedded_output, i)
            one_hidden_vector_rp = F.broadcast_to(one_hidden_vector, attention.shape)
            weight = attention.__mul__(one_hidden_vector_rp)
            weight = F.sum(weight, axis=1)

            weight = F.broadcast_to(weight, (2, int(weight.shape[0])))
            weight = F.softmax(weight)
            weight = F.get_item(weight, 0)
            weight = F.broadcast_to(weight, (1, int(weight.shape[0])))

            context = F.matmul(weight, attention)
            one_hidden_vector = F.broadcast_to(one_hidden_vector,(1, int(one_hidden_vector.shape[0])))

            if a == 0:
                b = F.concat((one_hidden_vector, context))
                a = 1
            else:
                c = F.concat((one_hidden_vector, context))
                b = F.concat((b, c), axis=0)

        return self.W_E(b)

    def _calculate_attention_layer_output(self, embedded_output, attention):
        inner_prod = F.matmul(embedded_output, attention, transb=True)
        weights = F.softmax(inner_prod)
        contexts = F.matmul(weights, attention)
        concatenated = F.concat((contexts, embedded_output))
        new_embedded_output = F.tanh(self.W_C(concatenated))
        return self.W_D(new_embedded_output)
    """
    def translate(self, xs, max_length=30):
        with chainer.no_backprop_mode(), chainer.using_config("train", False):
            #xs = xs[::-1]  # reverse list
            # exs = [self.embed_x(x) for x in xs]
            exs = self.embed_x(xs)
            hx, cx, yx = self.encoder(None, None, [exs])

            predicts = []
            eos = self.xp.array([EOS], dtype=np.int32)
            # EOSだけ入力，あとは予想した出力を入力にして繰り返す
            for y in yx:  # バッチ単位
                predict = []
                ys_in = [eos]
                for i in range(max_length):
                    eys = [self.embed_y(y) for y in ys_in]
                    _, _, os = self.decoder(hx, cx, eys)
                    #op = self._calculate_attention_layer_output(os[0], y)
                    op = self._contexts_vector(os[0], y)
                    word_id = int(F.argmax(F.softmax(op)).data)  # 単語IDに戻す

                    if word_id == EOS: break
                    predict.append(word_id)
                    ys_in = [self.xp.array([word_id], dtype=np.int32)]
                predicts.append(np.array(predict))
            return predict
    """
    def _translate_one_word(self, wid, hidden_states, cell_states, attentions):
        y = np.array([wid], dtype=np.int32)
        embedded_y = self.embed_y(y)
        hidden_states, cell_states, embedded_outputs = self.decoder(hidden_states, cell_states, [embedded_y])                                                    
        output = self._contexts_vector(embedded_outputs[0], attentions[0])
        output = F.softmax(output)
        return output[0], hidden_states, cell_states

    def translate_with_beam_search(self, sentence, max_length=100, beam_width=10):

        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            #sentence = sentence[::-1]

            embedded_xs = self.embed_x(sentence)
            hidden_states, cell_states, attentions = self.encoder(None, None, [embedded_xs])

            heaps = [[] for _ in range(max_length + 1)]

            # (score, translation, hidden_states, cell_states)                                                                                                                                
            heaps[0].append((0, [EOS], hidden_states, cell_states))

            solution = []
            solution_score = 1e8

            for i in range(max_length):
                heaps[i] = sorted(heaps[i], key=lambda t: t[0])[:beam_width]

                for score, translation, i_hidden_states, i_cell_states in heaps[i]:
                    wid = translation[-1]
                    output, new_hidden_states, new_cell_states = \
                        self._translate_one_word(wid, i_hidden_states, i_cell_states, attentions)

                    for next_wid in np.argsort(output.data)[::-1]:
                        if output.data[next_wid] < 1e-6:
                            break
                        next_score = score - np.log(output.data[next_wid])
                        if next_score > solution_score:
                            break
                        next_translation = translation + [next_wid]
                        next_item = (next_score, next_translation, new_hidden_states, new_cell_states)

                        if next_wid == EOS:
                            if next_score < solution_score:
                                solution = translation[1:]  # [1:] drops first EOS                                                                                                            
                                solution_score = next_score
                        else:
                            heaps[i + 1].append(next_item)

            return solution

class Data(chainer.dataset.DatasetMixin):
    def __init__(self,vocab,integrand_dataset,primitive_dataset):

        file_path_combined_words = vocab
        file_path_integrand = integrand_dataset
        file_path_primitive = primitive_dataset

        f_words = codecs.open(file_path_combined_words, 'r', 'utf8')
        line = f_words.readline()
        id_to_char = {}
        char_to_id = {}

        while line:
            l = line.strip().split(',')
            if len(l) == 2:
                char_to_id[l[1]] = int(l[0])
                id_to_char[(int(l[0]))] = l[1]
            elif len(l) != 2:  # for_comma
                char_to_id[l[1] + ',' + l[2]] = int(l[0])
                id_to_char[(int(l[0]))] = ','
            line = f_words.readline()
        f_words.close()

        self.vocab = char_to_id

        
        self.train_data = []
        self.test_data = []

        # integrand and primitive length
        questions, answers = [], []
        maximum_len_questions = 0
        maximum_len_answers = 0
        for line in open(file_path_integrand, 'r'):
            questions.append(line)
            if len(line) > maximum_len_questions:
                maximum_len_questions = len(line)
        for line in open(file_path_primitive, 'r'):
            answers.append(line)
            if len(line) > maximum_len_answers:
                maximum_len_answers = len(line)

        x = []
        t = []

        for i, sentence in enumerate(questions):
            line_question_list = sentence.strip().split(' ')
            line_question_list = [x for x in line_question_list if x]
            x.append([char_to_id[c] for c in line_question_list])
        for i, sentence in enumerate(answers):
            line_answer_list = sentence.strip().split(' ')
            line_answer_list = [x for x in line_answer_list if x]
            t.append([char_to_id[c] for c in line_answer_list])

        self.sentence = []
        for i in range(len(x)):
            self.sentence.append((np.array(x[i]).astype(np.int32), np.array(t[i]).astype(np.int32)))
        
        self.vocab_inv = {}
        self.vocab_inv = id_to_char

def flatten(nested_list):
    return [e for inner_list in nested_list for e in inner_list]

def convert(batch, device):
    def to_device_batch(batch):
        return [chainer.dataset.to_device(device, x) for x in batch]

    res = {'xs': to_device_batch([x for x, _ in batch]),
           'ys': to_device_batch([y for _, y in batch])}
    return res

def objective(trial):
    """Objective function to make optimization for Optuna.

    Args:
       trial: optuna.trial.Trial
    Returns:
        loss: float
            Loss value for the trial
    """
    mlp = generate_model(trial)
    seed = 1984
    random.seed(seed)
    np.random.seed(seed)

    data = Data(args.token_dataset,args.Integrand_dataset,args.Primitive_dataset)
    global n_vocab
    n_vocab = len(data.vocab)
    global n_out
    n_out= len(data.vocab)
    
    divide_train_and_validation = chainer.datasets.get_cross_validation_datasets_random(data.sentence, 10, seed=1984)
    k_fold_for_train_valid = args.kfold
    print("k_fold_for_train_valid:"+str(k_fold_for_train_valid))
    train = divide_train_and_validation[k_fold_for_train_valid][0]
    valid = divide_train_and_validation[k_fold_for_train_valid][1]

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize, shuffle=False)
    valid_iter = chainer.iterators.SerialIterator(valid, args.batchsize, repeat=False, shuffle=False)

    optimizer = create_optimizer(trial,mlp)
    optimizer.setup(mlp)
    updater = chainer.training.StandardUpdater(train_iter, optimizer, converter=convert, device=GPU)
    stop_trigger = chainer.training.triggers.EarlyStoppingTrigger(
        monitor='validation/main/loss', check_trigger=(300, 'epoch'),
        max_trigger=(args.epoch, 'epoch'))

    trainer = chainer.training.Trainer(updater, stop_trigger,
                                       out=MODEL_DIRECTORY/f"model_{trial.number}")
    
    eval_model = mlp.copy()

    trainer.extend(
        chainer.training.extensions.Evaluator(valid_iter, eval_model, converter=convert, device=GPU))

    log_report_extention = chainer.training.extensions.LogReport(trigger=(1,'epoch'),log_name=None)
    trainer.extend(log_report_extention)
    trainer.extend(
        chainer.training.extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss']))
    trainer.extend(extensions.LogReport(trigger=(1, 'epoch')))
    trigger = triggers.MinValueTrigger('validation/main/loss', trigger=(1, 'epoch'))
    trainer.extend(extensions.snapshot_object(eval_model,\
                   filename='best_loss_model_epoch_{.updater.epoch}'),\
                   trigger=trigger)
    
    trainer.extend(extensions.snapshot(filename='latest_snapshot'),trigger=(1, 'epoch'))
    #trainer.extend(
    #    optuna.integration.ChainerPruningExtension(
    #        trial, 'validation/main/loss', (PRUNER_INTERVAL, 'epoch')))
                                        
    trainer.run()
    if GPU >= 0:
        mlp.to_gpu()
    else:
        mlp.to_cpu()

    loss = log_report_extention.log[-1]['validation/main/loss']
    count = 0
    index = 0
    print("---validation data---")
    for source, target in valid:
        start = time.time()
        predict = mlp.translate_with_beam_search(np.array(source, dtype=np.int32),max_length=100, beam_width=1)
        elapsed_time = time.time() - start

        source = ' '.join([data.vocab_inv[int(w)] for w in source if w != EOS])
        predict = ' '.join([data.vocab_inv[int(w)] for w in predict if w != EOS])
        target = ' '.join([data.vocab_inv[int(w)] for w in target if w != EOS])

        print("-----")
        print("eq_num:",str(index))
        print("Integrand(Input):", str(source))
        print("Pritmitive(Output):", str(predict))
        print("Correct Answer:", str(target))
        print("elapsed_time:",str(elapsed_time))

        if predict == target:
            count += 1
        #print('- accuracy:',str(-(count/len(valid))))
        index += 1
    #return loss
    return -(count/len(valid))
def generate_model(trial):
    """
    :Args
         trial : optuna.trial.Trial
    :return:
         classifier: chainer.links.Classifier
    """
    # Suggest hyperparameters
    data = Data(args.token_dataset,args.Integrand_dataset,args.Primitive_dataset)
    global n_vocab
    n_vocab = len(data.vocab)
    global n_out
    n_out= len(data.vocab)
    n_hidden = trial.suggest_int("n_hidden", 929, 929)
    n_layer = trial.suggest_int("n_layer", 3, 3)
    dropout = trial.suggest_uniform('dropout', 0.13961107100947115, 0.13961107100947115)

    print('--')
    print(f"Trial: {trial.number}")
    print('Current hyperparameters:')
    print(f"    The number of layers: {n_layer}")
    print(f"    the dimensions of hidden vector: {n_hidden}")
    print(f"    the ratio for dropout: {dropout}")
    print('--')
    
    mlp = EncoderDecoder(n_layer, n_vocab, n_out, n_hidden, dropout)

    return mlp

def create_optimizer(trial,model):
    optimizer_name = optimizer_name = trial.suggest_categorical('optimizer', ['Adam'])
    if optimizer_name == 'Adam':
        adam_alpha = trial.suggest_loguniform('adam_alpha', 0.0018326844916546393, 0.0018326844916546393)
        print(f"    adam_alpha : {adam_alpha}")
        optimizer = chainer.optimizers.Adam(alpha=adam_alpha)
    else:
        momentum_sgd_lr = trial.suggest_loguniform('momentum_sgd_lr', 1e-5, 1e-1)
        print(f"    momentum_sgd_lr : {momentum_sgd_lr}")
        optimizer = chainer.optimizers.MomentumSGD(lr = momentum_sgd_lr)

    weight_decay = trial.suggest_loguniform('weight_decay', 8.590020281926726e-07, 8.590020281926726e-07)
    print(f"    weight decay : {weight_decay}")
    gradient_clipping = trial.suggest_uniform('gradient_clipping', 4.1231125968560445, 4.1231125968560445)
    print(f"    gradient clipping : {gradient_clipping}")
    
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(weight_decay))
    chainer.optimizer_hooks.GradientClipping(gradient_clipping)

    return optimizer

def main():

    parser = argparse.ArgumentParser(description='LSTM_strig_model')
    parser.add_argument('--batchsize', '-b', type=int, default=128,
                        help='Number of equations in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=200,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--kfold', type=int, default=0,
                        help='train_validation dataset fold number')
    parser.add_argument('--token_dataset', '-T',type=str, default='../dataset/LSTM_string_polish_token.txt')
    parser.add_argument('--Integrand_dataset', '-i',type=str, default='../dataset/LSTM_string_Polish_train_valid_Integrand_first_10eq.txt')
    parser.add_argument('--Primitive_dataset', '-p',type=str, default='../dataset/LSTM_string_Polish_train_valid_Primitive_first_10eq.txt')
    parser.add_argument('--study_name', '-s',type=str, default='MLP_cupy_successiveHalvingPruner_epoch30_complete_correct_2nd_try_cross_valid_new')
    parser.add_argument('--learned_model', '-m',type=str, default='../model/LSTM_string_polish_best_model_new')
    parser.add_argument('--integrated_model', action='store_true',help='use as a component of Integrated All models')
    global args
    args = parser.parse_args()
    
    sys.path.append('../dataset/')
    sys.path.append('../models')

    if args.gpu >=0:
        global np
        import cupy as np
    global GPU
    GPU = args.gpu

    vocab = args.token_dataset
    integrand_dataset = args.Integrand_dataset
    primitive_dataset = args.Primitive_dataset
    STUDY_NAME = args.study_name
    N_TRIALS = 1
    #global PRUNER_INTERVAL
    #PRUNER_INTERVAL = 1
    epochs = args.epoch
    batchsize = args.batchsize
    global MODEL_DIRECTORY
    MODEL_DIRECTORY = Path(args.learned_model+"_fold_"+str(args.kfold))
    #Path('models/MLP_cupy_successiveHalvingPruner_epoch30_complete_correct_2nd_try_cross_valid')
    study = optuna.create_study(study_name = STUDY_NAME,storage=f"sqlite:///{STUDY_NAME}.db",
                                load_if_exists=True, pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=N_TRIALS,timeout=162000)
    #evaluate_results(study.best_trial)
    print('=== Best Trial ===')
    print(study.best_trial)

"""
def evaluate_results(trial):
    #Evaluate the optimization results.
    #    
    #Args:
    #    study: optuna.trial.Trial
    #Returns: 
    #    None
    
    trial_number = trial.number
    print("triaL_number"+str(trial_number))
    data = Data(args.token_dataset,args.Integrand_dataset,args.Primitive_dataset)
    n_vocab = len(data.vocab)
    n_out = len(data.vocab)
    n_hidden = trial.params['n_hidden']
    mlp = EncoderDecoder(trial.params['n_layer'], n_vocab, n_out, trial.params['n_hidden'], trial.params['dropout'])
    
    snapshots = glob.glob(str(MODEL_DIRECTORY))
    latest_snapshot = max(snapshots, key=os.path.getctime)# The latest snapshot of the trial
    
    print(f"Loading: {latest_snapshot}")
    chainer.serializers.load_npz(latest_snapshot,mlp)

    seed = 1984
    random.seed(seed)
    np.random.seed(seed)

    divide_train_and_validation = chainer.datasets.get_cross_validation_datasets_random(data.sentence, 10, seed=1984)
    k_fold_for_train_valid = args.k_fold
    #train = divide_train_and_validation[k_fold_for_train_valid][0]
    valid = divide_train_and_validation[k_fold_for_train_valid][1]
    
    if GPU >= 0:
        mlp.to_gpu()
    else:
        mlp.to_cpu()
    count = 0
    index = 0
    
    for source, target in valid:
        #predict = mlp.translate(np.array(source,dtype=np.int32))
        start = time.time()
        predict = mlp.translate_with_beam_search(np.array(source, dtype=np.int32),max_length=100, beam_width=1)
        elapsed_time = time.time() - start

        source = ' '.join([data.vocab_inv[int(w)] for w in source if w != EOS])
        predict = ' '.join([data.vocab_inv[int(w)] for w in predict if w != EOS])
        target = ' '.join([data.vocab_inv[int(w)] for w in target if w != EOS])
        print("-----")
        if not args.integrated_model:
            print("eq_num:",str(index))
        print("Integrand(Input):", str(source))
        print("Primitive(Output):", str(predict))
        print("Correct Answer:", str(target))
        print("elapsed_time:",str(elapsed_time))
    
        if predict == target:
            count += 1
            print("Correct!")
            index+=1 
        else:
            wrong_eq_list.append(index)
            print("Wrong")
            index+=1
            
    if not args.integrated_model:
        print("---Result Summary---")
        print("Total correct equation num:{}".format(str(count)))
        print("len(valid):{}".format(len(valid)))
        print("Complete Correct Answer Rate:{}%".format(100*count/len(valid)))
"""
if __name__ == '__main__':
    main()
