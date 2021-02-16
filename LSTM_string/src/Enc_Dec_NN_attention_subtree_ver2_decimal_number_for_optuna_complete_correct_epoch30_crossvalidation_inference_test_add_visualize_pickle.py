import chainer
import chainer.links as L
import chainer.functions as F
import random
import cupy as np
import codecs
from chainer.training import extensions,triggers
import pickle
import optuna
from pathlib import Path
import glob 
import os 
import time
import collections
import argparse
import sys

SIZE = 10000
EOS = 1
BOS = 0

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
            self.attention_weight=[]
            self.attention_weight_one_eq = []
    def __call__(self, xs, ys):
        
        #add eos
        eos = self.xp.array([EOS], dtype=np.int32)
        ys_in = [F.concat((eos, y), axis=0) for y in ys]
        ys_out = [F.concat((y, eos), axis=0) for y in ys]
        
        #Both xs and ys_in are lists of arrays.
        exs = [self.embed_x(x) for x in xs]
        eys = [self.embed_y(y) for y in ys_in]
        
        # hx:dimension x batchsize
        # cx:dimension x batchsize
        # yx:batchsize x time size x dimension
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
        attention_weight_one_eq_part = []

        for i in range(len(embedded_output)):
            one_hidden_vector = F.get_item(embedded_output, i)
            one_hidden_vector_rp = F.broadcast_to(one_hidden_vector, attention.shape)
            weight = attention.__mul__(one_hidden_vector_rp)
            weight = F.sum(weight, axis=1)

            weight = F.broadcast_to(weight, (2, int(weight.shape[0])))
            weight = F.softmax(weight)
            weight = F.get_item(weight, 0)
            weight = F.broadcast_to(weight, (1, int(weight.shape[0])))
            attention_weight_one_eq_part.append(chainer.cuda.to_cpu(weight.data))
            context = F.matmul(weight, attention)
            one_hidden_vector = F.broadcast_to(one_hidden_vector,(1, int(one_hidden_vector.shape[0])))

            if a == 0:
                b = F.concat((one_hidden_vector, context))
                a = 1
            else:
                c = F.concat((one_hidden_vector, context))
                b = F.concat((b, c), axis=0)
        self.attention_weight_one_eq.append(attention_weight_one_eq_part)
            
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
            exs = self.embed_x(xs)
            hx, cx, yx = self.encoder(None, None, [exs])
            predicts = []
            eos = self.xp.array([EOS], dtype=np.int32)
            #input only eos, put previous output result for next output 
            for y in yx:  # batch-wise process
                predict = []
                ys_in = [eos]
                for i in range(max_length):
                    eys = [self.embed_y(y) for y in ys_in]
                    _, _, os = self.decoder(hx, cx, eys)
                    op = self._contexts_vector(os[0], y)
                    word_id = int(F.argmax(F.softmax(op)).data)  #convert to word ID
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
            self.attention_weight.append(self.attention_weight_one_eq)
            self.attention_weight_one_eq = []

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
        #with open('/home/kubota/LSTM_string_prework/pickle_data/12122_reverse_polish_prework.pickle','rb') as contents:
        #    self.sentence = pickle.load(contents)
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

    data = Data()
    global n_vocab
    n_vocab = len(data.vocab)
    global n_out
    n_out= len(data.vocab)
    
    dataset = chainer.datasets.get_cross_validation_datasets_random(data.sentence, 5, seed=1984)
    k_fold_for_test = 0
    train_and_validation = dataset[k_fold_for_test][0]
    test = dataset[k_fold_for_test][1]

    divide_train_and_validation = chainer.datasets.get_cross_validation_datasets_random(train_and_validation, 10, seed=1984)
    k_fold_for_train_valid = 6
    print("k_fold_for_train_valid:"+str(k_fold_for_train_valid))
    train = divide_train_and_validation[k_fold_for_train_valid][0]
    valid = divide_train_and_validation[k_fold_for_train_valid][1]

    train_iter = chainer.iterators.SerialIterator(train, batchsize, shuffle=False)
    valid_iter = chainer.iterators.SerialIterator(valid, batchsize, repeat=False, shuffle=False)

    optimizer = create_optimizer(trial,mlp)
    optimizer.setup(mlp)
    updater = chainer.training.StandardUpdater(train_iter, optimizer, converter=convert, device=0)
    stop_trigger = chainer.training.triggers.EarlyStoppingTrigger(
        monitor='validation/main/loss', check_trigger=(300, 'epoch'),
        max_trigger=(epochs, 'epoch'))

    trainer = chainer.training.Trainer(updater, stop_trigger,
                                       out=MODEL_DIRECTORY/f"model_{trial.number}")
    
    eval_model = mlp.copy()

    trainer.extend(
        chainer.training.extensions.Evaluator(valid_iter, eval_model, converter=convert, device=0))

    log_report_extention = chainer.training.extensions.LogReport(trigger=(10,'epoch'),log_name=None)
    trainer.extend(log_report_extention)
    trainer.extend(
        chainer.training.extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy', 'validation/main/loss']))
    trainer.extend(extensions.LogReport(trigger=(1, 'epoch')))
    trigger = triggers.MinValueTrigger('validation/main/loss', trigger=(1, 'epoch'))
    trainer.extend(extensions.snapshot_object(eval_model,\
                   filename='best_loss_model_epoch_{.updater.epoch}'),\
                   trigger=trigger)
    
    trainer.extend(extensions.snapshot(filename='latest_snapshot'),trigger=(1, 'epoch'))
    trainer.extend(
        optuna.integration.ChainerPruningExtension(
            trial, 'validation/main/loss', (PRUNER_INTERVAL, 'epoch')))
                                        
    trainer.run()
    mlp.to_gpu()

    loss = log_report_extention.log[-1]['validation/main/loss']
    count = 0
    for source, target in valid:
        start = time.time()
        predict = mlp.translate_with_beam_search(np.array(source, dtype=np.int32),max_length=100, beam_width=1)
        elapsed_time = time.time() - start

        source = ' '.join([data.vocab_inv[int(w)] for w in source if w != EOS and w != BOS])
        predict = ' '.join([data.vocab_inv[int(w)] for w in predict if w != EOS and w != BOS])
        target = ' '.join([data.vocab_inv[int(w)] for w in target if w != EOS and w != BOS])

        print("-----")
        print("source:", str(source))
        print("predict:", str(predict))
        print("elapsed_time:",str(elapsed_time))
        print("target:", str(target))

        if predict == target:
            count += 1
        print('- accuracy:',str(-(count/len(valid))))
        
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
    data = Data()
    global n_vocab
    n_vocab = len(data.vocab)
    global n_out
    n_out= len(data.vocab)
    n_hidden = trial.suggest_int("n_hidden", 100, 1024)
    n_layer = trial.suggest_int("n_layer", 1, 3)
    dropout = trial.suggest_uniform('dropout', 0.0, 0.2)

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
    optimizer_name = optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'MomentumSGD'])
    if optimizer_name == 'Adam':
        adam_alpha = trial.suggest_loguniform('adam_alpha', 1e-5, 1e-1)
        optimizer = chainer.optimizers.Adam(alpha=adam_alpha)
    else:
        momentum_sgd_lr = trial.suggest_loguniform('momentum_sgd_lr', 1e-5, 1e-1)
        optimizer = chainer.optimizers.MomentumSGD(lr = momentum_sgd_lr)

    weight_decay = trial.suggest_loguniform('weight_decay', 1e-10, 1e-3)
    gradient_clipping = trial.suggest_uniform('gradient_clipping', 0, 10)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(weight_decay))
    chainer.optimizer_hooks.GradientClipping(gradient_clipping)

    return optimizer

def main():

    parser = argparse.ArgumentParser(
        description='LSTM_strig_model')
    parser.add_argument('--batchsize', '-b', type=int, default=128,
                        help='Number of equations in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=200,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--token_dataset', '-T',type=str, default='../dataset/LSTM_string_polish_token.txt')
    parser.add_argument('--Integrand_dataset', '-i',type=str, default='../dataset/LSTM_string_Polish_test_Integrand.txt')
    parser.add_argument('--Primitive_dataset', '-p',type=str, default='../dataset/LSTM_string_Polish_test_Primitive.txt')
    parser.add_argument('--study_name', '-s',type=str, default='MLP_cupy_successiveHalvingPruner_epoch30_complete_correct_2nd_try_cross_valid')
    parser.add_argument('--learned_model', '-m',type=str, default='../models/LSTM_string_polish_best_model')
    args = parser.parse_args()

    sys.path.append('../dataset/')
    sys.path.append('../models')
    
    vocab = args.token_dataset 
    integrand_dataset = args.Integrand_dataset
    primitive_dataset = args.Primitive_dataset
    STUDY_NAME = args.study_name
    N_TRIALS = 100
    PRUNER_INTERVAL = 20 
    epochs = args.epoch
    batchsize = args.batchsize
    MODEL_DIRECTORY = Path(args.learned_model)
    study = optuna.create_study(study_name = STUDY_NAME,storage=f"sqlite:///{STUDY_NAME}.db",
            load_if_exists=True, pruner=optuna.pruners.MedianPruner())

    print('=== Best Trial ===')
    print(study.best_trial)
    evaluate_results(study.best_trial,vocab,integrand_dataset,primitive_dataset,MODEL_DIRECTORY)

def evaluate_results(trial,vocab,integrand_dataset,primitive_dataset,MODEL_DIRECTORY):
    """ Evaluate the optimization results.
        
    Args:
        study: optuna.trial.Trial
    Returns: 
e        None
    """
    trial_number = 8#string Polishの場合       
    print("triaL_number"+str(trial_number))
    data = Data(vocab,integrand_dataset,primitive_dataset)
    n_vocab = len(data.vocab)
    n_out = len(data.vocab)
    n_hidden = trial.params['n_hidden']
 
    mlp = EncoderDecoder(trial.params['n_layer'], n_vocab, n_out, trial.params['n_hidden'], trial.params['dropout'])
    
    snapshots = glob.glob(str(MODEL_DIRECTORY))
    print("snapshots:"+str(snapshots))
    latest_snapshot = max(snapshots, key=os.path.getctime)  # The latest snapshot of the trial
    print(f"Loading: {latest_snapshot}")
    
    chainer.serializers.load_npz(latest_snapshot,mlp)
        
    seed = 1984
    random.seed(seed)
    np.random.seed(seed)

    #data = Data()

    #dataset = chainer.datasets.get_cross_validation_datasets_random(data.sentence, 5, seed=1984)
    #k_fold_for_test = 0
    #train_and_validation = dataset[k_fold_for_test][0]
    test = data.sentence#dataset[k_fold_for_test][1]
    
    #divide_train_and_validation = chainer.datasets.get_cross_validation_datasets_random(train_and_validation, 10, seed=1984)
    #k_fold_for_train_valid = 6
    #print("k_fold_for_train_valid:"+str(k_fold_for_train_valid))
    #valid = divide_train_and_validation[k_fold_for_train_valid][1]

    mlp.to_gpu()
    count = 0
    wrong_eq_list =[]
    index= 0
    list_result_for_attention = []
    
    for source, target in test:

        target_original = test[index][1]           
        start = time.time()
        predict = mlp.translate_with_beam_search(np.array(source, dtype=np.int32),max_length=100,beam_width=1)
        elapsed_time = time.time() - start
        source_str_list = [data.vocab_inv[int(w)] for w in source]
        source = ' '.join([data.vocab_inv[int(w)] for w in source if w != EOS and w != BOS])
        predict_str_list = [data.vocab_inv[int(w)] for w in predict]
        predict = ' '.join([data.vocab_inv[int(w)] for w in predict if w != EOS and w != BOS])
        target = ' '.join([data.vocab_inv[int(w)] for w in target if w != EOS and w != BOS])
        list_result_for_attention.append((source_str_list,predict_str_list))

        print("-----")
        print("source:", str(source))
        print("predict:", str(predict))
        print("elapsed_time:",str(elapsed_time))
        print("target:", str(target))

        if predict == target:
            count += 1
            print("correct:"+str(index))
            index+=1
        else:
            print("wrong:"+str(index))
            wrong_eq_list.append(index)
            index+=1
            
        print('- accuracy:',str(-(count/len(test))))
    
    #with open('result_for_attention_12122_fold_{0}_test.pickle'.format(k_fold_for_train_valid),'wb') as f:
    #    pickle.dump(list_result_for_attention,f)
    #with open('attention_weight_12122_fold_{0}_test.pickle'.format(k_fold_for_train_valid),'wb') as f:
    #   pickle.dump(mlp.attention_weight,f)

    print("count:"+str(count))
    print("len(test):{}".format(len(test)))
    print("test_accuracy:{}%".format(count/len(test)))
    print("wrong_eq_list:{}".format(wrong_eq_list))
    
if __name__ == '__main__':
    main()
