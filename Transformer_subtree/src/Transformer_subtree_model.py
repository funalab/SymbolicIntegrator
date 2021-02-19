# encoding: utf-8
import argparse
import json
import os.path
import time

from nltk.translate import bleu_score
import numpy
import six
import pickle

import chainer
from chainer import cuda
from chainer.dataset import convert
from chainer import reporter
from chainer import training
from chainer.training import extensions

import preprocess
import net_check_visualization_all_layer

from subfuncs import VaswaniRule


def seq2seq_pad_concat_convert(xy_batch, device, eos_id=0, bos_id=2):
    """
    Args:
        xy_batch (list of tuple of two numpy.ndarray-s or cupy.ndarray-s):
            xy_batch[i][0] is an array
            of token ids of i-th input sentence in a minibatch.
            xy_batch[i][1] is an array
            of token ids of i-th target sentence in a minibatch.
            The shape of each array is `(sentence length, )`.
        device (int or None): Device ID to which an array is sent. If it is
            negative value, an array is sent to CPU. If it is positive, an
            array is sent to GPU with the given ID. If it is ``None``, an
            array is left in the original device.

    Returns:
        Tuple of Converted array.
            (input_sent_batch_array, target_sent_batch_input_array,
            target_sent_batch_output_array).
            The shape of each array is `(batchsize, max_sentence_length)`.
            All sentences are padded with -1 to reach max_sentence_length.
    """

    x_seqs, y_seqs = zip(*xy_batch)

    x_block = convert.concat_examples(x_seqs, device, padding=-1)
    xp = cuda.get_array_module(x_block)

    if len(x_block[0]) % 3 != 0:
        x_block_add_two_more_padding = xp.c_[x_block, xp.full((len(x_block), 1), -1)]
        while len(x_block_add_two_more_padding[0]) % 3 != 0:
            x_block_add_two_more_padding = xp.c_[x_block_add_two_more_padding, xp.full((len(x_block), 1), -1)]
        x_block = x_block_add_two_more_padding #update
    y_block = convert.concat_examples(y_seqs, device, padding=-1)
    xp = cuda.get_array_module(y_block)
    if len(y_block[0]) % 3 != 0:
        y_block_add_two_more_padding = xp.c_[y_block, xp.full((len(y_block), 1), -1)]
        while len(y_block_add_two_more_padding[0]) % 3 != 0:
            y_block_add_two_more_padding = xp.c_[y_block_add_two_more_padding, xp.full((len(y_block), 1), -1)]
        y_block = y_block_add_two_more_padding #update

    # The paper did not mention eos
    # add eos
    x_block = xp.pad(x_block, ((0, 0), (0, 1)),
                     'constant', constant_values=-1)
    x_block = xp.pad(x_block, ((0, 0), (0, 1)),
                     'constant', constant_values=-1)
    x_block = xp.pad(x_block, ((0, 0), (0, 1)),
                     'constant', constant_values=-1)
    for i_batch, seq in enumerate(x_seqs):
        x_block[i_batch, len(seq)] = eos_id
    x_block = xp.pad(x_block, ((0, 0), (1, 0)),
                     'constant', constant_values=bos_id)
    x_block = xp.pad(x_block, ((0, 0), (1, 0)),
                     'constant', constant_values=bos_id)
    x_block = xp.pad(x_block, ((0, 0), (1, 0)),
                     'constant', constant_values=bos_id)

    y_out_block = xp.pad(y_block, ((0, 0), (0, 1)),
                         'constant', constant_values=-1)
    y_out_block = xp.pad(y_out_block, ((0, 0), (0, 1)),
                         'constant', constant_values=-1)
    y_out_block = xp.pad(y_out_block, ((0, 0), (0, 1)),
                         'constant', constant_values=-1)
    for i_batch, seq in enumerate(y_seqs):
        y_out_block[i_batch, len(seq)] = eos_id

    y_in_block = xp.pad(y_block, ((0, 0), (1, 0)),
                        'constant', constant_values=bos_id)
    y_in_block = xp.pad(y_in_block, ((0, 0), (1, 0)),
                        'constant', constant_values=bos_id)
    y_in_block = xp.pad(y_in_block, ((0, 0), (1, 0)),
                        'constant', constant_values=bos_id)
    return (x_block, y_in_block, y_out_block)


def source_pad_concat_convert(x_seqs, device, eos_id=0, bos_id=2):
    x_block = convert.concat_examples(x_seqs, device, padding=-1)
    xp = cuda.get_array_module(x_block)
    
    if len(x_block[0]) % 3 != 0:
        x_block_add_two_more_padding = xp.c_[x_block, xp.full((len(x_block), 1), -1).astype('i')]
        while len(x_block_add_two_more_padding[0]) % 3 != 0:
            x_block_add_two_more_padding = xp.c_[x_block_add_two_more_padding, xp.full((len(x_block), 1), -1).astype('i')]
        x_block = x_block_add_two_more_padding #update  
    
    # add eos
    x_block = xp.pad(x_block, ((0, 0), (0, 1)),
                     'constant', constant_values=-1).astype('i')
    x_block = xp.pad(x_block, ((0, 0), (0, 1)),
                     'constant', constant_values=-1).astype('i')  
    x_block = xp.pad(x_block, ((0, 0), (0, 1)),
                     'constant', constant_values=-1).astype('i')  

    for i_batch, seq in enumerate(x_seqs):
        x_block[i_batch, len(seq)] = eos_id
    x_block = xp.pad(x_block, ((0, 0), (1, 0)),
                     'constant', constant_values=bos_id).astype('i')
    x_block = xp.pad(x_block, ((0, 0), (1, 0)),
                     'constant', constant_values=bos_id).astype('i')
    x_block = xp.pad(x_block, ((0, 0), (1, 0)),
                     'constant', constant_values=bos_id).astype('i')
    
    return x_block


class CalculateBleu(chainer.training.Extension):

    trigger = 1, 'epoch'
    priority = chainer.training.PRIORITY_WRITER

    def __init__(
            self, model, test_data, key, batch=50, device=-1, max_length=50):
        self.model = model
        self.test_data = test_data
        self.key = key
        self.batch = batch
        self.device = device
        self.max_length = max_length

    def __call__(self, trainer):
        print('## Calculate BLEU')
        with chainer.no_backprop_mode():
            with chainer.using_config('train', False):
                references = []
                hypotheses = []
                for i in range(0, len(self.test_data), self.batch):
                    sources, targets = zip(*self.test_data[i:i + self.batch])
                    references.extend([[t.tolist()] for t in targets])

                    sources = [
                        chainer.dataset.to_device(self.device, x) for x in sources]
                    ys = [y.tolist()
                          for y in self.model.translate(
                        sources, self.max_length, beam=False)]
                    # greedy generation for efficiency
                    hypotheses.extend(ys)

        bleu = bleu_score.corpus_bleu(
            references, hypotheses,
            smoothing_function=bleu_score.SmoothingFunction().method1) * 100
        print('BLEU:', bleu)
        reporter.report({self.key: bleu})


def main():
    parser = argparse.ArgumentParser(
        description='Chainer example: convolutional seq2seq')
    parser.add_argument('--batchsize', '-b', type=int, default=48,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=300,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--unit', '-u', type=int, default=512,
                        help='Number of units')
    parser.add_argument('--layer', '-l', type=int, default=6,
                        help='Number of layers')
    parser.add_argument('--head', type=int, default=8,
                        help='Number of heads in attention mechanism')
    parser.add_argument('--dropout', '-d', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--input', '-i', type=str, default='./',
                        help='Input directory')
    parser.add_argument('--source', '-s', type=str,
                        default='../dataset/Transformer_subtree_polish_test_Integrand.txt',
                        help='Filename of train data for source Integrand')
    parser.add_argument('--target', '-t', type=str,
                        default='../dataset/Transformer_subtree_polish_test_Primitive.txt',
                        help='Filename of train data for target Primitive function')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--source_vocab_list', type=str, default='../dataset/Transformer_subtree_polish_Integrand_vocab.pickle',
                        help='Vocabulary list of source language')
    parser.add_argument('--source-vocab', type=int, default=40000,
                        help='Vocabulary size of source language')
    parser.add_argument('--target_vocab_list', type=str, default='../dataset/Transformer_subtree_polish_Primitive_vocab.pickle',
                        help='Vocabulary list of target language')
    parser.add_argument('--target-vocab', type=int, default=40000,
                        help='Vocabulary size of target language')
    parser.add_argument('--learned_model',type=str, default='../model/Transformer_subtree_polish_best_model',
                        help='learned model to inference test data')
    parser.add_argument('--no-bleu', '-no-bleu', action='store_true',
                        help='Skip BLEU calculation')
    parser.add_argument('--use-label-smoothing', action='store_true',
                        help='Use label smoothing for cross entropy')
    parser.add_argument('--embed-position', action='store_true',
                        help='Use position embedding rather than sinusoid')
    parser.add_argument('--use-fixed-lr', action='store_true',
                        help='Use fixed learning rate rather than the ' +
                             'annealing proposed in the paper')
    parser.add_argument('--integrated_model', action='store_true',help='use as a component of Integrated All models')
    
    args = parser.parse_args()
    #print(json.dumps(args.__dict__, indent=4))

    # Check file
    en_path = os.path.join(args.input, args.source)
    #en_path_test = os.path.join(args.input, args.sourcetest)
    source_vocab = []

    with open(args.source_vocab_list,'rb') as f:
        source_vocab = pickle.load(f)
    
    source_data = preprocess.make_dataset(en_path, source_vocab)
    fr_path = os.path.join(args.input, args.target)
    #fr_path_test = os.path.join(args.input, args.targettest)
    target_vocab = []

    with open(args.target_vocab_list,'rb') as f:
        target_vocab = pickle.load(f)
    
    target_data = preprocess.make_dataset(fr_path, target_vocab)
    assert len(source_data) == len(target_data)
    #print('Original total data size: %d' % len(source_data))
    test_data = [(s, t)
                  for s, t in six.moves.zip(source_data, target_data)]
                  
    #print('Filtered training data size: %d' % len(train_data))
    #dataset = chainer.datasets.get_cross_validation_datasets_random(data, 5, seed=1984)
    #dataset = chainer.datasets.SubDataset(data,0,len(data))      ##########異なるデータセットを用いたい場合はここをコメントアウト
    #k_fold_for_test = 0
    #train_and_validation = dataset[k_fold_for_test][0]
    

    #with open('Transformer_subtree_polish_pickle_dataset.pickle','wb') as f:
    #    pickle.dump(test_data,f)
    
    #divide_train_and_validation = chainer.datasets.get_cross_validation_datasets_random(train_and_validation, 10, seed=1984)
    #k_fold_for_train_valid = 6
    #train_data = divide_train_and_validation[k_fold_for_train_valid][0]
    #test_data = divide_train_and_validation[k_fold_for_train_valid][1] #自分のモデルでいうvalidation_data
    #print('Filtered training data size: %d' % len(train_data))
    #print('Filtered validation data size: %d' % len(test_data))
    """
    en_path = os.path.join(args.input, args.source_valid)
    source_data = preprocess.make_dataset(en_path, source_vocab)
    fr_path = os.path.join(args.input, args.target_valid)
    target_data = preprocess.make_dataset(fr_path, target_vocab)
    assert len(source_data) == len(target_data)
    test_data = [(s, t) for s, t in six.moves.zip(source_data, target_data)
                 if 0 < len(s) and 0 < len(t)]
    """
    source_ids = {word: index for index, word in enumerate(source_vocab)}
    target_ids = {word: index for index, word in enumerate(target_vocab)}

    target_words = {i: w for w, i in target_ids.items()}
    source_words = {i: w for w, i in source_ids.items()}

    # Define Model
    model = net_check_visualization_all_layer.Transformer(
        args.layer,
        min(len(source_ids), len(source_words)),
        min(len(target_ids), len(target_words)),
        args.unit,
        h=args.head,
        dropout=args.dropout,
        max_length=500,
        use_label_smoothing=args.use_label_smoothing,
        embed_position=args.embed_position)
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu(args.gpu)

    # Setup Optimizer
    optimizer = chainer.optimizers.Adam(
        alpha=5e-5,
        beta1=0.9,
        beta2=0.98,
        eps=1e-9
    )
    optimizer.setup(model)
    
    #load model
    
    #model_epoch_num = 108
    if args.learned_model == "../model/Transformer_subtree_polish_best_model" or  args.learned_model == "../Transformer_subtree/model/Transformer_subtree_polish_best_model":
        print('=== Transformer subtree polish model ===')
    if args.learned_model == "../model/Transformer_subtree_IRPP_best_model" or args.learned_model == "../Transformer_subtree/model/Transformer_subtree_IRPP_best_model":
        print('=== Transformer subtree IRPP model ===')
    print(f"Loading:{args.learned_model}")
    ## for wrong dataset 
    #chainer.serializers.load_npz('/lustre/gn54/i95006/Transformer/attention_is_all_you_need/combine_onigiri_model/Transformer_combine_onigiri_test_fold0_train_valid_fold'+str(k_fold_for_train_valid)+'_first_try_epoch300_fix_output/best_model_valid_loss.npz',model,path='')
    chainer.serializers.load_npz(args.learned_model,model,path='')
    # Setup Trainer
    #train_iter = chainer.iterators.SerialIterator(train_data, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test_data, args.batchsize,
                                                 repeat=False, shuffle=False)
    #iter_per_epoch = len(train_data) // args.batchsize
    #print('Number of iter/epoch =', iter_per_epoch)

    #updater = training.StandardUpdater(
    #    train_iter, optimizer,
    #    converter=seq2seq_pad_concat_convert, device=args.gpu)

    #trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
    #print("trainer_setup")
    # If you want to change a logging interval, change this number
    """
    log_trigger = (min(200, iter_per_epoch // 2), 'iteration')

    def floor_step(trigger):
        floored = trigger[0] - trigger[0] % log_trigger[0]
        if floored <= 0:
            floored = trigger[0]
        return (floored, trigger[1])

    # Validation every half epoch
    eval_trigger = floor_step((iter_per_epoch // 2, 'iteration'))
    record_trigger = training.triggers.MinValueTrigger(
        'val/main/perp', eval_trigger)  
    
    record_trigger_valid_loss = training.triggers.MinValueTrigger(
        'val/main/loss', eval_trigger)

    trainer.extend(extensions.snapshot(),trigger=(1, 'epoch'))

    evaluator = extensions.Evaluator(
        test_iter, model,
        converter=seq2seq_pad_concat_convert, device=args.gpu)
    evaluator.default_name = 'val'
    trainer.extend(evaluator, trigger=eval_trigger)
    """
    # Use Vaswan's magical rule of learning rate(Eq. 3 in the paper)
    # But, the hyperparamter in the paper seems to work well
    # only with a large batchsize.
    # If you run on popular setup (e.g. size=48 on 1 GPU),
    # you may have to change the hyperparamter.
    # I scaled learning rate by 0.5 consistently.
    # ("scale" is always multiplied to learning rate.)

    # If you use a shallow layer network (<=2),
    # you may not have to change it from the paper setting.
    """
    if not args.use_fixed_lr:
        trainer.extend(
            # VaswaniRule('alpha', d=args.unit, warmup_steps=4000, scale=1.),
            # VaswaniRule('alpha', d=args.unit, warmup_steps=32000, scale=1.),
            # VaswaniRule('alpha', d=args.unit, warmup_steps=4000, scale=0.5),
            # VaswaniRule('alpha', d=args.unit, warmup_steps=16000, scale=1.),
            VaswaniRule('alpha', d=args.unit, warmup_steps=64000, scale=1.),
            trigger=(1, 'iteration'))
    observe_alpha = extensions.observe_value(
        'alpha',
        lambda trainer: trainer.updater.get_optimizer('main').alpha)
    trainer.extend(
        observe_alpha,
        trigger=(1, 'iteration'))

    # Only if a model gets best validation score,
    # save (overwrite) the model
    trainer.extend(extensions.snapshot_object(
        model, 'best_model.npz'),
        trigger=record_trigger)
    
    trainer.extend(extensions.snapshot_object(
        model, 'best_model_valid_loss.npz'),
        trigger=record_trigger_valid_loss)
    """
    #trainer.extend(extensions.snapshot(optimizer, 'latest optimizer.npz'), trigger=(1, 'epoch'))

    def translate_one(source, target):
        words = preprocess.split_sentence(source)    
        print("Integrand(Input):" + ' '.join(words))
        #start = time.time()       ###########time add ordering result
        x = model.xp.array(
            [source_ids.get(w, 1) for w in words], 'i')
        #ys = model.translate([x], beam=5)[0]
        start = time.time()
        ys = model.translate([x], beam=False)[0]
        elapsed_time = time.time() - start  
        """
        attention_weight_all_eq_l1.append(model.one_equation_attention_weight_l1)
        attention_weight_all_eq_l2.append(model.one_equation_attention_weight_l2)
        attention_weight_all_eq_l3.append(model.one_equation_attention_weight_l3)
        attention_weight_all_eq_l4.append(model.one_equation_attention_weight_l4)
        attention_weight_all_eq_l5.append(model.one_equation_attention_weight_l5)
        attention_weight_all_eq_l6.append(model.one_equation_attention_weight_l6)
        """
        words = [target_words[y] for y in ys]
        #elapsed_time = time.time() - start      ########time add ordering result 
        print('Primitive(Output):' + ' '.join(words))
        result = ' '.join(words)
        print('Correct Answer:'+target)
        print("elapsed_time:"+str(elapsed_time))
        if result == target:
            #print("correct")
            return 1
        else:
            #print("wrong")
            return 0
    
    count_correct_eq = 0
    count_correct_eq_one_step_before = 0 
    count_eq = 0
    wrong_eq_list = []
    """
    attention_weight_all_eq_l1 = []
    attention_weight_all_eq_l2 = []
    attention_weight_all_eq_l3 = []
    attention_weight_all_eq_l4 = []
    attention_weight_all_eq_l5 = []
    attention_weight_all_eq_l6 = []
    """
    for eq_num in range(len(test_data)):
        source, target = test_data[eq_num]
        print("-----")
        if not args.integrated_model:
            print("eq_num:",str(eq_num))
        source = ' '.join([source_words[i] for i in source])
        target = ' '.join([target_words[i] for i in target])
        count_correct_eq += translate_one(source, target)
        if count_correct_eq == count_correct_eq_one_step_before:
            wrong_eq_list.append(count_eq)
            #print("wrong_eq:"+str(count_eq))
            print("Wrong")
        else:
            #print("correct_eq:"+str(count_eq))
            print("Correct!")
        count_correct_eq_one_step_before = count_correct_eq
        count_eq+=1

    """
    with open('self_attention_weight_subtree_l1.pickle','wb') as f:
        pickle.dump(attention_weight_all_eq_l1, f)
    with open('self_attention_weight_subtree_l2.pickle','wb') as f:
        pickle.dump(attention_weight_all_eq_l2, f)
    with open('self_attention_weight_subtree_l3.pickle','wb') as f:
        pickle.dump(attention_weight_all_eq_l3, f)
    with open('self_attention_weight_subtree_l4.pickle','wb') as f:
        pickle.dump(attention_weight_all_eq_l4, f)
    with open('self_attention_weight_subtree_l5.pickle','wb') as f:
        pickle.dump(attention_weight_all_eq_l5, f)
    with open('self_attention_weight_subtree_l6.pickle','wb') as f:
        pickle.dump(attention_weight_all_eq_l6, f)
    """
    
    #print('epoch:'+str(trainer.updater.epoch))
    #print('iteration:'+str(trainer.updater.iteration))
    if not args.integrated_model:
        print("---Result Summary---")
        print('Total correct equation num:'+str(count_correct_eq))
        print('len(test):'+str(len(test_data)))
        print('Complete Correct Answer Rate:{}%'.format(str(100*count_correct_eq/len(test_data))))
    #print('wrong_eq_list:'+str(wrong_eq_list))

if __name__ == '__main__':
    main()
