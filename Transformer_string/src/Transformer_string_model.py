# encoding: utf-8
import argparse
import json
import os.path
import time
import pickle
import datetime 

from nltk.translate import bleu_score
import numpy
import six

import chainer
from chainer import cuda
from chainer.dataset import convert
from chainer import reporter
from chainer import training
from chainer.training import extensions

import preprocess
import net_check_visualization_all_layer_decoder

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
    y_block = convert.concat_examples(y_seqs, device, padding=-1)
    xp = cuda.get_array_module(x_block)

    # The paper did not mention eos
    # add eos
    x_block = xp.pad(x_block, ((0, 0), (0, 1)),
                     'constant', constant_values=-1)
    for i_batch, seq in enumerate(x_seqs):
        x_block[i_batch, len(seq)] = eos_id
    x_block = xp.pad(x_block, ((0, 0), (1, 0)),
                     'constant', constant_values=bos_id)

    y_out_block = xp.pad(y_block, ((0, 0), (0, 1)),
                         'constant', constant_values=-1)
    for i_batch, seq in enumerate(y_seqs):
        y_out_block[i_batch, len(seq)] = eos_id

    y_in_block = xp.pad(y_block, ((0, 0), (1, 0)),
                        'constant', constant_values=bos_id)
    return (x_block, y_in_block, y_out_block)


def source_pad_concat_convert(x_seqs, device, eos_id=0, bos_id=2):
    x_block = convert.concat_examples(x_seqs, device, padding=-1)
    xp = cuda.get_array_module(x_block)

    # add eos
    x_block = xp.pad(x_block, ((0, 0), (0, 1)),
                     'constant', constant_values=-1)
    for i_batch, seq in enumerate(x_seqs):
        x_block[i_batch, len(seq)] = eos_id
    x_block = xp.pad(x_block, ((0, 0), (1, 0)),
                     'constant', constant_values=bos_id)
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
    parser.add_argument('--batchsize', '-b', type=int, default=256,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=100,
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
                        default='../dataset/Transformer_string_polish_test_Integrand.txt',
                        help='Filename of train data for source language')
    parser.add_argument('--target', '-t', type=str,
                        default='../dataset/Transformer_string_polish_test_Primitive.txt',
                        help='Filename of train data for target language')
    parser.add_argument('--source_vocab_list', type=str, default='../dataset/Transformer_string_polish_Integrand_vocab.pickle',
                        help='Vocabulary list of source language')
    parser.add_argument('--target_vocab_list', type=str, default='../dataset/Transformer_string_polish_Primitive_vocab.pickle',
                        help='Vocabulary list of target language')
    parser.add_argument('--learned_model',type=str, default='../model/Transformer_string_polish_best_model',
                        help='learned model to inference test data')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--outputfile','-of', default = str(datetime.datetime.today())+'.log', 
                        help='output_file_name')
    parser.add_argument('--source-vocab', type=int, default=40000,
                        help='Vocabulary size of source language')
    parser.add_argument('--target-vocab', type=int, default=40000,
                        help='Vocabulary size of target language')
    parser.add_argument('--no-bleu', '-no-bleu', action='store_true',
                        help='Skip BLEU calculation')
    parser.add_argument('--use-label-smoothing', action='store_true',
                        help='Use label smoothing for cross entropy')
    parser.add_argument('--embed-position', action='store_true',
                        help='Use position embedding rather than sinusoid')
    parser.add_argument('--use-fixed-lr', action='store_true',
                        help='Use fixed learning rate rather than the ' +
                             'annealing proposed in the paper')
    args = parser.parse_args()
    print(json.dumps(args.__dict__, indent=4))

    # Check file
    en_path = os.path.join(args.input, args.source)
    source_vocab = []
    with open(args.source_vocab_list,'rb') as f:
        source_vocab = pickle.load(f)
    source_data = preprocess.make_dataset(en_path, source_vocab)
    fr_path = os.path.join(args.input, args.target)
    target_vocab = []
    with open(args.target_vocab_list,'rb') as f:
        target_vocab = pickle.load(f)
    target_data = preprocess.make_dataset(fr_path, target_vocab)
    assert len(source_data) == len(target_data)
    print('Original total data size: %d' % len(source_data))
    test_data = [(s, t)
                 for s, t in six.moves.zip(source_data, target_data)]
    print('test data size: %d' % len(test_data))

    source_ids = {word: index for index, word in enumerate(source_vocab)}
    target_ids = {word: index for index, word in enumerate(target_vocab)}

    target_words = {i: w for w, i in target_ids.items()}
    source_words = {i: w for w, i in source_ids.items()}

    # Define Model
    model = net_check_visualization_all_layer_decoder.Transformer(
        args.layer,
        min(len(source_ids), len(source_words)),
        min(len(target_ids), len(target_words)),
        args.unit,
        h=args.head,
        dropout=args.dropout,
        max_length=512,
        use_label_smoothing=args.use_label_smoothing,
        embed_position=args.embed_position)
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu(args.gpu)

    # Setup Optimizer
    optimizer = chainer.optimizers.Adam(
        alpha=1e-4,
        #beta1=0.9,
        #beta2=0.98,
        #eps=1e-9
    )
    optimizer.setup(model)

    
    chainer.serializers.load_npz(args.learned_model,model,path='')
    
    # Setup Trainer
    test_iter = chainer.iterators.SerialIterator(test_data, args.batchsize,
                                                 repeat=False, shuffle=False)

    def translate_one(source, target):
        words = preprocess.split_sentence(source)
        #with open('result_eq_accuracy_polish_prework_epoch300_test_fold_0_inference_test_train_valid_fold'+str(k_fold_for_train_valid)+'_fix_primitive_beam1_visualization_add_decoder_12122_dataset.txt','a') as f:
        with open(args.outputfile, 'a') as f:
            print('# source : ' + ' '.join(words))
            start = time.time()
        x = model.xp.array(
            [source_ids.get(w, 1) for w in words], 'i')
        ###beam_search###
        #ys = model.translate([x], beam=5)[0]
        ###gready_search#
        # uncomment this part if you need attention weight
         
        ys = model.translate([x], beam=1)[0]
        """
        attention_weight_all_eq_l1.append(model.one_equation_attention_weight_l1)
        attention_weight_all_eq_l2.append(model.one_equation_attention_weight_l2)
        attention_weight_all_eq_l3.append(model.one_equation_attention_weight_l3)
        attention_weight_all_eq_l4.append(model.one_equation_attention_weight_l4)
        attention_weight_all_eq_l5.append(model.one_equation_attention_weight_l5)
        attention_weight_all_eq_l6.append(model.one_equation_attention_weight_l6)
        
        attention_weight_all_eq_decoder_l1.append(model.one_equation_attention_weight_decoder_l1)
        attention_weight_all_eq_decoder_l2.append(model.one_equation_attention_weight_decoder_l2)
        attention_weight_all_eq_decoder_l3.append(model.one_equation_attention_weight_decoder_l3)
        attention_weight_all_eq_decoder_l4.append(model.one_equation_attention_weight_decoder_l4)
        attention_weight_all_eq_decoder_l5.append(model.one_equation_attention_weight_decoder_l5)
        attention_weight_all_eq_decoder_l6.append(model.one_equation_attention_weight_decoder_l6)
        """ 
        words = [target_words[y] for y in ys]
        elapsed_time = time.time() - start
        print('#  result : ' + ' '.join(words))
        result = ' '.join(words)
        print('#  expect : ' + target)
        #with open('result_eq_accuracy_polish_prework_epoch300_test_fold_0_inference_test_train_valid_fold'+str(k_fold_for_train_valid)+'_fix_primitive_beam1_visualization_add_decoder_12122_dataset.txt','a') as f:
        with open(args.outputfile, 'a') as f:
            print("result:"+' '.join(words))
            print("expect:"+ target)
            print("time_elapsed:" + str(elapsed_time))
            
        if result == target:
            return 1
        else:
            return 0

    count_correct_eq = 0
    count_correct_eq_one_step_before = 0
    count_eq = 0
    wrong_eq_list = []
    attention_weight_all_eq_l1 = []
    attention_weight_all_eq_l2 = []
    attention_weight_all_eq_l3 = []
    attention_weight_all_eq_l4 = []
    attention_weight_all_eq_l5 = []
    attention_weight_all_eq_l6 = []
    attention_weight_all_eq_decoder_l1 = []
    attention_weight_all_eq_decoder_l2 = []
    attention_weight_all_eq_decoder_l3 = []
    attention_weight_all_eq_decoder_l4 = []
    attention_weight_all_eq_decoder_l5 = []
    attention_weight_all_eq_decoder_l6 = []
    
    for eq_num in range(len(test_data)):
        source, target = test_data[eq_num]
        source = ' '.join([source_words[i] for i in source])
        target = ' '.join([target_words[i] for i in target])
        count_correct_eq += translate_one(source, target)
        if count_correct_eq == count_correct_eq_one_step_before:
            wrong_eq_list.append(count_eq)
            with open(args.outputfile, 'a') as f:
                print("wrong_eq:"+str(count_eq))
        else:
            with open(args.outputfile, 'a') as f:
                print("correct_eq:"+str(count_eq))
        count_correct_eq_one_step_before = count_correct_eq
        count_eq+=1

    # if you need attention weight, uncomment this part
    """    
    with open('self_attention_weight_l1'+str(datetime.datetime.today())+'.pickle','wb') as f:
        pickle.dump(attention_weight_all_eq_l1, f)
    with open('self_attention_weight_l2'+str(datetime.datetime.today())+'.pickle','wb') as f:
        pickle.dump(attention_weight_all_eq_l2, f)
    with open('self_attention_weight_l3'+str(datetime.datetime.today())+'.pickle','wb') as f:
        pickle.dump(attention_weight_all_eq_l3, f)
    with open('self_attention_weight_l4'+str(datetime.datetime.today())+'.pickle','wb') as f:
        pickle.dump(attention_weight_all_eq_l4, f)
    with open('self_attention_weight_l5'+str(datetime.datetime.today())+'.pickle','wb') as f:
        pickle.dump(attention_weight_all_eq_l5, f)
    with open('self_attention_weight_l6'+str(datetime.datetime.today())+'.pickle','wb') as f:
        pickle.dump(attention_weight_all_eq_l6, f)
    
    with open('self_attention_weight_decoder_l1'+str(datetime.datetime.today())+'.pickle','wb') as f:
        pickle.dump(attention_weight_all_eq_decoder_l1, f)
    with open('self_attention_weight_decoder_l2'+str(datetime.datetime.today())+'.pickle','wb') as f:
        pickle.dump(attention_weight_all_eq_decoder_l2, f)
    with open('self_attention_weight_decoder_l3'+str(datetime.datetime.today())+'.pickle','wb') as f:
        pickle.dump(attention_weight_all_eq_decoder_l3, f)
    with open('self_attention_weight_decoder_l4'+str(datetime.datetime.today())+'.pickle','wb') as f:
        pickle.dump(attention_weight_all_eq_decoder_l4, f)
    with open('self_attention_weight_decoder_l5'+str(datetime.datetime.today())+'.pickle','wb') as f:
        pickle.dump(attention_weight_all_eq_decoder_l5, f)
    with open('self_attention_weight_decoder_l6'+str(datetime.datetime.today())+'.pickle','wb') as f:
        pickle.dump(attention_weight_all_eq_decoder_l6, f)
    """
    #with open('result_eq_accuracy_polish_prework_epoch300_test_fold_0_inference_test_train_valid_fold'+str(k_fold_for_train_valid)+'_fix_primitive_beam1_visualization_add_decoder_5times_12122_dataset.txt','a') as f:
    with open(args.outputfile, 'a') as f:
        print('count_correct_eq:'+str(count_correct_eq))
        print('test_data:'+str(len(test_data)))
        print('exact_same_eq_accuracy:'+str(count_correct_eq/len(test_data))+'%\n',file=f)
        print('wrong_eq_list:'+str(wrong_eq_list))

if __name__ == '__main__':
    main()
