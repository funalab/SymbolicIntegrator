#!/usr/bin/env bash

#export PYENV_ROOT="$HOME/.pyenv"
#export PATH="$PYENV_ROOT/bin:$PATH"
#eval "$(pyenv init -)"

myname=`basename $0`
# usage
usage () {
  cat<<_EOU_
$myname: Integrate eight models and perform inference
Usage: $myname [--gpu GPUID | -g GPUID] test_data.txt
Options
    --gpu GPUID, -g GPUID    specify GPU ID (default: -1 (execute on CPU))
    --help, -h               show this message
_EOU_
  exit 0
}

if [ $# = 0 ]; then
    usage
fi

# parse args
gpuid=-1
while [ "$1" != "" ];
do
  case "$1" in
    -g | --gpu) shift; gpuid="$1" ;;
    -h | --help) usage ;;
    *)  break ;;
  esac
  shift
done

declare -i COUNTER
declare -i EQ_NUM  

COUNTER=0 

EQ_NUM=$(cat $1 | wc -l)

echo "Total test equation num:"$EQ_NUM
   
for ((COUNTER=1;COUNTER<=EQ_NUM;COUNTER++)); do

    echo "----------eq_num:"`expr $COUNTER - 1`"----------"
    #LSTM dataset 
    sed -n "$COUNTER"p ../LSTM_string/dataset/LSTM_string_Polish_test_Integrand_first_10eq.txt > one_eq_LSTM_string_Polish_test_Integrand.txt   
    sed -n "$COUNTER"p ../LSTM_string/dataset/LSTM_string_Polish_test_Primitive_first_10eq.txt > one_eq_LSTM_string_Polish_test_Primitive.txt
    sed -n "$COUNTER"p ../LSTM_string/dataset/LSTM_string_IRPP_test_Integrand_first_10eq.txt > one_eq_LSTM_string_IRPP_test_Integrand.txt
    sed -n "$COUNTER"p ../LSTM_string/dataset/LSTM_string_IRPP_test_Primitive_first_10eq.txt > one_eq_LSTM_string_IRPP_test_Primitive.txt
    sed -n "$COUNTER"p ../LSTM_subtree/dataset/LSTM_subtree_polish_test_Integrand_first_10eq.txt > one_eq_LSTM_subtree_polish_test_Integrand.txt 
    sed -n "$COUNTER"p ../LSTM_subtree/dataset/LSTM_subtree_polish_test_Primitive_first_10eq.txt > one_eq_LSTM_subtree_polish_test_Primitive.txt 
    sed -n "$COUNTER"p ../LSTM_subtree/dataset/LSTM_subtree_IRPP_test_Integrand_first_10eq.txt > one_eq_LSTM_subtree_IRPP_test_Integrand.txt
    sed -n "$COUNTER"p ../LSTM_subtree/dataset/LSTM_subtree_IRPP_test_Primitive_first_10eq.txt > one_eq_LSTM_subtree_IRPP_test_Primitive.txt

    #Transformer dataset
    sed -n "$COUNTER"p ../Transformer_string/dataset/Transformer_string_polish_test_Integrand_first_10eq.txt > one_eq_Transformer_string_polish_test_Integrand.txt
    sed -n "$COUNTER"p ../Transformer_string/dataset/Transformer_string_polish_test_Primitive_first_10eq.txt > one_eq_Transformer_string_polish_test_Primitive.txt
    sed -n "$COUNTER"p ../Transformer_string/dataset/Transformer_string_IRPP_test_Integrand_first_10eq.txt > one_eq_Transformer_string_IRPP_test_Integrand.txt
    sed -n "$COUNTER"p ../Transformer_string/dataset/Transformer_string_IRPP_test_Primitive_first_10eq.txt > one_eq_Transformer_string_IRPP_test_Primitive.txt
    sed -n "$COUNTER"p ../Transformer_subtree/dataset/Transformer_subtree_polish_test_Integrand_first_10eq.txt > one_eq_Transformer_subtree_polish_test_Integrand.txt
    sed -n "$COUNTER"p ../Transformer_subtree/dataset/Transformer_subtree_polish_test_Primitive_first_10eq.txt > one_eq_Transformer_subtree_polish_test_Primitive.txt
    sed -n "$COUNTER"p ../Transformer_subtree/dataset/Transformer_subtree_IRPP_test_Integrand_first_10eq.txt > one_eq_Transformer_subtree_IRPP_test_Integrand.txt 
    sed -n "$COUNTER"p ../Transformer_subtree/dataset/Transformer_subtree_IRPP_test_Primitive_first_10eq.txt > one_eq_Transformer_subtree_IRPP_test_Primitive.txt

    #LSTM string polish model 
    python ../LSTM_string/src/LSTM_string_polish_model.py -g $gpuid --Integrand_dataset one_eq_LSTM_string_Polish_test_Integrand.txt --Primitive_dataset one_eq_LSTM_string_Polish_test_Primitive.txt --token_dataset ../LSTM_string/dataset/LSTM_string_polish_token.txt --study_name MLP_cupy_successiveHalvingPruner_epoch30_complete_correct_2nd_try_cross_valid --learned_model ../LSTM_string/model/LSTM_string_polish_best_model --integrated_model 

    #LSTM subtree polish model
    python ../LSTM_subtree/src/LSTM_subtree_model.py -g $gpuid --token_dataset ../LSTM_subtree/dataset/LSTM_subtree_polish_token.txt --Integrand_dataset one_eq_LSTM_subtree_polish_test_Integrand.txt --Primitive_dataset one_eq_LSTM_subtree_polish_test_Primitive.txt --study_name MLP_cupy_MedianPruner_epoch30_subtree_complete_correct_continue --learned_model ../LSTM_subtree/model/LSTM_subtree_polish_best_model --integrated_model

    #Transformer subtree IRPP model
    python ../Transformer_subtree/src/Transformer_subtree_model.py -g $gpuid --source one_eq_Transformer_subtree_polish_test_Integrand.txt --target one_eq_Transformer_subtree_polish_test_Primitive.txt --source_vocab_list ../Transformer_subtree/dataset/Transformer_subtree_polish_Integrand_vocab.pickle --target_vocab_list ../Transformer_subtree/dataset/Transformer_subtree_polish_Primitive_vocab.pickle --learned_model ../Transformer_subtree/model/Transformer_subtree_polish_best_model --integrated_model
    
    #LSTM string IRPP model
    python ../LSTM_string/src/LSTM_string_IRPP_model.py -g $gpuid --Integrand_dataset one_eq_LSTM_string_IRPP_test_Integrand.txt --Primitive_dataset one_eq_LSTM_string_IRPP_test_Primitive.txt --token_dataset ../LSTM_string/dataset/LSTM_string_polish_token.txt --study_name MLP_cupy_MedianPruner_epoch30_integrand_reverse_polish_Primitive_polish_third_try_memory_edited_v102_continue_untilepoch200 --learned_model ../LSTM_string/model/LSTM_string_IRPP_best_model --integrated_model
    
    #LSTM subtree IRPP model
    python ../LSTM_subtree/src/LSTM_subtree_model.py -g $gpuid --token_dataset ../LSTM_subtree/dataset/LSTM_subtree_polish_token.txt --Integrand_dataset one_eq_LSTM_subtree_IRPP_test_Integrand.txt --Primitive_dataset one_eq_LSTM_subtree_IRPP_test_Primitive.txt --study_name MLP_cupy_MedianPruner_epoch30_subtree_Integrand_reverse_polish_Primitive_polish_continue --learned_model ../LSTM_subtree/model/LSTM_subtree_IRPP_best_model --integrated_model

    #Transformer subtree polish model
    python ../Transformer_subtree/src/Transformer_subtree_model.py -g $gpuid --source one_eq_Transformer_subtree_polish_test_Integrand.txt --target one_eq_Transformer_subtree_polish_test_Primitive.txt --source_vocab_list ../Transformer_subtree/dataset/Transformer_subtree_IRPP_Integrand_vocab.pickle --target_vocab_list ../Transformer_subtree/dataset/Transformer_subtree_IRPP_Primitive_vocab.pickle --learned_model ../Transformer_subtree/model/Transformer_subtree_polish_best_model --integrated_model
    
    #Transformer string polish model
    python ../Transformer_string/src/Transformer_string_model.py -g $gpuid --source one_eq_Transformer_string_polish_test_Integrand.txt --target one_eq_Transformer_string_polish_test_Primitive.txt --source_vocab_list ../Transformer_string/dataset/Transformer_string_polish_Integrand_vocab.pickle --target_vocab_list ../Transformer_string/dataset/Transformer_string_polish_Primitive_vocab.pickle --learned_model ../Transformer_string/model/Transformer_string_polish_best_model --integrated_model
    
    #Transformer string IRPP model
    python ../Transformer_string/src/Transformer_string_model.py -g $gpuid --source one_eq_Transformer_string_IRPP_test_Integrand.txt --target one_eq_Transformer_string_IRPP_test_Primitive.txt --source_vocab_list ../Transformer_string/dataset/Transformer_string_IRPP_source_vocab.pickle --target_vocab_list ../Transformer_string/dataset/Transformer_string_IRPP_target_vocab.pickle --learned_model ../Transformer_string/model/Transformer_string_IRPP_best_model --integrated_model
    
done

echo "FINISHED ALL TEST DATA!"

rm one_eq_*
