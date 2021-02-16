# Symbolic Integration Model 

This is the code for Symbolic Integration by integrating learning models with different strengths and weaknesses.
This project is carried out in cooperation with [Funahashi lab at Keio University](https://fun.bio.keio.ac.jp/).

## Overview

We developed eight learning models for symbolic integration that were combinations of input-output schemes(string/subtree, polish/IRPP) and learning models (LSTM/Transformer).

## installation

## Requirement
- Python 3.6.0+
- [Chainer](https://github.com/chainer/chainer/) 4.0.0+
- [numpy](https://github.com/numpy/numpy) 1.12.1+
- [cupy](https://github.com/cupy/cupy) 1.0.0+ (if using gpu)
- nltk
- optuna 1.3.0
- and their dependencies

## QuickStart

1. Download learned model and dataset.

- On Linux:

- On macOS:

2. Inference on test dataset.

   The best learned model for LSTM string polish model is `SymbolicIntegrationModel/LSTM_string/models/LSTM_string_polish_best_model`.
   To verify the accuracy of the learned model using test data in `SymbolicIntegrationModel/LSTM_string/dataset/LSTM_string_Polish_test_Integrand.txt`, run the following:
   
   ```sh
   % cd ./SymbolicIntegrationModel/LSTM_string/src
   % python LSTM_string_polish_model.py (--gpu 0 --token_dataset ../dataset/LSTM_string_polish_token.txt --Integrand_dataset ../dataset/LSTM_string_Polish_test_Integrand.txt --Primitive_dataset ../dataset/LSTM_string_Polish_test_Primitive.txt --study_name MLP_cupy_successiveHalvingPruner_epoch30_complete_correct_2nd_try_cross_valid --learned_model ../models/LSTM_string_polish_best_model)   
   ```

   The best learned model for LSTM string IRPP model is `SymbolicIntegrationModel/LSTM_string/models/LSTM_string_IRPP_best_model`.
   To verify the accuracy of the learned model using test data in `SymbolicIntegrationModel/LSTM_string/dataset/LSTM_string_IRPP_test_Integrand.txt`, run the following:

   ```sh
   % cd ./SymbolicIntegrationModel/LSTM_string/src
   % python LSTM_string_IRPP_model.py (--gpu 0 --token_dataset ../dataset/LSTM_string_polish_token.txt --Integrand_dataset ../dataset/LSTM_string_IRPP_test_Integrand.txt --Primitive_dataset ../dataset/LSTM_string_IRPP_test_Primitive.txt --study_name MLP_cupy_MedianPruner_epoch30_integrand_reverse_polish_Primitive_polish_third_try_memory_edited_v102_continue_untilepoch200 --learned_model ../models/LSTM_string_IRPP_best_model )
   ```

   The best learned model for LSTM subtree polish model is `SymbolicIntegrationModel/LSTM_subtree/model/LSTM_subtree_polish_best_model`
   To verify the accuracy of the learned model using test data in `SymbolicIntegrationModel/LSTM_subtree/dataset/LSTM_subtree_polish_test_Integrand.txt`, run the following:

   ```sh
   % cd ./SymbolicIntegrationModel/LSTM_subtree/src
   % python LSTM_subtree_model.py --gpu 0 --token_dataset ../dataset/LSTM_subtree_polish_token.txt --Integrand_dataset ../dataset/LSTM_subtree_polish_test_Integrand.txt --Primitive_dataset ../dataset/LSTM_subtree_polish_test_Primitive.txt --study_name MLP_cupy_MedianPruner_epoch30_subtree_complete_correct_continue --learned_model ../model/LSTM_subtree_polish_best_model 
   ```
   
   The best learned model for LSTM subtree IRPP model is `SymbolicIntegrationModel/LSTM_subtree/model/LSTM_subtree_IRPP_best_model` 
   To verify the accuracy of the learned model using test data in `SymbolicIntegrationModel/LSTM_subtree/dataset/LSTM_subtree_IRPP_test_Integrand.txt`, run the following:

   ```sh
   % cd ./SymbolicIntegrationModel/LSTM_subtree/src
   % python LSTM_subtree_model.py --gpu 0 --token_dataset ../dataset/LSTM_subtree_polish_token.txt --Integrand_dataset ../dataset/LSTM_subtree_IRPP_test_Integrand.txt --Primitive_dataset ../dataset/LSTM_subtree_IRPP_test_Primitive.txt --study_name MLP_cupy_MedianPruner_epoch30_subtree_Integrand_reverse_polish_Primitive_polish_continue.db --learned_model ../model/LSTM_subtree_IRPP_best_model
   ```

   The best learned model for Transformer string polish model is `SymbolicIntegrationModel/Transformer_string/model/Transformer_string_polish_best_model`
   To verify the accuracy of the learned model using test data in `SymbolicIntegrationModel/Transformer_string/dataset/Transformer_string_polish_test_Integrand.txt`, run the following:

   ```sh
   % cd ./SymbolicIntegrationModel/Transformer_string/src
   % python Transformer_string_model.py -g 0 --source ../dataset/Transformer_string_polish_test_Integrand.txt --target ../dataset/Transformer_string_polish_test_Primitive.txt --source_vocab_list ../dataset/Transformer_string_polish_Integrand_vocab.pickle --target_vocab_list ../dataset/Transformer_string_polish_Primitive_vocab.pickle --learned_model ../model/Transformer_string_polish_best_model 
   ```

   The best learned model for Transformer string IRPP model is `SymbolicIntegrationModel/Transformer_string/model/Transformer_string_IRPP_best_model`
   To verify the accuracy of the learned model using test data in `SymbolicIntegrationModel/Transformer_string/dataset/Transformer_string_IRPP_test_Integrand.txt`, run the following:
 
   ```sh
   % cd ./SymbolicIntegrationModel/Transformer_string/src
   % python Transformer_string_model.py -g 0 --source ../dataset/Transformer_string_IRPP_test_Integrand.txt --target ../dataset/Transformer_string_IRPP_test_Primitive.txt --source_vocab_list ../dataset/Transformer_string_IRPP_source_vocab.pickle --target_vocab_list ../dataset/Transformer_string_IRPP_target_vocab.pickle --learned_model ../model/Transformer_string_IRPP_best_model
   ```
   The best learned model for Transformer subtree polish model is `SymbolicIntegrationModel/Transformer_subtree/model/Transformer_subtree_polish_best_model`
   To verify the accuracy of the learned model using test data in `SymbolicIntegrationModel/Transformer_subtree/dataset/Transformer_subtree_polish_test_Integrand.txt`,run the following:

   ```sh
   % cd ./SymbolicIntegrationModel/Transformer_subtree/src
   % python Transformer_subtree_model.py -g 0 --source ../dataset/Transformer_subtree_polish_test_Integrand.txt --target ../dataset/Transformer_subtree_polish_test_Primitive.txt --source_vocab_list ../dataset/Transformer_subtree_polish_Integrand_vocab.pickle --target_vocab_list ../dataset/Transformer_subtree_polish_Primitive_vocab.pickle --learned_model ../model/Transformer_subtree_polish_best_model
   ```
   The best learned model for Transformer subtree IRPP model is `SymbolicIntegrationModel/Transformer_subtree/model/Transformer_subtree_IRPP_best_model`
   To verify the accuracy of the learned model using test data in  `SymbolicIntegrationModel/Transformer_subtree/dataset/Transformer_subtree_IRPP_test_Integrand.txt`,run the following:

   ```sh
   % cd ./SymbolicIntegrationModel/Transformer_subtree/src
   % python Transformer_subtree_model.py -g 0 --source ../dataset/Transformer_subtree_IRPP_test_Integrand.txt  --target ../dataset/Transformer_subtree_IRPP_test_Primitive.txt --source_vocab_list ../dataset/Transformer_subtree_IRPP_Integrand_vocab.pickle --target_vocab_list ../dataset/Transformer_subtree_IRPP_Primitive_vocab.pickle --learned_model ../model/Transformer_subtree_IRPP_best_model 
   ```
   
   To verify the Integrated All Models, run the following:

   ```sh
   % cd ./SymbolicIntegrationModel/Integrated_all_model
   % /bin/zsh run.sh
   ```sh
  
   


   