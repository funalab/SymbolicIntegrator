# Symbolic Integration Model 

This is the code for Symbolic Integration by integrating learning models with different strengths and weaknesses.
This project is carried out by [Funahashi Lab. at Keio University](https://fun.bio.keio.ac.jp/).

![symbolic integration](https://gitlab.com/funalab/symbolicintegrationmodel/-/raw/images/image1.png)

## Overview

We developed following eight learning models for symbolic integration that were combinations of input-output schemes (string/subtree, polish/IRPP) and learning models (LSTM/Transformer).

- LSTM string polish model
- LSTM string IRPP model
- LSTM subtree polish model
- LSTM subtree IRPP model
- Transformer string polish model
- Transformer string IRPP model
- Transformer subtree polish model
- Transformer subtree IRPP model


## Requirement

- [Python](https://www.python.org/) 3.6.0+
- [Chainer](https://github.com/chainer/chainer/) 4.0.0+
- [NumPy](https://github.com/numpy/numpy) 1.12.1+
- [CuPy](https://github.com/cupy/cupy) 1.0.0+ (if using gpu)
- [NLTK](https://www.nltk.org/)
- [Optuna](https://github.com/optuna/optuna/releases/tag/v1.3.0) 1.3.0


## QuickStart

1. Download this repository by `git clone`.

   ```sh
   % git clone git@gitlab.com:funalab/symbolicintegrationmodel.git
   ```

2. Download learned model and dataset (3.8 GB).

   - On Linux:
     ```sh
     % cd SymbolicIntegrationModel/
     % wget https://fun.bio.keio.ac.jp/software/SymbolicIntegrationModel/SymbolicIntegrationModel.zip
     % unzip SymbolicIntegrationModel.zip
     % rm SymbolicIntegrationModel.zip
     ```
   - On macOS:
     ```sh
     % cd SymbolicIntegrationModel/
     % curl -O https://fun.bio.keio.ac.jp/software/SymbolicIntegrationModel/SymbolicIntegrationModel.zip
     % unzip SymbolicIntegrationModel.zip
     % rm SymbolicIntegrationModel.zip
     ```

3. Inference on test dataset.

   To run LSTM models, follow the commands below after change directory.   

   - LSTM string polish model
     
     The best learned model for LSTM string polish model is `SymbolicIntegrationModel/LSTM_string/model/LSTM_string_polish_best_model`.
     To verify the accuracy of the learned model using test data in `SymbolicIntegrationModel/LSTM_string/dataset/LSTM_string_Polish_test_Integrand.txt`, run the following:
   
     ```sh
     % cd LSTM_string/src
     % python LSTM_string_polish_model.py --token_dataset ../dataset/LSTM_string_polish_token.txt --Integrand_dataset ../dataset/LSTM_string_Polish_test_Integrand.txt --Primitive_dataset ../dataset/LSTM_string_Polish_test_Primitive.txt --study_name MLP_cupy_successiveHalvingPruner_epoch30_complete_correct_2nd_try_cross_valid --learned_model ../model/LSTM_string_polish_best_model [--gpu gpu]
     ```

   - LSTM string IRPP model
   
     The best learned model for LSTM string IRPP model is `SymbolicIntegrationModel/LSTM_string/model/LSTM_string_IRPP_best_model`.
     To verify the accuracy of the learned model using test data in `SymbolicIntegrationModel/LSTM_string/dataset/LSTM_string_IRPP_test_Integrand.txt`, run the following:

     ```sh
     % cd LSTM_string/src
     % python LSTM_string_IRPP_model.py --token_dataset ../dataset/LSTM_string_polish_token.txt --Integrand_dataset ../dataset/LSTM_string_IRPP_test_Integrand.txt --Primitive_dataset ../dataset/LSTM_string_IRPP_test_Primitive.txt --study_name MLP_cupy_MedianPruner_epoch30_integrand_reverse_polish_Primitive_polish_third_try_memory_edited_v102_continue_untilepoch200 --learned_model ../model/LSTM_string_IRPP_best_model [--gpu gpu]
     ```

   - LSTM subtree polish model
   
     The best learned model for LSTM subtree polish model is `SymbolicIntegrationModel/LSTM_subtree/model/LSTM_subtree_polish_best_model`.
     To verify the accuracy of the learned model using test data in `SymbolicIntegrationModel/LSTM_subtree/dataset/LSTM_subtree_polish_test_Integrand.txt`, run the following:

     ```sh
     % cd LSTM_subtree/src
     % python LSTM_subtree_model.py --token_dataset ../dataset/LSTM_subtree_polish_token.txt --Integrand_dataset ../dataset/LSTM_subtree_polish_test_Integrand.txt --Primitive_dataset ../dataset/LSTM_subtree_polish_test_Primitive.txt --study_name MLP_cupy_MedianPruner_epoch30_subtree_complete_correct_continue --learned_model ../model/LSTM_subtree_polish_best_model [--gpu gpu]
     ```

   - LSTM subtree IRPP model
   
     The best learned model for LSTM subtree IRPP model is `SymbolicIntegrationModel/LSTM_subtree/model/LSTM_subtree_IRPP_best_model`. 
     To verify the accuracy of the learned model using test data in `SymbolicIntegrationModel/LSTM_subtree/dataset/LSTM_subtree_IRPP_test_Integrand.txt`, run the following:

     ```sh
     % cd LSTM_subtree/src
<<<<<<< Updated upstream
     % python LSTM_subtree_model.py --token_dataset ../dataset/LSTM_subtree_IRPP_token.txt --Integrand_dataset ../dataset/LSTM_subtree_IRPP_test_Integrand.txt --Primitive_dataset ../dataset/LSTM_subtree_IRPP_test_Primitive.txt --study_name MLP_cupy_MedianPruner_epoch30_subtree_Integrand_reverse_polish_Primitive_polish_continue --learned_model ../model/LSTM_subtree_IRPP_best_model [--gpu gpu]
=======
     % python LSTM_subtree_model.py --gpu 0 --token_dataset ../dataset/LSTM_subtree_IRPP_token.txt --Integrand_dataset ../dataset/LSTM_subtree_IRPP_test_Integrand.txt --Primitive_dataset ../dataset/LSTM_subtree_IRPP_test_Primitive.txt --study_name MLP_cupy_MedianPruner_epoch30_subtree_Integrand_reverse_polish_Primitive_polish_continue --learned_model ../model/LSTM_subtree_IRPP_best_model
>>>>>>> Stashed changes
     ```

   The following list of options will be displayed by adding -h option to each script for LSTM models.

   ```
   --Integrand_dataset                          : Specify Integrand data (text file).
   --Primitive_dataset                          : Specify Primitive data (text file).
   --token_dataset                              : Specify dictionary of mathematical symbols used in Integrand and Primitive data (text file).
   --study_name                                 : Specify hyperparameter values from Optuna result (SQLite database).
   --learned_model                              : Specify learned model (npz file).
   --gpu GPU, -g GPU                            : Specify GPU ID (negative value indicates CPU).
   ```


   To run Transformer models, follow the commands below after change directory. 

   - Transformer string polish model
   
     The best learned model for Transformer string polish model is `SymbolicIntegrationModel/Transformer_string/model/Transformer_string_polish_best_model`.
     To verify the accuracy of the learned model using test data in `SymbolicIntegrationModel/Transformer_string/dataset/Transformer_string_polish_test_Integrand.txt`, run the following:

     ```sh
     % cd Transformer_string/src
     % python Transformer_string_model.py --source ../dataset/Transformer_string_polish_test_Integrand.txt --target ../dataset/Transformer_string_polish_test_Primitive.txt --source_vocab_list ../dataset/Transformer_string_polish_Integrand_vocab.pickle --target_vocab_list ../dataset/Transformer_string_polish_Primitive_vocab.pickle --learned_model ../model/Transformer_string_polish_best_model [--gpu gpu]
     ```

   - Transformer string IRPP model
   
     The best learned model for Transformer string IRPP model is `SymbolicIntegrationModel/Transformer_string/model/Transformer_string_IRPP_best_model`.
     To verify the accuracy of the learned model using test data in `SymbolicIntegrationModel/Transformer_string/dataset/Transformer_string_IRPP_test_Integrand.txt`, run the following:
 
     ```sh
     % cd Transformer_string/src
     % python Transformer_string_model.py --source ../dataset/Transformer_string_IRPP_test_Integrand.txt --target ../dataset/Transformer_string_IRPP_test_Primitive.txt --source_vocab_list ../dataset/Transformer_string_IRPP_source_vocab.pickle --target_vocab_list ../dataset/Transformer_string_IRPP_target_vocab.pickle --learned_model ../model/Transformer_string_IRPP_best_model [--gpu gpu]
     ```

   - Transformer subtree polish model
   
     The best learned model for Transformer subtree polish model is `SymbolicIntegrationModel/Transformer_subtree/model/Transformer_subtree_polish_best_model`.
     To verify the accuracy of the learned model using test data in `SymbolicIntegrationModel/Transformer_subtree/dataset/Transformer_subtree_polish_test_Integrand.txt`,run the following:

     ```sh
     % cd Transformer_subtree/src
     % python Transformer_subtree_model.py --source ../dataset/Transformer_subtree_polish_test_Integrand.txt --target ../dataset/Transformer_subtree_polish_test_Primitive.txt --source_vocab_list ../dataset/Transformer_subtree_polish_Integrand_vocab.pickle --target_vocab_list ../dataset/Transformer_subtree_polish_Primitive_vocab.pickle --learned_model ../model/Transformer_subtree_polish_best_model [--gpu gpu]
     ```

   - Transformer subtree IRPP model
   
     The best learned model for Transformer subtree IRPP model is `SymbolicIntegrationModel/Transformer_subtree/model/Transformer_subtree_IRPP_best_model`.
     To verify the accuracy of the learned model using test data in  `SymbolicIntegrationModel/Transformer_subtree/dataset/Transformer_subtree_IRPP_test_Integrand.txt`,run the following:

     ```sh
     % cd Transformer_subtree/src
     % python Transformer_subtree_model.py --source ../dataset/Transformer_subtree_IRPP_test_Integrand.txt  --target ../dataset/Transformer_subtree_IRPP_test_Primitive.txt --source_vocab_list ../dataset/Transformer_subtree_IRPP_Integrand_vocab.pickle --target_vocab_list ../dataset/Transformer_subtree_IRPP_Primitive_vocab.pickle --learned_model ../model/Transformer_subtree_IRPP_best_model [--gpu gpu]
     ```

   The following list of options will be displayed by adding -h option to each script for Transformer models.

   ```
   --source                                     : Specify Integrand data (text file).
   --target                                     : Specify Primitive data (text file).
   --source_vocab_list                          : Specify dictionary of mathematical symbols used in Integrand data (pickle file).
   --target_vocab_list                          : Specify dictionary of mathematical symbols used in Primitive data (pickle file).
   --learned_model                              : Specify learned model (npz file).
   --gpu GPU, -g GPU                            : Specify GPU ID (negative value indicates CPU).
   ```

   - Integrated All Models
   
     To integrate the above eight models and perform inference, run the following:

     ```sh
     % cd Integrated_all_model
     % ./run.sh ../LSTM_string/dataset/LSTM_string_Polish_test_Integrand.txt 
     ```

## Acknowledgement

The development of this algorithm was funded by Japan Science and Technology Agency CREST grant (Grant Number: JPMJCR2011) to [Akira Funahashi](https://github.com/funasoul).


## References

[https://github.com/soskek/attention_is_all_you_need](https://github.com/soskek/attention_is_all_you_need)
