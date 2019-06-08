/** You could basically use this config to train your own BERT classifier,
in the future, training_dataset_reader might become multitasl: https://github.com/allenai/allennlp/pull/2369
 */


 local bert_model = "bert-base-multilingual-cased";
 
 {
     "dataset_reader": {
         "lazy": false,
         "type": "snli",
         "tokenizer": {
             "word_splitter": "bert-basic"
         },
         "token_indexers": {
             "bert": {
                 "type": "bert-pretrained",
                 "pretrained_model": bert_model
             }
         }
     },

     "validation_dataset_reader": {
         "lazy": false,
         "type": "xnli",
         "tokenizer": {
             "word_splitter": "bert-basic"
         },
         "token_indexers": {
             "bert": {
                 "type": "bert-pretrained",
                 "pretrained_model": bert_model
             }
         }
     },

     "train_data_path": "fixtures/data/mnli.jsonl",
     "validation_data_path": "fixtures/data/xnli.jsonl",

     "model": {
         "type": "xnli_bert",
         "bert_model": bert_model,
         "dropout": 0.2
     },

     "iterator": {
         "type": "bucket",
         "sorting_keys": [["premise", "num_tokens"], ["hypothesis", "num_tokens"]],
         "batch_size": 5
     },
     
     "trainer": {
         "optimizer": {
             "type": "adam",
             "lr": 0.001
         },
         "validation_metric": "+accuracy",
         "num_serialized_models_to_keep": 1,
         "num_epochs": 30,
         "grad_norm": 10.0,
         "patience": 5,
         "cuda_device": -1
     }
 }
 