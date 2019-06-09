/** You could basically use this config to train your own BERT classifier,
in the future, training_dataset_reader might become multitasl: https://github.com/allenai/allennlp/pull/2369
 */


 local bert_model = "bert-base-multilingual-cased";
 
 {
     "dataset_reader": {
         "lazy": false,
         "type": "mnli",
         "max_sent_len": 101,
         "tokenizer": {
             "word_splitter": {
                "type": "bert-basic",
                "do_lower_case": false
             }
         },
         "token_indexers": {
             "bert": {
                 "type": "bert-pretrained",
                 "pretrained_model": bert_model,
                 "do_lowercase": false
             }
         }
     },

     "validation_dataset_reader": {
         "lazy": false,
         "type": "xnli",
         "max_sent_len": 101,
         "tokenizer": {
             "word_splitter": {
                "type": "bert-basic",
                "do_lower_case": false
             }         
        },
         "token_indexers": {
             "bert": {
                 "type": "bert-pretrained",
                 "pretrained_model": bert_model,
                 "do_lowercase": false
             }
         }
     },

     "train_data_path": "data/multinli-orig/multinli_1.0_train.jsonl",
     "validation_data_path": "data/xnli-orig/xnli.dev.jsonl",

     "model": {
         "type": "xnli_bert",
         "bert_model": bert_model,
         "dropout": 0.0
     },

     "iterator": {
         "type": "bucket",
         "sorting_keys": [["premise", "num_tokens"], ["hypothesis", "num_tokens"]],
         "batch_size": 16,
         "biggest_batch_first": true
     },
     
     "trainer": {
         "optimizer": {
             "type": "bert_adam",
             "lr": 6e-5
         },
         "validation_metric": "+accuracy",
         "num_serialized_models_to_keep": 5,
         "num_epochs": 100,
         # "grad_norm": 10.0,
         "patience": 10,
         "cuda_device": 0
     }
 }
 