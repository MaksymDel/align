 local dir_xnli = "data/xnli/";
 local prefix_xnli_dev = "xnli.dev";
 local prefix_xnli_test = "xnli.test";

 # local bert_model = "bert-base-multilingual-cased";
 local bert_model = "bert-base-cased";

 local mnli_reader = {
         "type": "mnli",
         "lazy": false,
         "is_bert_pair": false,
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
                 "do_lowercase": false,
             }
         }
     };

 {
     "dataset_reader": {
         "type": "interleaving",
         "lazy": false,
         "scheme": "round_robin",
         "readers": {
             'nli-en': mnli_reader, 
         }
     },

     "validation_dataset_reader": {
         "type": "interleaving",
         "lazy": false,
         "scheme": "all_at_once",
         "readers": {
             'nli-ar': mnli_reader, 
             'nli-bg': mnli_reader, 
             'nli-de': mnli_reader,
             'nli-el': mnli_reader, 
             'nli-en': mnli_reader, 
             'nli-es': mnli_reader, 
             'nli-fr': mnli_reader, 
             'nli-hi': mnli_reader, 
             'nli-ru': mnli_reader, 
             'nli-sw': mnli_reader, 
             'nli-th': mnli_reader, 
             'nli-tr': mnli_reader, 
             'nli-ur': mnli_reader, 
             'nli-vi': mnli_reader, 
             'nli-zh': mnli_reader
         }
     },

     # "train_data_path": "data/multinli/multinli_1.0_train.jsonl",
     "train_data_path": std.toString({
              'nli-en': dir_xnli + prefix_xnli_dev + ".en"}),
     
     "validation_data_path": std.toString({
             'nli-ar': dir_xnli + prefix_xnli_dev + ".ar", 
             'nli-bg': dir_xnli + prefix_xnli_dev + ".bg", 
             'nli-de': dir_xnli + prefix_xnli_dev + ".de",
             'nli-el': dir_xnli + prefix_xnli_dev + ".el", 
             'nli-en': dir_xnli + prefix_xnli_dev + ".en", 
             'nli-es': dir_xnli + prefix_xnli_dev + ".es", 
             'nli-fr': dir_xnli + prefix_xnli_dev + ".fr", 
             'nli-hi': dir_xnli + prefix_xnli_dev + ".hi", 
             'nli-ru': dir_xnli + prefix_xnli_dev + ".ru", 
             'nli-sw': dir_xnli + prefix_xnli_dev + ".sw", 
             'nli-th': dir_xnli + prefix_xnli_dev + ".th", 
             'nli-tr': dir_xnli + prefix_xnli_dev + ".tr", 
             'nli-ur': dir_xnli + prefix_xnli_dev + ".ur", 
             'nli-vi': dir_xnli + prefix_xnli_dev + ".vi", 
             'nli-zh': dir_xnli + prefix_xnli_dev + ".zh"}),

     "model": {
         "type": "xnli_bert",
         "bert_model": bert_model,
         "dropout": 0.0
     },

     "iterator": {
         "type": "homogeneous_batch",
         "batch_size": 16,
     },
     
     "trainer": {
         "optimizer": {
             "type": "bert_adam",
             "lr": 6e-5
         },
         "validation_metric": "+nli-en",
         "num_serialized_models_to_keep": 5,
         "num_epochs": 100,
         # "grad_norm": 10.0,
         "patience": 10,
         "cuda_device": 0
     }
 }