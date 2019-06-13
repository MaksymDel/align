local dir_xnli = "data/xnli/";
local prefix_xnli_dev = "xnli.dev";
local prefix_xnli_test = "xnli.test";

local dev_paths = {
             'nli-ar': dir_xnli + prefix_xnli_dev + ".ar", 
            # 'nli-bg': dir_xnli + prefix_xnli_dev + ".bg", 
             'nli-de': dir_xnli + prefix_xnli_dev + ".de",
            # 'nli-el': dir_xnli + prefix_xnli_dev + ".el", 
             'nli-en': dir_xnli + prefix_xnli_dev + ".en", 
            # 'nli-es': dir_xnli + prefix_xnli_dev + ".es", 
             'nli-fr': dir_xnli + prefix_xnli_dev + ".fr", 
            # 'nli-hi': dir_xnli + prefix_xnli_dev + ".hi", 
             'nli-ru': dir_xnli + prefix_xnli_dev + ".ru", 
            # 'nli-sw': dir_xnli + prefix_xnli_dev + ".sw", 
            # 'nli-th': dir_xnli + prefix_xnli_dev + ".th", 
            # 'nli-tr': dir_xnli + prefix_xnli_dev + ".tr", 
            # 'nli-ur': dir_xnli + prefix_xnli_dev + ".ur", 
            # 'nli-vi': dir_xnli + prefix_xnli_dev + ".vi", 
             'nli-zh': dir_xnli + prefix_xnli_dev + ".zh"};

local train_paths = {'nli-en': "data/multinli/multinli_1.0_train.jsonl"};

// local dev_paths = {'nli-en': dir_xnli + prefix_xnli_dev + ".en"};
// local train_paths = {'nli-en': dir_xnli + prefix_xnli_dev + ".en"};

local bert_model = "bert-base-multilingual-cased";
local bert_data_format = true;
local bert_trainable = true;
local bert_lower = false;

# local bert_model = "bert-base-cased";

local mnli_reader = {
        "type": "mnli",
        "lazy": false,
        "bert_format": bert_data_format,
        "max_sent_len": 100,
        "tokenizer": {
            "word_splitter": {
                "type": "bert-basic",
                "do_lower_case": bert_lower
            }
        },
        "token_indexers": {
            "bert": {
                "type": "bert-pretrained",
                "pretrained_model": bert_model,
                "do_lowercase": bert_lower,
            }
        }
    };

{
    // "dataset_reader": {
    //     "type": "interleaving",
    //     "lazy": false,
    //     "scheme": "round_robin",
    //     "readers": {
    //         'nli-en': mnli_reader, 
    //     }
    // },

    "dataset_reader": mnli_reader,

    "validation_dataset_reader": {
        "type": "interleaving",
        "lazy": false,
        "scheme": "all_at_once",
         "readers": {
             'nli-ar': mnli_reader, 
            # 'nli-bg': mnli_reader, 
             'nli-de': mnli_reader,
            # 'nli-el': mnli_reader, 
             'nli-en': mnli_reader, 
            # 'nli-es': mnli_reader, 
             'nli-fr': mnli_reader, 
            # 'nli-hi': mnli_reader, 
             'nli-ru': mnli_reader, 
            # 'nli-sw': mnli_reader, 
            # 'nli-th': mnli_reader, 
            # 'nli-tr': mnli_reader, 
            # 'nli-ur': mnli_reader, 
            # 'nli-vi': mnli_reader, 
             'nli-zh': mnli_reader
         }
        // "readers": {
        //     'nli-en': mnli_reader, 
        // }
    },

    "train_data_path": train_paths['nli-en'],
    "validation_data_path": std.toString(dev_paths),

    "model": {
        "type": "simple_projection",
        "training_tasks": train_paths,
        "validation_tasks": dev_paths,
    
        "input_embedder": {
            "allow_unmatched_keys": true,
            "embedder_to_indexer_map": {
                "bert": ["bert", "bert-offsets"],
            },
            "token_embedders": {
                "bert": {
                    "type": "bert-pretrained",
                    "pretrained_model": bert_model,
                    "top_layer_only": true,
                    "requires_grad": bert_trainable
                }
            }
        },
        "pooler": {
            "type": "bert_pooler",
            "pretrained_model": bert_model,
            "requires_grad": bert_trainable

        },
        "dropout": 0.0,
        "nli_projection_layer": {
            "input_dim": if bert_data_format then 768 else 768 * 2,
            "num_layers": 1,
            "hidden_dims": 3,
            "activations": "linear",
            "dropout": 0.0
        },
    },

    "iterator": {
        "type": "bucket",
        "sorting_keys": if bert_data_format then [["premise_hypothesis", "num_tokens"]] else [["premise", "num_tokens"], ["hypothesis", "num_tokens"]],
        "batch_size": 16,
        "biggest_batch_first": true,
    },

    "validation_iterator": {
        "type": "homogeneous_batch",
        "batch_size": 32,
    },
    
    "trainer": {
        "optimizer": {
            "type": "bert_adam",
            # "lr": if bert_data_format then 9e-6 else 9e-7 ------> if batch 32
            "lr": 1e-6
        },
        "validation_metric": "+nli-en",
        "num_serialized_models_to_keep": 5,
        "num_epochs": 100,
        # "grad_norm": 10.0,
        "patience": 10,
        "cuda_device": 0
    }
}