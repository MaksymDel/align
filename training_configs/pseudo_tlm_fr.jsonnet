# local bert_model = "bert-base-multilingual-cased";
#local bert_model = "xlm-mlm-xnli15-1024";
local bert_model = "xlm-mlm-tlm-xnli15-1024";
local bert_data_format = true;
local bert_trainable = true;
local bert_lower = true;
local XNLI_TASKS = ['nli-ar', 'nli-bg', 'nli-de', 'nli-el', 'nli-en', 'nli-es', 'nli-fr', 'nli-hi', 'nli-ru', 'nli-sw', 'nli-th', 'nli-tr', 'nli-ur', 'nli-vi', 'nli-zh'];

{
    "dataset_reader": {
        "type": "xnli_xlm",
        "lazy": false,
        "max_sent_len": 256,
        "xlm_model_name": bert_model,
        "do_lowercase": bert_lower,
        "token_indexers": {
            "bert": {
                "type": "pretrained_transformer",
                "model_name": bert_model,
                "do_lowercase": bert_lower,
            }
        }
    },
    // "train_data_path": "data/xnli/xnli.dev.en",
    #"train_data_path": "data/multinli/multinli_1.0_train.jsonl",

    "train_data_path": "data/translate_train/multinli.train.fr.jsonl",    
    "validation_data_path": "data/xnli/xnli.dev.jsonl",
    "test_data_path": "data/xnli/xnli.test.jsonl",
    "evaluate_on_test": true,
    "datasets_for_vocab_creation": ["train"],

    "model": {
        "type": "simple_projection_xlm",
        "training_tasks": ['nli-fr'],
        "validation_tasks": XNLI_TASKS,
    
        "input_embedder": {
            "token_embedders": {
                "bert": {
                    "type": "xlm15",
                    "model_name": bert_model,
                    # "requires_grad": bert_trainable
                }
            }
        },
        
        "dropout": 0.0,
        "nli_projection_layer": {
            "input_dim": if bert_data_format then 1024 else 768 * 2,
            "num_layers": 1,
            "hidden_dims": 3,
            "activations": "linear",
            "dropout": 0.0
        },
    },

    "iterator": {
        "type": "bucket",
        "sorting_keys": if bert_data_format then [["premise_hypothesis", "num_tokens"]] else [["premise", "num_tokens"], ["hypothesis", "num_tokens"]],
        "batch_size": 8,
        "biggest_batch_first": true,
        "instances_per_epoch": 20000
    },

    "validation_iterator": {
        "type": "homogeneous_batch",
        "batch_size": 32,
    },
    
    "trainer": {
        "optimizer": {
            "type": "adam",
            "lr": 0.000005,
            # "lr": 0.00000125,
            # "lr": 0.000002
            # "lr": 0.00001,

            # "betas": [0.9, 0.999]
            # "lr": if bert_data_format then 9e-6 else 9e-7 ------> if batch 32
            # "lr": 0.000005,
        },
        
    # "should_log_learning_rate": true,

        "validation_metric": "+nli-fr",
        "num_serialized_models_to_keep": 2,
        "num_epochs": 400,
        # "grad_norm": 10.0,
        "patience": 60,
        "cuda_device": [0]
    }
}