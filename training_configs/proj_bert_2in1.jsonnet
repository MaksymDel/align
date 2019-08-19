local bert_model = "bert-base-multilingual-cased";
local bert_data_format = true;
local bert_trainable = true;
local bert_lower = false;
local XNLI_TASKS = ['nli-ar', 'nli-bg', 'nli-de', 'nli-el', 'nli-en', 'nli-es', 'nli-fr', 'nli-hi', 'nli-ru', 'nli-sw', 'nli-th', 'nli-tr', 'nli-ur', 'nli-vi', 'nli-zh'];

{
    "dataset_reader": {
        "type": "xnli",
        "lazy": false,
        "bert_format": bert_data_format,
        "max_sent_len": 80,
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
    },
    // "train_data_path": "data/xnli/xnli.dev.en",
    "train_data_path": "data/multinli/multinli_1.0_train.jsonl",
    "validation_data_path": "data/xnli/xnli.dev.jsonl",
    "test_data_path": "data/xnli/xnli.test.jsonl",
    "evaluate_on_test": true,
    "datasets_for_vocab_creation": ["train"],

    "model": {
        "type": "simple_projection",
        "training_tasks": ['nli-en'],
        "validation_tasks": XNLI_TASKS,
    
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
        "batch_size": 32,
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
            "lr": 5e-6
        },
        "validation_metric": "+nli-en",
        "num_serialized_models_to_keep": 10,
        "num_epochs": 10,
        # "grad_norm": 10.0,
        "patience": 2,
        "cuda_device": [0, 1]
    }
}