# local bert_model = "bert-base-multilingual-cased";
local bert_model = "xlm-mlm-xnli15-1024";
#local bert_model = "xlm-mlm-tlm-xnli15-1024";
local bert_data_format = true;
local bert_trainable = true;
local bert_lower = true;
local ALIGN_TASKS = ['ar', 'bg', 'de', 'el', 'en', 'es', 'fr', 'hi', 'ru', 'sw', 'th', 'tr', 'ur', 'vi', 'zh'];
# local ALIGN_TASKS_NO_EN = ['ar', 'bg', 'de', 'el', 'es', 'fr', 'hi', 'ru', 'sw', 'th', 'tr', 'ur', 'vi', 'zh'];
# local ALIGN_TASKS_2PIRNT = ['ar', 'de', 'es', 'hi', 'ru', 'sw', 'th', 'ur', 'vi', 'zh'];
local VALID_LAGNS_2PRINT = ['fr', 'en', 'de', 'bg', 'ur', 'sw'];
local XNLI_TASKS = ['nli-ar', 'nli-bg', 'nli-de', 'nli-el', 'nli-en', 'nli-es', 'nli-fr', 'nli-hi', 'nli-ru', 'nli-sw', 'nli-th', 'nli-tr', 'nli-ur', 'nli-vi', 'nli-zh'];

# "ar-en bg-en de-en el-en en-es en-fr en-hi en-ru en-sw en-th en-tr en-ur en-vi en-zh"
local ALIGN_LANG_PAIRS = "en-sw";
local ALIGN_TASKS_2PIRNT = ['en', 'fr', 'de', 'sw', 'ur'];

local learners_ser_dir = "/home/maksym/research/align/experiments/baseline_mlm/";
local teacher_archive = learners_ser_dir + "model.tar.gz";
local student_archive = learners_ser_dir + "model.tar.gz";
local labels_vocab_file = learners_ser_dir + "vocabulary/labels.txt";

{
    "validation_dataset_reader": {
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
    "dataset_reader": {
        "type": "aligner_reader_xnli",
        "xlm_model_name": bert_model,
        "do_lowercase": bert_lower,
        "token_indexers": {
            "bert": {
                "type": "pretrained_transformer",
                "model_name": bert_model,
                "do_lowercase": bert_lower,
            }
        },
        "max_sent_len": 128,
        #"lg_pairs": "ar-en bg-en de-en el-en en-es en-fr en-hi en-ru en-sw en-th en-tr en-ur en-vi en-zh", 
        "lg_pairs": ALIGN_LANG_PAIRS,
        "target_lang": "en",
        "scheme": "round_robin",
        "lazy": false
    },
    #"train_data_path": "/home/maksym/research/XLM/data/para/prep/valid",
    #"train_data_path": "/home/maksym/research/XLM/data/para/prep/train",
    "train_data_path": "data/translate_train",
    
    #"validation_data_path": "/home/maksym/research/XLM/data/para/prep/valid",
    "validation_data_path": "data/xnli/xnli.dev.jsonl",
    #"test_data_path": "/home/maksym/research/XLM/data/para/prep/valid",
    #"evaluate_on_test": true,
    #"datasets_for_vocab_creation": ["train"],
    #"test_data_path": "data/xnli/xnli.test.jsonl",
    #"evaluate_on_test": true,

    "model": {
        "type": "aligner",
        "loss": "mse", # l1 | mse | cos | smooth_l1
        "reduction": "mean", # mean | sum
        "labels_vocab_file" : labels_vocab_file,
        "teacher_xlm": {
            "_pretrained": {
                "archive_file": teacher_archive,
                "module_path": "_input_embedder",
                "freeze": true
            },
        },
        "student_xlm": {
            "_pretrained": {
                "archive_file": student_archive,
                "module_path": "_input_embedder",
                "freeze": false
            },
        },
        "teacher_nli_head": { # needed for evaluation
            "_pretrained": {
                "archive_file": teacher_archive,
                "module_path": "_nli_projection_layer",
                "freeze": true
            },
        },

        // "projector_feedforward": {
        //     "input_dim": 1024,
        //     "num_layers": 1,
        //     "hidden_dims": [1024],
        //     "activations": ["linear"],
        //     "dropout": [0.0]
        // }, 
        "training_tasks": ALIGN_TASKS,
        "training_tasks_2print": ALIGN_TASKS_2PIRNT,
        "valid_langs_2print": VALID_LAGNS_2PRINT,
        "validation_tasks": XNLI_TASKS,

        "dropout": 0.0,
    },

    "iterator": {
        "type": "homogeneous_batch",
        "batch_size": 8,
        #"max_instances_in_memory": 20000,
        "instances_per_epoch":  20000 # 200
        #"instances_per_epoch":  0 # 200
    },


    "validation_iterator": {
        "type": "homogeneous_batch",
        "batch_size": 32,
    },
    
    "trainer": {
        "optimizer": {
            "type": "adam",
            # "lr": 1e-4
            # "lr": 5e-4,
            # "lr": 0.00000125,
            # "lr": 9e-5 # doesn't work
            "lr": 5e-6
            # "lr": 0.00001,

            # "betas": [0.9, 0.999]
            # "lr": if bert_data_format then 9e-6 else 9e-7 ------> if batch 32
            # "lr": 0.000005,
        },
        
    # "should_log_learning_rate": true,

        "validation_metric": "+nli-sw",
        "num_serialized_models_to_keep": 2,
        "num_epochs": 400,
        # "grad_norm": 10.0,
        "patience": 60,
        "cuda_device": [1]
    }
}