// Configuraiton for a textual entailment model based on:
//  Parikh, Ankur P. et al. “A Decomposable Attention Model for Natural Language Inference.” EMNLP (2016).

 local dir_xnli = "data/xnli/";
 local prefix_xnli_dev = "xnli.dev";
 local prefix_xnli_test = "xnli.test";

 local bert_model = "bert-base-cased";

{
  "dataset_reader": {
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
        "tokens": {
            "type": "bert-pretrained",
            "pretrained_model": bert_model,
            "do_lowercase": false,
        }
    },
  },
  "train_data_path": dir_xnli + prefix_xnli_dev + ".en",
  "validation_data_path": dir_xnli + prefix_xnli_dev + ".en",
  "model": {
    "type": "decomposable_attention",
    "text_field_embedder": {
      "allow_unmatched_keys": true,
        "embedder_to_indexer_map": {
            "tokens": ["tokens", "tokens-offsets"],
      },

      "token_embedders": {
        "tokens": {
            "type": "bert-pretrained",
            "pretrained_model": bert_model,
            "top_layer_only": true,
            "requires_grad": true
        }
      }
    },
    "attend_feedforward": {
      "input_dim": 768,
      "num_layers": 2,
      "hidden_dims": 200,
      "activations": "relu",
      "dropout": 0.2
    },
    "similarity_function": {"type": "dot_product"},
    "compare_feedforward": {
      "input_dim": 768 * 2,
      "num_layers": 2,
      "hidden_dims": 200,
      "activations": "relu",
      "dropout": 0.2
    },
    "aggregate_feedforward": {
      "input_dim": 400,
      "num_layers": 2,
      "hidden_dims": [200, 3],
      "activations": ["relu", "linear"],
      "dropout": [0.2, 0.0]
    },
     "initializer": [
      [".*linear_layers.*weight", {"type": "xavier_normal"}],
      [".*token_embedder_tokens\\._projection.*weight", {"type": "xavier_normal"}]
     ]
   },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [["premise", "num_tokens"], ["hypothesis", "num_tokens"]],
    "batch_size": 32
  },

  "trainer": {
    "num_epochs": 140,
    "patience": 20,
    "cuda_device": 0,
    "grad_clipping": 5.0,
    "validation_metric": "+accuracy",
    "optimizer": {
        "type": "bert_adam",
        "lr": 2e-5
    },
  }
}
