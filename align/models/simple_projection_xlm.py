from typing import Any, Dict, List, Optional, Union

import torch
from allennlp.common.checks import ConfigurationError, check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import (FeedForward, Seq2SeqEncoder, Seq2VecEncoder,
                              SimilarityFunction, TextFieldEmbedder,
                              TimeDistributed)
from allennlp.modules.matrix_attention.legacy_matrix_attention import \
    LegacyMatrixAttention
from allennlp.modules.token_embedders.bert_token_embedder import \
    PretrainedBertModel
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.initializers import InitializerApplicator
from allennlp.nn.util import get_text_field_mask, masked_softmax, weighted_sum
from allennlp.training.metrics import Average, CategoricalAccuracy
from overrides import overrides
from pytorch_pretrained_bert.modeling import BertModel

@Model.register("simple_projection_xlm")
class SimpleProjectionXlm(Model):
    """

    """
    def __init__(self, 
                 vocab: Vocabulary,
                 input_embedder: TextFieldEmbedder,
                 nli_projection_layer: FeedForward,
                 training_tasks: Any,
                 validation_tasks: Any,
                 langs_print_train: List[str] = ["en"],
                 dropout: float = 0.0,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 feed_lang_ids: bool = True,
                 avg: bool = False) -> None:
        super(SimpleProjectionXlm, self).__init__(vocab, regularizer)
        
        self._avg = avg

        if type(training_tasks) == dict:
            self._training_tasks = list(training_tasks.keys())
        else:
            self._training_tasks = training_tasks

        if type(validation_tasks) == dict:
            self._validation_tasks = list(validation_tasks.keys())
        else:
            self._validation_tasks = validation_tasks

        self._input_embedder = input_embedder 

        self._label_namespace = "labels"
        self._num_labels = vocab.get_vocab_size(namespace=self._label_namespace)

        self._nli_projection_layer = nli_projection_layer
        print(vocab.get_token_to_index_vocabulary(namespace=self._label_namespace))
        assert nli_projection_layer.get_output_dim() == self._num_labels


        self._dropout = torch.nn.Dropout(p=dropout)

        self._loss = torch.nn.CrossEntropyLoss()

        initializer(self._nli_projection_layer)

        self._nli_per_lang_acc: Dict[str, CategoricalAccuracy] = dict()
        
        for taskname in self._validation_tasks:
            # this will hide some metrics from tqdm, but they will still be computed
            self._nli_per_lang_acc[taskname] = CategoricalAccuracy()
        self._nli_avg_acc = Average()
        
        self._langs_pring_train = langs_print_train or "en"
        if '*' in self._langs_pring_train:
            self._langs_pring_train = [t.split("")[-1] for t in training_tasks] 
        
        self._feed_lang_ids = feed_lang_ids   

    def forward(self,  # type: ignore
                premise_hypothesis: Dict[str, torch.Tensor],
                dataset: List[str] = None,
                label: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        premise_hypothesis : Dict[str, torch.LongTensor]
            Combined in a single text field for BERT encoding
        premise : Dict[str, torch.LongTensor]
            From a ``TextField``
        hypothesis : Dict[str, torch.LongTensor]
            From a ``TextField``
        label : torch.IntTensor, optional, (default = None)
            From a ``LabelField``
        dataset : List[str]
            Task indicator
        metadata : ``List[Dict[str, Any]]``, optional, (default = None)
            Metadata containing the original tokenization of the premise and
            hypothesis with 'premise_tokens' and 'hypothesis_tokens' keys respectively.
        Returns
        -------
        An output dictionary consisting of:

        label_logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing unnormalised log
            probabilities of the entailment label.
        label_probs : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing probabilities of the
            entailment label.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        taskname = "nli-en"
        if dataset is not None: # TODO: hardcoded; used when not multitask reader was used
            taskname = dataset[0]
        else:
            taskname = "nli-en"

        # xlm case, lang should be bs_seql tensor of lang ids
        mask = get_text_field_mask(premise_hypothesis).float()
        lang = taskname.split("-")[-1]
        lang_id = self._input_embedder._token_embedders["bert"].transformer_model.config.lang2id[lang]
        bs = mask.size()[0]
        seq_len = mask.size()[1]
        lang_ids = mask.new_full((bs, seq_len), lang_id).long()
        if self._feed_lang_ids:
            embedded_combined = self._input_embedder(premise_hypothesis, lang=lang_ids)
        else:
            embedded_combined = self._input_embedder(premise_hypothesis)            
        
        if not self._avg:
            pooled_combined = embedded_combined[:, 0, :]
            #pooled_combined = self._pooler(embedded_combined, mask=mask)
        else:
            pooled_combined = embedded_combined.mean(dim=1)

        pooled_combined = self._dropout(pooled_combined)

        logits = self._nli_projection_layer(pooled_combined)
        probs = torch.nn.functional.softmax(logits, dim=-1)

        output_dict = {"logits": logits, "probs": probs}

        if label is not None:
            loss = self._loss(logits, label.long().view(-1))
            output_dict["loss"] = loss
            
            self._nli_per_lang_acc[taskname](logits, label)

        if metadata is not None:
            output_dict["premise_tokens"] = [x["premise_tokens"] for x in metadata]
            output_dict["hypothesis_tokens"] = [x["hypothesis_tokens"] for x in metadata]

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the probabilities, converts index to string label, and
        add ``"label"`` key to the dictionary with the result.
        """
        predictions = output_dict["probs"]
        if predictions.dim() == 2:
            predictions_list = [predictions[i] for i in range(predictions.shape[0])]
        else:
            predictions_list = [predictions]
        classes = []
        for prediction in predictions_list:
            label_idx = prediction.argmax(dim=-1).item()
            label_str = (self.vocab.get_index_to_token_vocabulary(self._label_namespace)
                         .get(label_idx, str(label_idx)))
            classes.append(label_str)
        output_dict["label"] = classes
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {}

        if self.training:
            tasks = self._training_tasks
        else:
            tasks = self._validation_tasks

        for taskname in tasks:
            metricname = taskname
            if metricname.split("-")[-1] not in self._langs_pring_train: # hide other langs from tqdn
                metricname = '_' + metricname
            metrics[metricname] = self._nli_per_lang_acc[taskname].get_metric(reset)
        
        accs = metrics.values() # TODO: should only count 'nli-*' metrics
        #d = sum(x > 0 for x in accs)
        #if d == 0:
        #    d = 0.0001
        avg = sum(accs) / len(accs)
        self._nli_avg_acc(avg)
        metrics["nli-avg"] = self._nli_avg_acc.get_metric(reset)

        return metrics
