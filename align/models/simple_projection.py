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

@Model.register("simple_projection")
class SimpleProjection(Model):
    """

    """
    def __init__(self, 
                 vocab: Vocabulary,
                 input_embedder: TextFieldEmbedder,
                 pooler: Seq2VecEncoder,
                 nli_projection_layer: FeedForward,
                 training_tasks: Any,
                 validation_tasks: Any,
                 dropout: float = 0.0,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(SimpleProjection, self).__init__(vocab, regularizer)
        if type(training_tasks) == dict:
            self._training_tasks = list(training_tasks.keys())
        else:
            self._training_tasks = training_tasks

        if type(validation_tasks) == dict:
            self._validation_tasks = list(validation_tasks.keys())
        else:
            self._validation_tasks = validation_tasks

        self._input_embedder = input_embedder 
        self._pooler = pooler

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
        
    def forward(self,  # type: ignore
                premise_hypothesis: Dict[str, torch.Tensor] = None,
                premise: Dict[str, torch.Tensor] = None,
                hypothesis: Dict[str, torch.LongTensor] = None,
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
        if dataset is not None: # TODO: hardcoded; used when not multitask reader was used
            taskname = dataset[0]
        else:
            taskname = "nli-en"

        if premise_hypothesis is not None:
            assert premise is None and hypothesis is None
        
        if premise_hypothesis is not None:
            embedded_combined = self._input_embedder(premise_hypothesis)
            mask = get_text_field_mask(premise_hypothesis).float()
            pooled_combined = self._pooler(embedded_combined, mask=mask)
        elif premise is not None and hypothesis is not None:
            embedded_premise = self._input_embedder(premise)
            embedded_hypothesis = self._input_embedder(hypothesis)

            mask_premise = get_text_field_mask(premise).float()
            mask_hypothesis = get_text_field_mask(hypothesis).float()

            pooled_premise = self._pooler(embedded_premise, mask=mask_premise)
            pooled_hypothesis = self._pooler(embedded_hypothesis, mask=mask_hypothesis)

            pooled_combined = torch.cat([pooled_premise, pooled_hypothesis], dim=-1)
        else:
            raise ConfigurationError("One of premise or hypothesis is None. Check your DatasetReader")

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
            if metricname[-2:] != 'en': # hide other langs from tqdn
                metricname = '_' + metricname
            metrics[metricname] = self._nli_per_lang_acc[taskname].get_metric(reset)
        
        accs = metrics.values() # TODO: should only count 'nli-*' metrics
        avg = sum(accs) / sum(x > 0 for x in accs)
        self._nli_avg_acc(avg)
        metrics["nli-avg"] = self._nli_avg_acc.get_metric(reset)

        return metrics
