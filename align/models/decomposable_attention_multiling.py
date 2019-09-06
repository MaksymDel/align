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
from allennlp.models import DecomposableAttention
from overrides import overrides
from pytorch_pretrained_bert.modeling import BertModel


@Model.register("decomposable_attention_multiling")
class DecomposableAttentionMultiling(Model):
    """
    Requires premise and hypothesis to be embedded separatly
    """
    def __init__(self, 
                 vocab: Vocabulary,
                 training_tasks: Any,
                 validation_tasks: Any,
                 
                 text_field_embedder: TextFieldEmbedder,
                 attend_feedforward: FeedForward,
                 similarity_function: SimilarityFunction,
                 compare_feedforward: FeedForward,
                 aggregate_feedforward: FeedForward,
                 premise_encoder: Optional[Seq2SeqEncoder] = None,
                 hypothesis_encoder: Optional[Seq2SeqEncoder] = None,

                 langs_print_train: List[str] = None,
                 dropout: float = 0.0,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(DecomposableAttentionMultiling, self).__init__(vocab, regularizer=regularizer)
        if type(training_tasks) == dict:
            self._training_tasks = list(training_tasks.keys())
        else:
            self._training_tasks = training_tasks

        if type(validation_tasks) == dict:
            self._validation_tasks = list(validation_tasks.keys())
        else:
            self._validation_tasks = validation_tasks

        self._label_namespace = "labels"
        self._num_labels = vocab.get_vocab_size(namespace=self._label_namespace)

        # elmo / bert
        self._text_field_embedder = text_field_embedder
        # decomposable attention stuff
        self._attend_feedforward = TimeDistributed(attend_feedforward)
        self._matrix_attention = LegacyMatrixAttention(similarity_function)
        self._compare_feedforward = TimeDistributed(compare_feedforward)
        self._aggregate_feedforward = aggregate_feedforward
        self._premise_encoder = premise_encoder
        self._hypothesis_encoder = hypothesis_encoder or premise_encoder
        
        check_dimensions_match(text_field_embedder.get_output_dim(), attend_feedforward.get_input_dim(),
                               "text field embedding dim", "attend feedforward input dim")
        check_dimensions_match(aggregate_feedforward.get_output_dim(), self._num_labels,
                               "final output dimension", "number of labels")



        self._dropout = torch.nn.Dropout(p=dropout)

        self._loss = torch.nn.CrossEntropyLoss()

        # initializer(self._nli_projection_layer)

        self._nli_per_lang_acc: Dict[str, CategoricalAccuracy] = dict()
        
        for taskname in self._validation_tasks:
            # this will hide some metrics from tqdm, but they will still be computed
            self._nli_per_lang_acc[taskname] = CategoricalAccuracy()
        self._nli_avg_acc = Average()
        
        self._langs_pring_train = langs_print_train or "en"
        if '*' in self._langs_pring_train:
            self._langs_pring_train = [t.split("")[-1] for t in training_tasks] 

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
        taskname = "nli-en"
        if dataset is not None: # TODO: hardcoded; used when not multitask reader was used
            taskname = dataset[0]
        else:
            taskname = "nli-en"

        if premise_hypothesis is not None:
            raise ConfigurationError("you should not combine premise and hypothesis for this model")
        
        if premise is not None and hypothesis is not None:
            embedded_premise = self._text_field_embedder(premise, lang=taskname.split("-")[-1])
            embedded_hypothesis = self._text_field_embedder(hypothesis, lang=taskname.split("-")[-1])

            mask_premise = get_text_field_mask(premise).float()
            mask_hypothesis = get_text_field_mask(hypothesis).float()

            if self._premise_encoder:
                embedded_premise = self._premise_encoder(embedded_premise, mask_premise)
            if self._hypothesis_encoder:
                embedded_hypothesis = self._hypothesis_encoder(embedded_hypothesis, mask_hypothesis)

            projected_premise = self._attend_feedforward(embedded_premise)
            projected_hypothesis = self._attend_feedforward(embedded_hypothesis)
    
            # Shape: (batch_size, premise_length, hypothesis_length)
            similarity_matrix = self._matrix_attention(projected_premise, projected_hypothesis)

            # Shape: (batch_size, premise_length, hypothesis_length)
            p2h_attention = masked_softmax(similarity_matrix, mask_hypothesis)
            # Shape: (batch_size, premise_length, embedding_dim)
            attended_hypothesis = weighted_sum(embedded_hypothesis, p2h_attention)

            # Shape: (batch_size, hypothesis_length, premise_length)
            h2p_attention = masked_softmax(similarity_matrix.transpose(1, 2).contiguous(), mask_premise)
            # Shape: (batch_size, hypothesis_length, embedding_dim)
            attended_premise = weighted_sum(embedded_premise, h2p_attention)

            premise_compare_input = torch.cat([embedded_premise, attended_hypothesis], dim=-1)
            hypothesis_compare_input = torch.cat([embedded_hypothesis, attended_premise], dim=-1)
            
            compared_premise = self._compare_feedforward(premise_compare_input)
            compared_premise = compared_premise * mask_premise.unsqueeze(-1)
            # Shape: (batch_size, compare_dim)
            compared_premise = compared_premise.sum(dim=1)

            compared_hypothesis = self._compare_feedforward(hypothesis_compare_input)
            compared_hypothesis = compared_hypothesis * mask_hypothesis.unsqueeze(-1)
            # Shape: (batch_size, compare_dim)
            compared_hypothesis = compared_hypothesis.sum(dim=1)

            aggregate_input = torch.cat([compared_premise, compared_hypothesis], dim=-1)
            logits = self._aggregate_feedforward(aggregate_input)
            probs = torch.nn.functional.softmax(logits, dim=-1)
        else:
            raise ConfigurationError("One of premise or hypothesis is None. Check your DatasetReader")

        output_dict = {"logits": logits, 
                        "probs": probs, 
                       "h2p_attention": h2p_attention,
                       "p2h_attention": p2h_attention}

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
        avg = sum(accs) / sum(x > 0 for x in accs)
        self._nli_avg_acc(avg)
        metrics["nli-avg"] = self._nli_avg_acc.get_metric(reset)

        return metrics
