from typing import Dict, Optional, List, Any, Union

import torch

from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward
from allennlp.modules import Seq2SeqEncoder, SimilarityFunction, TimeDistributed, TextFieldEmbedder
from allennlp.modules.matrix_attention.legacy_matrix_attention import LegacyMatrixAttention
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, masked_softmax, weighted_sum
from allennlp.training.metrics import CategoricalAccuracy

from overrides import overrides
import torch
from pytorch_pretrained_bert.modeling import BertModel

from allennlp.models.model import Model
from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertModel
from allennlp.nn.initializers import InitializerApplicator
from allennlp.training.metrics import CategoricalAccuracy


XNLI_LANGS = ['ar', 'bg', 'de', 'el', 'en', 'es', 'fr', 'hi', 'ru', 'sw', 'th', 'tr', 'ur', 'vi', 'zh']

@Model.register("xnli_bert_pairs")
class XnliBertPairs(Model):
    """
    Uses sentence pairs as BERT suggests
    Parameters
    ----------
    vocab : ``Vocabulary``
    text_field_embedder : ``TextFieldEmbedder``
        Used to embed the ``premise`` and ``hypothesis`` ``TextFields`` we get as input to the
        model.
    attend_feedforward : ``FeedForward``
        This feedforward network is applied to the encoded sentence representations before the
        similarity matrix is computed between words in the premise and words in the hypothesis.
    similarity_function : ``SimilarityFunction``
        This is the similarity function used when computing the similarity matrix between words in
        the premise and words in the hypothesis.
    compare_feedforward : ``FeedForward``
        This feedforward network is applied to the aligned premise and hypothesis representations,
        individually.
    aggregate_feedforward : ``FeedForward``
        This final feedforward network is applied to the concatenated, summed result of the
        ``compare_feedforward`` network, and its output is used as the entailment class logits.
    premise_encoder : ``Seq2SeqEncoder``, optional (default=``None``)
        After embedding the premise, we can optionally apply an encoder.  If this is ``None``, we
        will do nothing.
    hypothesis_encoder : ``Seq2SeqEncoder``, optional (default=``None``)
        After embedding the hypothesis, we can optionally apply an encoder.  If this is ``None``,
        we will use the ``premise_encoder`` for the encoding (doing nothing if ``premise_encoder``
        is also ``None``).
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """
    def __init__(self, 
                 vocab: Vocabulary,
                 bert_model: str = "bert-base-multilingual-cased",
                 dropout: float = 0.0,
                 index: str = "bert",
                 trainable: bool = True,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(XnliBertPairs, self).__init__(vocab, regularizer)
        self.bert_model = PretrainedBertModel.load(bert_model)

        for param in self.bert_model.parameters():
            param.requires_grad = trainable

        #  we concatenate premise and hypothesis

        self._label_namespace = "labels"
        self._num_labels = vocab.get_vocab_size(namespace=self._label_namespace)

        in_features = self.bert_model.config.hidden_size
        self._classification_layer = torch.nn.Linear(in_features, self._num_labels)

        self._dropout = torch.nn.Dropout(p=dropout)

        self._loss = torch.nn.CrossEntropyLoss()
        self._accuracy = CategoricalAccuracy()

        self._index = index

        initializer(self._classification_layer)

        self._per_lang_accs = dict()
        for lang in XNLI_LANGS:
            self._per_lang_accs[lang] = CategoricalAccuracy()

    def forward(self,  # type: ignore
                premise_hypothesis: Dict[str, torch.LongTensor],
                label: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        premise_hypothesis : Dict[str, torch.LongTensor]
            From a ``TextField``
        label : torch.IntTensor, optional, (default = None)
            From a ``LabelField``
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

        ids_premise_hypothesis = premise_hypothesis[self._index]
        
        type_ids_premise_hypothesis = premise_hypothesis[f"{self._index}-type-ids"]
        
        mask_premise_hypothesis = (ids_premise_hypothesis != 0).long()

        _, pooled_premise_hypothesis = self.bert_model(input_ids=ids_premise_hypothesis,
                                    token_type_ids=type_ids_premise_hypothesis,
                                    attention_mask=mask_premise_hypothesis)

        pooled_premise_hypothesis = self._dropout(pooled_premise_hypothesis)
        
        logits = self._classification_layer(pooled_premise_hypothesis)
        probs = torch.nn.functional.softmax(logits, dim=-1)

        output_dict = {"logits": logits, "probs": probs}

        if label is not None:
            loss = self._loss(logits, label.long().view(-1))
            output_dict["loss"] = loss
            self._accuracy(logits, label)
            
            if "language" in metadata[0]: # we validate on multiple languages 
                batch_lang = metadata[0]["language"] # assumes homogenious batches
                self._per_lang_accs[batch_lang](logits, label)

        if metadata is not None:
            output_dict["premise_tokens"] = [x["premise_tokens"] for x in metadata]
            output_dict["hypothesis_tokens"] = [x["hypothesis_tokens"] for x in metadata]

            if "language" in metadata[0]: # we validate on multiple languages 
                output_dict["language"] = [x["language"] for x in metadata]

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
        metrics = {'accuracy': self._accuracy.get_metric(reset)}

        if not self.training: # we validate on multiple languages 
            for lang in XNLI_LANGS:
                metrics[lang] = self._per_lang_accs[lang].get_metric(reset)

        return metrics