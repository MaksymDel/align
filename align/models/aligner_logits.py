from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn.functional as F
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
from allennlp.nn import RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, masked_softmax, weighted_sum
from allennlp.training.metrics import Average, CategoricalAccuracy
from overrides import overrides
from pytorch_pretrained_bert.modeling import BertModel
from allennlp.models.archival import load_archive

from align.models.simple_projection_xlm import SimpleProjectionXlm

# TODO: loss weights based on MT system quolity
# TODO: multisource translation (translate test opponent)
# TODO: look at other papers from ruder's blog for stuff to report

@Model.register("aligner_logits")
class AlignerLogits(Model):
    """

    """
    def __init__(self, 
                 vocab: Vocabulary,
                 student_xlm: TextFieldEmbedder,
                 teacher_xlm: TextFieldEmbedder,
                 labels_vocab_file: str,
                 training_tasks: Any,
                 validation_tasks: Any,
                 student_nli_head: FeedForward,
                 teacher_nli_head: FeedForward,
                 projector_feedforward: FeedForward = None,
                 loss: str = "l1",
                 reduction: str = "mean",
                 training_tasks_2print: List[str] = ["en", "de", "ru", "fr", "ur", "sw"],
                 valid_langs_2print: List[str]= ["en", "de", "ur", "sw", "ru"],
                 dropout: float = 0.0,
                 regularizer: Optional[RegularizerApplicator] = None,
                 feed_lang_ids: bool = True,
                 avg: bool = False) -> None:

        vocab.set_from_file(filename=labels_vocab_file, is_padded=False, namespace="labels")

        super(AlignerLogits, self).__init__(vocab, regularizer)

        self._avg = avg

        self._teacher_xlm = teacher_xlm
        self._student_xlm = student_xlm

        self._teacher_nli_head = teacher_nli_head
        self._student_nli_head = student_nli_head


        self._projector_feedforward = projector_feedforward
        if projector_feedforward is not None:
            assert projector_feedforward.get_input_dim() == student_xlm.get_output_dim()
            assert projector_feedforward.get_output_dim() == teacher_xlm.get_output_dim()

        if type(training_tasks) == dict:
            self._training_tasks = list(training_tasks.keys())
        else:
            self._training_tasks = training_tasks

        if type(validation_tasks) == dict:
            self._validation_tasks = list(validation_tasks.keys())
        else:
            self._validation_tasks = validation_tasks

        # self._src_embedder =  

        self._dropout = torch.nn.Dropout(p=dropout)
        
        if loss == "l1":
            self._loss = torch.nn.L1Loss(reduction=reduction)
        elif loss == "mse":
            self._loss = torch.nn.MSELoss(reduction=reduction)
        elif loss == "cos":
            self._loss = torch.nn.CosineEmbeddingLoss(reduction=reduction)
        elif loss == "smooth_l1":
            self._loss = torch.nn.SmoothL1Loss(reduction=reduction)
        else:
            raise NotImplementedError # TODO: try margin based losses

        self._per_lang_align_loss: Dict[str, Average] = dict()
        
        for taskname in self._training_tasks:
            # this will hide some metrics from tqdm, but they will still be computed
            self._per_lang_align_loss[taskname] = Average()
        self._avg_loss = Average()
        
        self._langs_pring_train = training_tasks_2print or "en"
        self._langs_print_val = valid_langs_2print
        if '*' in self._langs_pring_train:
            self._langs_pring_train = [t.split("")[-1] for t in training_tasks] 
        
        self._feed_lang_ids = feed_lang_ids   

        self._nli_per_lang_acc: Dict[str, CategoricalAccuracy] = dict()
        for taskname in self._validation_tasks:
            # this will hide some metrics from tqdm, but they will still be computed
            self._nli_per_lang_acc[taskname] = CategoricalAccuracy()
        self._nli_avg_acc = Average()
    
    def pool(self, sentence: torch.Tensor):
        if not self._avg:
            return sentence[:, 0, :]
            #pooled_combined = self._pooler(embedded_combined, mask=mask)
        else:
            return sentence.mean(dim=1) # multiply by mask also

    def encode_project(self, src_tokens, lang_ids_src):
        # 2) get CLS embedding for source sentence
        embedded_src = self._student_xlm(src_tokens, lang=lang_ids_src)        
        pooled_src = self.pool(embedded_src)

        # 3) optionaly project 
        # NOTE: projection is necessury if dimentions mismatch or if student is also frozen
        pooled_src = self._dropout(pooled_src)
        
        if self._projector_feedforward is not None:
            pooled_src = self._projector_feedforward(pooled_src)
        
        
        return pooled_src

    def fwd_align(self,
            src_tokens: Dict[str, torch.Tensor],
            tgt_tokens: Dict[str, torch.Tensor],
            target_language: List[str],
            dataset: List[str],
            metadata: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        """
        lang_src = dataset[0]
        lang_tgt = target_language[0]

        mask_src = get_text_field_mask(src_tokens).float()
        mask_tgt = get_text_field_mask(tgt_tokens).float()

        # 0) prepare lang ids if needed
        lang_ids_src = None
        lang_ids_tgt = None
        if self._feed_lang_ids:
            lang_ids_src = self._student_xlm._token_embedders["bert"].transformer_model.config.lang2id[lang_src]
            lang_ids_tgt = self._teacher_xlm._token_embedders["bert"].transformer_model.config.lang2id[lang_tgt]

            lang_ids_src = mask_src.new_full(mask_src.size(), lang_ids_src).long()
            lang_ids_tgt = mask_tgt.new_full(mask_tgt.size(), lang_ids_tgt).long()

        # 1) get gold CLS embedding from target sentence with teach mode
        with torch.no_grad(): # TODO: try encoding src here and see if models learns to do nothing
            embedded_tgt = self._teacher_xlm(tgt_tokens, lang=lang_ids_tgt)
            pooled_tgt = self.pool(embedded_tgt)
            teacher_logits = self._teacher_nli_head(pooled_tgt)

        
        pooled_src = self.encode_project(src_tokens, lang_ids_src)
        student_logits = self._student_nli_head(pooled_src)

        loss = self._loss(student_logits, teacher_logits.detach())
            
        self._per_lang_align_loss[lang_src](loss.item())

        return {"loss": loss}

    def fwd_xnli(self,  # type: ignore
                premise_hypothesis: Dict[str, torch.Tensor],
                dataset: List[str],
                label: torch.IntTensor,
                metadata: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:

        with torch.no_grad():
            mask = get_text_field_mask(premise_hypothesis).float()
            lang = dataset[0].split('-')[-1]
            lang_id = self._student_xlm._token_embedders["bert"].transformer_model.config.lang2id[lang]
            lang_ids = mask.new_full(mask.size(), lang_id).long()
            # pooled_combined = self.encode_src(premise_hypothesis, lang_ids)
            pooled_combined = self.encode_project(premise_hypothesis, lang_ids)
            logits = self._student_nli_head(pooled_combined)
            probs = torch.nn.functional.softmax(logits, dim=-1)
            output_dict = {"logits": logits, "probs": probs}

            output_dict["cls_emb"] = pooled_combined

            #loss = self._loss(logits, label.long().view(-1))

            #output_dict["loss"] = loss            
            self._nli_per_lang_acc[dataset[0]](logits, label)

        return output_dict




    def forward(self,  # type: ignore
                src_tokens: Dict[str, torch.Tensor]=None,
                tgt_tokens: Dict[str, torch.Tensor]=None,
                target_language: List[str]=None,
                dataset: List[str]=None,
                metadata: List[Dict[str, Any]] = None,
                premise_hypothesis=None,
                label: torch.IntTensor=None
                ) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        """
        if self.training:
            return self.fwd_align(src_tokens=src_tokens, 
                    tgt_tokens=tgt_tokens, 
                    target_language=target_language, 
                    dataset=dataset,
                    metadata=metadata)
        else:
            return self.fwd_xnli(premise_hypothesis=premise_hypothesis, dataset=dataset, label=label, metadata=metadata)


    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {}

        if self.training:
            tasks = self._training_tasks
    
            for taskname in tasks:
                metricname = taskname
                if metricname.split("-")[-1] not in self._langs_pring_train: # hide other langs from tqdn
                    metricname = '_' + metricname
                metrics[metricname] = self._per_lang_align_loss[taskname].get_metric(reset)
            accs = metrics.values() 
            avg = sum(accs) / len(tasks)
            self._avg_loss(avg)
            metrics["avg"] = self._avg_loss.get_metric(reset)

        else:
            tasks = self._validation_tasks
            for taskname in tasks:
                metricname = taskname
                if metricname.split("-")[-1] not in self._langs_print_val:
                    metricname = '_' + metricname
                metrics[metricname] = self._nli_per_lang_acc[taskname].get_metric(reset)
            accs = metrics.values() # TODO: should only count 'nli-*' metrics
            avg = sum(accs) / len(tasks)
            self._nli_avg_acc(avg)
            metrics["nli-avg"] = self._nli_avg_acc.get_metric(reset)
        return metrics
