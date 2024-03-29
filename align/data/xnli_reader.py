import json
import logging
from typing import Dict

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data import Token
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, LabelField, MetadataField, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter
from overrides import overrides

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

# TODO: unify with XNLI; just hardcore "en" as language tag

@DatasetReader.register("xnli")
class XnliReader(DatasetReader):
    """
    Reads a file from the Multi Natural Language Inference (MNLI) dataset.  This data is
    formatted as jsonl, one json-formatted instance per line.  The keys in the data are
    "gold_label", "sentence1", and "sentence2".  We convert these keys into fields named "label",
    "premise" and "hypothesis", along with a metadata field containing the tokenized strings of the
    premise and hypothesis.

    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional (default=``WordTokenizer()``)
        We use this ``Tokenizer`` for both the premise and the hypothesis.  See :class:`Tokenizer`.
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We similarly use this for both the premise and the hypothesis.  See :class:`TokenIndexer`.
    max_sent_len : ``int``
        Examples where premis or hypothesis are larger then this will be filtered out
    """

    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 bert_format: bool = False,  
                 max_sent_len: int = 80,
                 dataset_field_name: str = "dataset",
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer(word_splitter=JustSpacesWordSplitter())
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._max_sent_len = max_sent_len
        self._bert_format = bert_format
        self._dataset_field_name = dataset_field_name 

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, 'r') as nli_file:
            logger.info("Reading XNLI / MultiNLI / SNLI instances from jsonl dataset at: %s", file_path)
            for line in nli_file:
                example = json.loads(line)

                language = "en" # by default we assume we work with MNLI which is in en  
                if "language" in example:
                    language = example["language"] # xnli

                label = example["gold_label"]
                if label == '-':
                    # These were cases where the annotators disagreed; we'll just skip them.  It's
                    # like 800 out of 400k examples in the training data.
                    continue


                premise = example["sentence1"]
                hypothesis = example["sentence2"]

                try:
                    yield self.text_to_instance(premise, hypothesis, language, label=label)
                except ValueError:
                    continue

    @overrides
    def text_to_instance(self,  # type: ignore
                         premise: str,
                         hypothesis: str,
                         language: str,
                         label: str = None) -> Instance:
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        premise_tokens = self._tokenizer.tokenize(premise)
        hypothesis_tokens = self._tokenizer.tokenize(hypothesis)

        # filter out very long sentences
        if self._max_sent_len is not None:
            # These were sentences are too long for bert; we'll just skip them.  It's
            # like 1000 out of 400k examples in the training data.

            if len(premise_tokens) + len(hypothesis_tokens) > self._max_sent_len:
                raise ValueError()

        if self._bert_format:
            premise_hypothesis_tokens = premise_tokens + [Token("[SEP]")] + hypothesis_tokens
            fields['premise_hypothesis'] = TextField(premise_hypothesis_tokens, self._token_indexers)
        else:
            
            fields['premise'] = TextField(premise_tokens, self._token_indexers)
            fields['hypothesis'] = TextField(hypothesis_tokens, self._token_indexers)
        
        if label:
            if label == 'contradictory':
                label = 'contradiction'
            fields['label'] = LabelField(label)
        
        metadata = {"premise_tokens": [x.text for x in premise_tokens],
                    "hypothesis_tokens": [x.text for x in hypothesis_tokens]}
        fields["metadata"] = MetadataField(metadata)

        fields[self._dataset_field_name] = MetadataField(metadata="nli-" + language)

        return Instance(fields)
