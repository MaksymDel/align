from typing import Dict
import json
import logging

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, LabelField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("xnli")
class XnliReader(DatasetReader):
    """

    """

    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_sent_len: int = None, 
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._max_sent_len = max_sent_len

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, 'r') as snli_file:
            logger.info("Reading XNLI instances from jsonl dataset at: %s", file_path)
            for line in snli_file:
                example = json.loads(line)

                label = example["gold_label"]
                if label == '-':
                    continue

                premise = example["sentence1"]
                hypothesis = example["sentence2"]

                # filter out very long sentences
                if self._max_sent_len is not None:
                    # These were sentences are too long for bert; we'll just skip them.  It's
                    # like 1000 out of 400k examples in the training data.
                    if len(premise.split(" ")) > self._max_sent_len or len(hypothesis.split(" ")) > self._max_sent_len:
                        continue

                language = example["language"]
                yield self.text_to_instance(premise, hypothesis, language, label)

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
        fields['premise'] = TextField(premise_tokens, self._token_indexers)
        fields['hypothesis'] = TextField(hypothesis_tokens, self._token_indexers)
       
        if label:
            fields['label'] = LabelField(label)

        metadata = {"language": language,
                    "premise_tokens": [x.text for x in premise_tokens],
                    "hypothesis_tokens": [x.text for x in hypothesis_tokens]}

        fields["metadata"] = MetadataField(metadata)
        return Instance(fields)
