import json
import logging
from typing import Dict, Iterable
import os

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
from pytorch_transformers.tokenization_auto import AutoTokenizer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


_VALID_SCHEMES = {"round_robin", "all_at_once"}

# TODO: unify with XNLI; just hardcore "en" as language tag

@DatasetReader.register("aligner_reader_xnli")
class AlignerReaderXnli(DatasetReader):
    """
    # NOTE: filepath in read should point to the dir with para files structured as in XLM repo

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
                 xlm_model_name: str,
                 do_lowercase: bool,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_sent_len: int = 128,
                 dataset_field_name: str = "dataset",
                 source_fname_prefix: str = "multinli.train.",
                 lg_pairs: str = "ar-en bg-en de-en el-en en-es en-fr en-hi en-ru en-sw en-th en-tr en-ur en-vi en-zh",
                 target_lang="en", scheme: str = "round_robin", lazy: bool = False) -> None:
        super().__init__(lazy)
        tokenizer = AutoTokenizer.from_pretrained(xlm_model_name, do_lower_case=do_lowercase)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

        self._max_sent_len = max_sent_len
        self._dataset_field_name = dataset_field_name
        self._lg_pairs = lg_pairs.split(" ")

        self._scheme = scheme
        self._readers: Dict[str, DatasetReader] = {}
        for pair in self._lg_pairs:
            self._readers[pair] = ParaCorpusReader(xlm_tokenizer=tokenizer, lang_pair=pair, xlm_model_name=xlm_model_name, do_lowercase=do_lowercase, 
                                                 token_indexers=token_indexers, max_sent_len=max_sent_len, 
                                                 dataset_field_name=dataset_field_name, target_lang=target_lang, lazy=lazy, 
                                                 source_fname_prefix=source_fname_prefix)

    def _read_round_robin(self, datasets: Dict[str, Iterable[Instance]]) -> Iterable[Instance]:
        remaining = set(datasets)
        dataset_iterators = {key: iter(dataset) for key, dataset in datasets.items()}

        while remaining:
            for key, dataset in dataset_iterators.items():
                if key in remaining:
                    try:
                        instance = next(dataset)
                        yield instance
                    except StopIteration:
                        remaining.remove(key)

    def _read_all_at_once(self, datasets: Dict[str, Iterable[Instance]]) -> Iterable[Instance]:
        for key, dataset in datasets.items():
            for instance in dataset:
                yield instance


    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        # NOTE: filepath should point to the dir with para files structured as in XLM repo
        data_dir_path = file_path
        if data_dir_path[-1] != "/":
            data_dir_path = data_dir_path + "/"
        filenames = os.listdir(data_dir_path)

        # open filestreams
        pair2fpath = {}
        for fname in filenames:
            pair2fpath[fname.split(".")[0]] = data_dir_path + fname
  
        # Load datasets
        datasets = {pair: reader.read(data_dir_path) for pair, reader in self._readers.items()}

        if self._scheme == "round_robin":
            yield from self._read_round_robin(datasets)
        elif self._scheme == "all_at_once":
            yield from self._read_all_at_once(datasets)
        else:
            raise RuntimeError("impossible to get here")

    def text_to_instance(self) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        raise RuntimeError("text_to_instance doesn't make sense here")



class ParaCorpusReader(DatasetReader):
    """
    # NOTE: filepath in read should point to the dir with para files structured as in XLM repo

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
                 xlm_tokenizer: AutoTokenizer,
                 lang_pair: str,
                 xlm_model_name: str,
                 do_lowercase: bool,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_sent_len: int = 80,
                 dataset_field_name: str = "dataset",
                 target_lang: str = "en",
                 source_fname_prefix: str = "multinli.train.",
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._tokenizer = xlm_tokenizer
        self._lang_pair = lang_pair
        
        l1, l2 = lang_pair.split("-")
        self._target_lang = target_lang
        if l1 == target_lang:
            self._source_lang = l2
        elif l2 == target_lang:
            self._source_lang = l1
        else:
            raise ValueError

        self._source_fname_prefix = source_fname_prefix + self._source_lang
        self._target_fname_prefix = source_fname_prefix + self._target_lang

        self._max_sent_len = max_sent_len
        self._dataset_field_name = dataset_field_name

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        # NOTE: filepath should point to the dir with para files structured as in XLM repo
        data_dir_path = file_path


        if data_dir_path[-1] != "/":
            data_dir_path = data_dir_path + "/"
        filenames = os.listdir(data_dir_path)

        source_fname = [data_dir_path + n for n in filenames if n.startswith(self._source_fname_prefix)][0]
        target_fname = [data_dir_path + n for n in filenames if n.startswith(self._target_fname_prefix)][0]

        with open(source_fname, 'r') as src_file, open(target_fname, 'r') as tgt_file:
            logger.info("Reading para lines from: %s and %s", source_fname, target_fname)
            
            src1 = None
            tgt1 = None
            src2 = None
            tgt2 = None
            
            second_pair = False
            for src, tgt in zip(src_file, tgt_file):
                src_dict = json.loads(src)
                assert src_dict["language"] == self._source_lang
                label_src = src_dict["gold_label"]
                if label_src == '-':
                    # These were cases where the annotators disagreed; we'll just skip them.  It's
                    # like 800 out of 400k examples in the training data.
                    continue
                src1 = src_dict["sentence1"]
                src2 = src_dict["sentence2"]

                tgt_dict = json.loads(tgt)
                assert self._target_lang == tgt_dict["language"]
                label_tgt = src_dict["gold_label"]
                if label_src == '-':
                    # These were cases where the annotators disagreed; we'll just skip them.  It's
                    # like 800 out of 400k examples in the training data.
                    continue
                tgt1 = tgt_dict["sentence1"]
                tgt2 = tgt_dict["sentence2"]

                assert label_src == label_tgt
                try:
                    yield self.text_to_instance(src1, src2, tgt1, tgt2, label_tgt)
                except ValueError:
                    continue
                    



    @overrides
    def text_to_instance(self,  # type: ignore
                         src1: str, src2: str,
                         tgt1: str, tgt2: str,
                         label: str = None) -> Instance:
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        src_tokens1 = self._tokenizer.tokenize(src1, lang=self._source_lang)
        src_tokens1 = [Token(t) for t in src_tokens1]
        
        src_tokens2 = self._tokenizer.tokenize(src2, lang=self._source_lang)
        src_tokens2 = [Token(t) for t in src_tokens2]
        
        tgt_tokens1 = self._tokenizer.tokenize(tgt1, lang=self._target_lang)
        tgt_tokens1 = [Token(t) for t in tgt_tokens1]

        tgt_tokens2 = self._tokenizer.tokenize(tgt2, lang=self._target_lang)
        tgt_tokens2 = [Token(t) for t in tgt_tokens2]
        
        # filter out very long sentences
        if self._max_sent_len is not None:
            # These were sentences are too long for bert; we'll just skip them.  It's
            # like 1000 out of 400k examples in the training data.

            if len(src_tokens1) + len(src_tokens2) > self._max_sent_len or len(tgt_tokens1) + len(tgt_tokens2) > self._max_sent_len:
                raise ValueError()
        
        src_tokens = [Token("</s>")] + src_tokens1 + [Token("</s>")] + src_tokens2 + [Token("</s>")]
        tgt_tokens = [Token("</s>")] + tgt_tokens1 + [Token("</s>")] + tgt_tokens2 + [Token("</s>")]

        fields['src_tokens'] = TextField(src_tokens, self._token_indexers)
        fields['tgt_tokens'] = TextField(tgt_tokens, self._token_indexers)
        
        if label:
            if label == 'contradictory':
                label = 'contradiction'
            fields['label'] = LabelField(label)
    
        metadata = {"src_tokens": [x.text for x in src_tokens],
                    "tgt_tokens": [x.text for x in tgt_tokens]}
        fields["metadata"] = MetadataField(metadata)

        fields["target_language"] = MetadataField(metadata=self._target_lang)
        fields[self._dataset_field_name] = MetadataField(metadata=self._source_lang)



        return Instance(fields)

