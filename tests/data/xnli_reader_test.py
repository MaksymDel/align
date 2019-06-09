# pylint: disable=no-self-use,invalid-name
import pytest

from align.data.xnli_reader import XnliReader
from allennlp.common.util import ensure_list
from allennlp.common.testing import AllenNlpTestCase

class TestXnliReader():
    # @pytest.mark.parametrize("lazy", (True, False))
    def test_read_from_file(self, lazy=False):
        reader = XnliReader(lazy=lazy)
        instances = reader.read('fixtures/data/xnli.jsonl')
        instances = ensure_list(instances)

        instance1 = {"premise": ["And", "he", "said", ",", "Mama", ",", "I", "'m", "home", "."],
                     "hypothesis": ['He', 'called', 'his', 'mom', 'as', 'soon', 'as', 'the', 'school', 'bus', 'dropped', 'him', 'off', "."],
                     "label": "neutral"}

        instance2 = {"premise": ['Es', 'fallen', 'zwanzig', 'Prozent', 'Zinsen', 'an'],
                     "hypothesis": ['K\u00f6nnte', 'das', 'Interesse', 'mehr', 'als', '20', 'sein', '?'],
                     "label": "neutral"}


        assert len(instances) == 3
        fields = instances[0].fields
        assert [t.text for t in fields["premise"].tokens] == instance1["premise"]
        assert [t.text for t in fields["hypothesis"].tokens] == instance1["hypothesis"]
        assert fields["label"].label == instance1["label"]
        fields = instances[1].fields
        assert [t.text for t in fields["premise"].tokens] == instance2["premise"]
        assert [t.text for t in fields["hypothesis"].tokens] == instance2["hypothesis"]
        assert fields["label"].label == instance2["label"]
