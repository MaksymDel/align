# pylint: disable=no-self-use,invalid-name
from flaky import flaky
import pytest
import numpy
from numpy.testing import assert_almost_equal

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import ModelTestCase
from allennlp.models import Model

from align.models.xnli_bert_pairs import XnliBertPairs

class TestXnliBertPairs(ModelTestCase):
    def setUp(self):
        super(TestXnliBertPairs, self).setUp()
        self.set_up_model('fixtures/xnli_bert_pairs.jsonnet',
                          'fixtures/data/mnli.jsonl')

    def test_forward_pass_runs_correctly(self):
        training_tensors = self.dataset.as_tensor_dict()
        output_dict = self.model(**training_tensors)
        assert_almost_equal(numpy.sum(output_dict["probs"][0].data.numpy(), -1), 1, decimal=6)

    @flaky
    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    @flaky
    def test_batch_predictions_are_consistent(self):
        self.ensure_batch_predictions_are_consistent()

    def test_model_load(self):
        params = Params.from_file("fixtures/xnli_bert_pairs.jsonnet")
        model = Model.load(params, serialization_dir="fixtures/serialization/bert_toy_pairs")

        assert isinstance(model, XnliBertPairs)
