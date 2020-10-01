import os

import pytest

from kbc_evaluation.dataset import DataSet
from kbc_evaluation.evaluator import Evaluator


class TestEvaluator:
    def test_evaluator_failure(self):
        with pytest.raises(Exception):
            Evaluator(file_to_be_evaluated="xzy")
        with pytest.raises(Exception):
            Evaluator()

    def test_hits_at(self):
        test_file_path = "./tests/test_resources/eval_test_file.txt"
        if not os.path.isfile(test_file_path):
            os.chdir("./..")
        assert os.path.isfile(test_file_path)

        evaluator = Evaluator(file_to_be_evaluated=test_file_path)
        assert evaluator.calculate_hits_at(1) == 2
        assert evaluator.calculate_hits_at(3) == 3
        assert evaluator.calculate_hits_at(10) == 4

    def test_hits_at_with_confidence(self):
        test_file_path = "./tests/test_resources/eval_test_file_with_confidences.txt"
        if not os.path.isfile(test_file_path):
            os.chdir("./..")
        assert os.path.isfile(test_file_path)

        evaluator = Evaluator(file_to_be_evaluated=test_file_path)
        assert evaluator.calculate_hits_at(1) == 2
        assert evaluator.calculate_hits_at(3) == 3
        assert evaluator.calculate_hits_at(10) == 4

    def test_hits_at_filtering(self):
        test_file_path = "./tests/test_resources/eval_test_file_filtering.txt"
        if not os.path.isfile(test_file_path):
            os.chdir("./..")
        assert os.path.isfile(test_file_path)

        evaluator = Evaluator(
            file_to_be_evaluated=test_file_path, is_apply_filtering=True
        )
        assert evaluator.calculate_hits_at(1) == 2
        assert evaluator.calculate_hits_at(3) == 4
        assert evaluator.calculate_hits_at(10) == 6

    def test_hits_at_filtering_with_confidence(self):
        test_file_path = (
            "./tests/test_resources/eval_test_file_filtering_with_confidences.txt"
        )
        if not os.path.isfile(test_file_path):
            os.chdir("./..")
        assert os.path.isfile(test_file_path)

        evaluator = Evaluator(
            file_to_be_evaluated=test_file_path, is_apply_filtering=True
        )
        assert evaluator.calculate_hits_at(1) == 2
        assert evaluator.calculate_hits_at(3) == 4
        assert evaluator.calculate_hits_at(10) == 6

    def test_mean_rank(self):
        test_file_path = "./tests/test_resources/eval_test_file.txt"
        assert os.path.isfile(test_file_path)
        evaluator = Evaluator(file_to_be_evaluated=test_file_path)
        assert evaluator.mean_rank() == 3

    def test_mean_rank_with_confidence(self):
        test_file_path = "./tests/test_resources/eval_test_file_with_confidences.txt"
        assert os.path.isfile(test_file_path)
        evaluator = Evaluator(file_to_be_evaluated=test_file_path)
        assert evaluator.mean_rank() == 3

    def test_write_results_to_file(self):
        test_file_path = "./tests/test_resources/eval_test_file.txt"
        assert os.path.isfile(test_file_path)
        Evaluator.write_results_to_file(
            file_to_be_evaluated=test_file_path, data_set=DataSet.WN18
        )
        assert os.path.isfile("./results.txt")
        os.remove("./results.txt")
        Evaluator.write_results_to_file(
            file_to_be_evaluated=test_file_path,
            file_to_be_written="./results_test.txt",
            data_set=DataSet.WN18,
        )
        assert os.path.isfile("./results_test.txt")
        os.remove("./results_test.txt")
