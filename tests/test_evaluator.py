import os

import pytest

from kbc_evaluation.dataset import DataSet
from kbc_evaluation.evaluator import EvaluationRunner, Evaluator


class TestEvaluator:
    def test_evaluator_failure(self):
        with pytest.raises(Exception):
            EvaluationRunner(file_to_be_evaluated="xzy")
        with pytest.raises(Exception):
            EvaluationRunner()

    def test_hits_at(self):
        test_file_path = "./tests/test_resources/eval_test_file.txt"
        if not os.path.isfile(test_file_path):
            os.chdir("./..")
        assert os.path.isfile(test_file_path)

        runner = EvaluationRunner(file_to_be_evaluated=test_file_path)
        assert runner.calculate_hits_at(1)[2] == 2
        assert runner.calculate_hits_at(3)[2] == 3
        assert runner.calculate_hits_at(10)[2] == 4

    def test_hits_at_with_confidence(self):
        test_file_path = "./tests/test_resources/eval_test_file_with_confidences.txt"
        if not os.path.isfile(test_file_path):
            os.chdir("./..")
        assert os.path.isfile(test_file_path)

        runner = EvaluationRunner(file_to_be_evaluated=test_file_path)
        assert runner.calculate_hits_at(1)[2] == 2
        assert runner.calculate_hits_at(3)[2] == 3
        assert runner.calculate_hits_at(10)[2] == 4

    def test_calculate_results_no_filtering(self):
        test_file_path = "./tests/test_resources/eval_test_file_with_confidences.txt"
        if not os.path.isfile(test_file_path):
            os.chdir("./..")
        assert os.path.isfile(test_file_path)

        results = Evaluator.calculate_results(
            file_to_be_evaluated=test_file_path, data_set=DataSet.WN18, n=1
        )
        assert results.filtered_hits_at_n_all == 2
        assert results.filtered_hits_at_n_all >= results.filtered_hits_at_n_all
        assert results.n == 1

        # simple type assertions
        assert type(results.evaluated_file) == str
        assert type(results.n) == int
        assert type(results.filtered_hits_at_n_heads) == int
        assert type(results.filtered_hits_at_n_tails) == int
        assert type(results.filtered_hits_at_n_all) == int

        results = Evaluator.calculate_results(
            file_to_be_evaluated=test_file_path, data_set=DataSet.WN18, n=3
        )
        assert results.filtered_hits_at_n_all == 3
        assert results.filtered_hits_at_n_all >= results.filtered_hits_at_n_all
        assert results.n == 3

        results = Evaluator.calculate_results(
            file_to_be_evaluated=test_file_path, data_set=DataSet.WN18, n=10
        )
        assert results.filtered_hits_at_n_all == 4
        assert results.filtered_hits_at_n_all >= results.filtered_hits_at_n_all
        assert results.n == 10

        test_file_path = "./tests/test_resources/eval_test_file.txt"
        assert os.path.isfile(test_file_path)
        results = Evaluator.calculate_results(
            # n and data_set are irrelevant here
            file_to_be_evaluated=test_file_path,
            data_set=DataSet.WN18,
            n=4,
        )
        assert results.non_filtered_mean_rank_all == 3

    def test_hits_at_filtering(self):
        test_file_path = "./tests/test_resources/eval_test_file_filtering.txt"
        if not os.path.isfile(test_file_path):
            os.chdir("./..")
        assert os.path.isfile(test_file_path)

        runner = EvaluationRunner(
            file_to_be_evaluated=test_file_path, is_apply_filtering=True
        )
        assert runner.calculate_hits_at(1)[2] == 3
        assert runner.calculate_hits_at(3)[2] == 9
        assert runner.calculate_hits_at(10)[0] == 6
        assert runner.calculate_hits_at(10)[1] == 5
        assert runner.calculate_hits_at(10)[2] == 11

    def test_calculate_results_filtering(self):
        test_file_path = "./tests/test_resources/eval_test_file_filtering.txt"
        if not os.path.isfile(test_file_path):
            os.chdir("./..")
        assert os.path.isfile(test_file_path)

        results = Evaluator.calculate_results(
            file_to_be_evaluated=test_file_path, data_set=DataSet.WN18, n=1
        )
        assert results.filtered_hits_at_n_all == 3  # n is 1
        assert results.n == 1
        assert type(results.filtered_hits_at_n_all) == int  # n is 1
        assert type(results.filtered_hits_at_n_tails) == int
        assert type(results.filtered_hits_at_n_heads) == int

    def test_hits_at_filtering_with_confidence(self):
        test_file_path = (
            "./tests/test_resources/eval_test_file_filtering_with_confidences.txt"
        )
        if not os.path.isfile(test_file_path):
            os.chdir("./..")
        assert os.path.isfile(test_file_path)

        runner = EvaluationRunner(
            file_to_be_evaluated=test_file_path, is_apply_filtering=True
        )
        assert runner.calculate_hits_at(1)[2] == 2
        assert runner.calculate_hits_at(3)[2] == 4
        assert runner.calculate_hits_at(10)[2] == 6

    def test_mean_rank(self):
        test_file_path = "./tests/test_resources/eval_test_file.txt"
        assert os.path.isfile(test_file_path)
        runner = EvaluationRunner(file_to_be_evaluated=test_file_path)
        assert runner.mean_rank()[2] == 3

    def test_mean_rank_with_confidence(self):
        test_file_path = "./tests/test_resources/eval_test_file_with_confidences.txt"
        assert os.path.isfile(test_file_path)
        runner = EvaluationRunner(file_to_be_evaluated=test_file_path)
        assert runner.mean_rank()[2] == 3

    def test_mean_rank_with_filtering(self):
        test_file_path = "./tests/test_resources/eval_test_file_filtering.txt"
        if not os.path.isfile(test_file_path):
            os.chdir("./..")
        assert os.path.isfile(test_file_path)

        runner = EvaluationRunner(
            file_to_be_evaluated=test_file_path, is_apply_filtering=True
        )
        result = runner.mean_rank()

        # note that the result is rounded
        # as explained in eval_test_file_filtering_readme.md, MR for H is 3.333 and for T is 2.6
        # rounded, this is 3
        # the weighted (!) average is exactly three
        assert result[0] == 3  # MR head
        assert result[1] == 3  # MR tail
        assert result[2] == 3  # MR all

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
