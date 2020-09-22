import os

from evaluator import Evaluator


def test_hits_at():
    test_file_path = "./tests/test_resources/eval_test_file.txt"
    if os.path.isfile(test_file_path):
        pass
    else:
        test_file_path = "./test_resources/eval_test_file.txt"
        assert os.path.isfile(test_file_path)

    evaluator = Evaluator(file_to_be_evaluated=test_file_path)
    assert evaluator.calculate_hits_at(1) == 2
    assert evaluator.calculate_hits_at(3) == 3
    assert evaluator.calculate_hits_at(10) == 4


if __name__ == "__main__":
    test_hits_at()