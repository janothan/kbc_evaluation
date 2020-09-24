import logging
import os
from kbc_evaluation.dataset import DataSet, ParsedSet

# noinspection PyArgumentList
logging.basicConfig(handlers=[logging.FileHandler(__file__ + '.log', 'w', 'utf-8'), logging.StreamHandler()],
                    format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG)


class Evaluator:

    def __init__(self, file_to_be_evaluated: str, is_apply_filtering: bool = False):
        """Constructor

        Parameters
        ----------
        file_to_be_evaluated : str
            Path to the text file with the predicted links that shall be evaluated.
        is_apply_filtering : bool
            Indicates whether filtering is desired (if True, results will likely improve).
        """

        self._file_to_be_evaluated = file_to_be_evaluated
        self._is_apply_filtering = is_apply_filtering

        if file_to_be_evaluated is None or not os.path.isfile(file_to_be_evaluated):
            logging.error(f"The evaluator will not work because the specified file "
                          f"does not exist {file_to_be_evaluated}")
            raise Exception(f"The specified file ({file_to_be_evaluated}) does not exist.")

        self.parsed = ParsedSet(is_apply_filtering=self._is_apply_filtering,
                                file_to_be_evaluated=self._file_to_be_evaluated)

    def mean_rank(self) -> float:
        ignored_heads = 0
        ignored_tails = 0
        head_rank = 0
        tail_rank = 0

        for truth, prediction in self.parsed.triple_predictions.items():
            try:
                h_index = prediction[0].index(truth[0]) + 1  # (first position has index 0)
                head_rank += h_index
            except ValueError:
                ignored_heads += 1
            try:
                t_index = prediction[1].index(truth[2]) + 1  # (first position has index 0)
                tail_rank += t_index
            except ValueError:
                ignored_tails += 1

        mean_head_rank = 0
        mean_tail_rank = 0
        total_tasks = self.parsed.total_prediction_tasks
        if total_tasks - ignored_heads > 0:
            mean_head_rank = head_rank / (total_tasks/2 - ignored_heads)
        if total_tasks/2 - ignored_tails > 0:
            mean_tail_rank = tail_rank / (total_tasks/2 - ignored_tails)

        logging.info(f"Mean Head Rank: {mean_head_rank} ({ignored_heads} ignored lines)")
        logging.info(f"Mean Tail Rank: {mean_tail_rank} ({ignored_tails} ignored lines)")

        mean_rank = 0
        if (total_tasks - ignored_tails - ignored_heads) > 0:
            mean_rank = (head_rank + tail_rank) / (total_tasks - ignored_tails - ignored_heads)
        logging.info(f"Mean rank: {mean_rank}")
        mean_rank = round(mean_rank)
        return mean_rank

    def calculate_hits_at(self, n: int = 10) -> int:
        """Calculation of hits@n.

        Parameters
        ----------
        n : int
            Hits@n. This parameter specifies the n.

        Returns
        -------
        int
            The hits at n. Note that head hits and tail hits are added.
        """
        heads_hits = 0
        tails_hits = 0

        for truth, prediction in self.parsed.triple_predictions.items():
            # perform the actual evaluation
            if truth[0] in prediction[0][:(n + 1)]:
                heads_hits += 1
            if truth[2] in prediction[1][:(n + 1)]:
                tails_hits += 1

        result = heads_hits + tails_hits
        logging.info(f"Hits@{n} Heads: {heads_hits}")
        logging.info(f"Hits@{n} Tails: {tails_hits}")
        logging.info(f"Hits@{n} Total: {result}")
        return result


if __name__ == "__main__":
    evaluator = Evaluator(
        file_to_be_evaluated="/Users/janportisch/PycharmProjects/KBC_RDF2Vec/wn_evaluation_file_5000.txt",
        is_apply_filtering=True)
    hits_at_10 = evaluator.calculate_hits_at(10)
    test_set_size = len(DataSet.WN18.test_set())
    mr = evaluator.mean_rank()
    print(f"Test set size: {test_set_size}")
    print(f"Hits at 10: {hits_at_10}")
    print(f"Relative Hits at 10: {hits_at_10 / (2 * test_set_size)}")  # really 2 x test set size
    print(f"Mean rank: {mr}")

