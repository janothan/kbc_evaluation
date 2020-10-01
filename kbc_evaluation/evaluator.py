import logging.config
import os
from typing import Tuple

from kbc_evaluation.dataset import DataSet, ParsedSet

logging.config.fileConfig(fname="log.conf", disable_existing_loggers=False)
logger = logging.getLogger(__name__)


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
            logging.error(
                f"The evaluator will not work because the specified file "
                f"does not exist {file_to_be_evaluated}"
            )
            raise Exception(
                f"The specified file ({file_to_be_evaluated}) does not exist."
            )

        self.parsed = ParsedSet(
            is_apply_filtering=self._is_apply_filtering,
            file_to_be_evaluated=self._file_to_be_evaluated,
        )

    def mean_rank(self) -> Tuple[int, int, int]:
        """Calculates the mean rank using the given file.

        Returns
        -------
        Tuple[int, int, int]
            [0] Mean rank as int for heads (rounded float).
            [0] Mean rank as int for tails (rounded float).
            [0] Mean rank as int (rounded float).

        """
        print("Calculating Mean Rank")
        ignored_heads = 0
        ignored_tails = 0
        head_rank = 0
        tail_rank = 0

        for truth, prediction in self.parsed.triple_predictions.items():
            try:
                h_index = (
                    prediction[0].index(truth[0]) + 1
                )  # (first position has index 0)
                head_rank += h_index
            except ValueError:
                logging.error(
                    f"ERROR: Failed to retrieve head predictions for (correct) head concept: {truth[0]} "
                    f"Triple: {truth}"
                )
                ignored_heads += 1
            try:
                t_index = (
                    prediction[1].index(truth[2]) + 1
                )  # (first position has index 0)
                tail_rank += t_index
            except ValueError:
                logging.error(
                    f"ERROR: Failed to retrieve tail predictions for (correct) tail concept: {truth[2]} "
                    f"Triple: {truth}"
                )
                ignored_tails += 1

        mean_head_rank = 0
        mean_tail_rank = 0
        total_tasks = self.parsed.total_prediction_tasks
        if total_tasks - ignored_heads > 0:
            mean_head_rank = head_rank / (total_tasks / 2 - ignored_heads)
        if total_tasks / 2 - ignored_tails > 0:
            mean_tail_rank = tail_rank / (total_tasks / 2 - ignored_tails)

        logging.info(
            f"Mean Head Rank: {mean_head_rank} ({ignored_heads} ignored lines)"
        )
        logging.info(
            f"Mean Tail Rank: {mean_tail_rank} ({ignored_tails} ignored lines)"
        )

        mean_rank = 0
        if (total_tasks - ignored_tails - ignored_heads) > 0:
            mean_rank = (head_rank + tail_rank) / (
                total_tasks - ignored_tails - ignored_heads
            )
        mean_rank_rounded = round(mean_rank)
        logging.info(f"Mean rank: {mean_rank}; rounded: {mean_rank_rounded}")
        return round(mean_head_rank), round(mean_tail_rank), mean_rank_rounded

    def calculate_hits_at(self, n: int = 10) -> Tuple[int, int, int]:
        """Calculation of hits@n.

        Parameters
        ----------
        n : int
            Hits@n. This parameter specifies the n.

        Returns
        -------
        Tuple[int, int, int]
            [0] Hits at n only for heads.
            [1] Hits at n only for tails.
            [2] The hits at n. Note that head hits and tail hits are added.
        """
        heads_hits = 0
        tails_hits = 0

        for truth, prediction in self.parsed.triple_predictions.items():
            # perform the actual evaluation
            if truth[0] in prediction[0][: (n + 1)]:
                heads_hits += 1
            if truth[2] in prediction[1][: (n + 1)]:
                tails_hits += 1

        result = heads_hits + tails_hits
        logging.info(f"Hits@{n} Heads: {heads_hits}")
        logging.info(f"Hits@{n} Tails: {tails_hits}")
        logging.info(f"Hits@{n} Total: {result}")
        return heads_hits, tails_hits, result

    @staticmethod
    def write_results_to_file(
        file_to_be_evaluated: str,
        data_set: DataSet,
        file_to_be_written: str = "./results.txt",
    ) -> None:
        """Executes a filtered and non-filtered evaluation and prints the results to the console and to a file.

        Parameters
        ----------
        file_to_be_evaluated : str
            File path to the file that shall be evaluated.
        data_set : DataSet
            The data set that is under evaluation.
        file_to_be_written : str
            File path to the file that shall be written.
        """
        evaluator = Evaluator(
            file_to_be_evaluated=file_to_be_evaluated,
            is_apply_filtering=False,
        )
        hits_at_10 = evaluator.calculate_hits_at(10)
        test_set_size = len(data_set.test_set())
        mr = evaluator.mean_rank()

        non_filtered_text = (
            f"\nThis is the evaluation of file {file_to_be_evaluated}\n\n"
            + "Non-filtered Results\n"
            + "--------------------\n"
            + f"Test set size: {test_set_size}\n"
            + f"Hits at 10 (Heads): {hits_at_10[0]}\n"
            + f"Hits at 10 (Tails): {hits_at_10[1]}\n"
            + f"Hits at 10 (All): {hits_at_10[2]}\n"
            + f"Relative Hits at 10: {hits_at_10[2] / (2 * test_set_size)}\n"
            + f"Mean rank (Heads): {mr[0]}\n"
            + f"Mean rank (Tails): {mr[1]}\n"
            + f"Mean rank (All): {mr[2]}\n"
        )

        evaluator = Evaluator(
            file_to_be_evaluated=file_to_be_evaluated,
            is_apply_filtering=True,
        )
        hits_at_10 = evaluator.calculate_hits_at(10)
        test_set_size = len(data_set.test_set())
        mr = evaluator.mean_rank()

        filtered_text = (
            "\nFiltered Results\n"
            + "----------------\n"
            + f"Test set size: {test_set_size}\n"
            + f"Hits at 10 (Heads): {hits_at_10[0]}\n"
            + f"Hits at 10 (Tails): {hits_at_10[1]}\n"
            + f"Hits at 10 (All): {hits_at_10[2]}\n"
            + f"Relative Hits at 10: {hits_at_10[2] / (2 * test_set_size)}\n"
            + f"Mean rank (Heads): {mr[0]}\n"
            + f"Mean rank (Tails): {mr[1]}\n"
            + f"Mean rank (All): {mr[2]}\n"
        )

        with open(file_to_be_written, "w+", encoding="utf8") as f:
            f.write(non_filtered_text + "\n")
            f.write(filtered_text)

        print(non_filtered_text + "\n" + filtered_text)
