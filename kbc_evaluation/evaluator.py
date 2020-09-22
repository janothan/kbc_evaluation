import logging
from typing import List

# noinspection PyArgumentList
logging.basicConfig(handlers=[logging.FileHandler(__file__ + '.log', 'w', 'utf-8'), logging.StreamHandler()],
                    format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG)


class Evaluator:

    def __init__(self, file_to_be_evaluated: str, apply_filtering: bool = False):
        """Constructor

        Parameters
        ----------
        file_to_be_evaluated : str
            Path to the text file with the predicted links that shall be evaluated.
        """

        self.file_to_be_evaluated = file_to_be_evaluated
        self.apply_filtering = apply_filtering

    def mean_rank(self) -> float:
        total_tasks = 0
        ignored_heads = 0
        ignored_tails = 0
        head_rank = 0
        tail_rank = 0

        with open(self.file_to_be_evaluated, "r", encoding="utf8") as f:
            while True:
                # read three lines
                truth = f.readline()
                if not truth:
                    break
                heads = f.readline()
                if not heads:
                    break
                tails = f.readline()
                if not tails:
                    break

                # parse the lines
                truth, heads, tails = self._parse_lines(truth, heads, tails)
                total_tasks += 2

                try:
                    h_index = heads.index(truth[0]) + 1  # (first position has index 0)
                    head_rank += h_index
                except ValueError:
                    ignored_heads += 1
                try:
                    t_index = tails.index(truth[2]) + 1  # (first position has index 0)
                    tail_rank += t_index
                except ValueError:
                    ignored_tails += 1

        mean_head_rank = 0
        mean_tail_rank = 0
        if total_tasks - ignored_heads > 0:
            mean_head_rank = head_rank / (total_tasks/2 - ignored_heads)
        if total_tasks - ignored_tails > 0:
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
        with open(self.file_to_be_evaluated, "r", encoding="utf8") as f:
            while True:
                # read three lines
                truth = f.readline()
                if not truth:
                    break
                heads = f.readline()
                if not heads:
                    break
                tails = f.readline()
                if not tails:
                    break

                # parse the lines
                truth, heads, tails = self._parse_lines(truth, heads, tails)

                # perform the actual evaluation
                if truth[0] in heads[:(n + 1)]:
                    heads_hits += 1
                if truth[2] in tails[:(n + 1)]:
                    tails_hits += 1

        result = heads_hits + tails_hits
        logging.info(f"Hits@{n} Heads: {heads_hits}")
        logging.info(f"Hits@{n} Tails: {tails_hits}")
        logging.info(f"Hits@{n} Total: {result}")
        return result

    @staticmethod
    def _parse_lines(truth_line: str, heads_line: str, tails_line) -> (List, List, List):
        """Parses three lines from the evaluation file.

        Parameters
        ----------
        truth_line : str
            True line containing the correct triple.
        heads_line : str
            Line containing the heads.
        tails_line : str
            Line containing the tails.

        Returns
        -------
        (List, List, List)
            Tuple with element 0 being the parsed truth, element 1 being the parsed heads, and element 2 being the
            parsed tails.

        """
        # parse truth
        truth = truth_line.split(" ")
        if len(truth) != 3:
            logging.error(f"Problem evaluating the following triple: {truth}")
        else:
            truth[2] = truth[2].replace("\n", "")

        # parse heads
        heads = []
        heads_prefix = "\tHeads: "
        if not heads_line.startswith(heads_prefix):
            logging.error(f"Invalid heads line: {heads_line}")
        else:
            heads = heads_line[len(heads_prefix):]
            heads = heads.replace("\n", "")
            heads = heads.split(" ")

        # parse tails
        tails = []
        tails_prefix = "\tTails: "
        if not tails_line.startswith(tails_prefix):
            logging.error(f"Invalid tails line: {tails_line}")
        else:
            tails = tails_line[len(tails_prefix):]
            tails = tails.replace("\n", "")
            tails = tails.split(" ")

        return truth, heads, tails


if __name__ == "__main__":
    evaluator = Evaluator(file_to_be_evaluated="/Users/janportisch/PycharmProjects/KBC_RDF2Vec/wn_evaluation_file.txt")
    print(evaluator.calculate_hits_at(10))
