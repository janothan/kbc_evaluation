import logging
from enum import Enum
import io
from typing import List

# noinspection PyArgumentList
logging.basicConfig(handlers=[logging.FileHandler(__file__ + '.log', 'w', 'utf-8'), logging.StreamHandler()],
                    format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG)


class DataSet(Enum):
    """The datasets that are available for evaluation. If a new enum value shall be added, a triple is to be stated with
    [0]: relative test set path
    [1]: relative train set path
    [2]: relative validation set path
    """

    FB15K = "./kbc_evaluation/datasets/fb15k/freebase_mtr100_mte100-test.txt", \
            "./kbc_evaluation/datasets/fb15k/freebase_mtr100_mte100-train.txt", \
            "./kbc_evaluation/datasets/fb15k/freebase_mtr100_mte100-valid.txt"
    WN18 = "./kbc_evaluation/datasets/wn18/wordnet-mlj12-test.txt", \
           "./kbc_evaluation/datasets/wn18/wordnet-mlj12-train.txt", \
           "./kbc_evaluation/datasets/wn18/wordnet-mlj12-valid.txt"

    def test_set(self) -> List[List[str]]:
        """Get the parsed test dataset.

        Returns
        -------
        List[List[str]]
            A list of parsed triples.
        """
        return self._parse_tab_separated_data(self.test_set_path())

    def train_set(self):
        """Get the parsed training dataset.

        Returns
        -------
        List[List[str]]
            A list of parsed triples.
        """
        return self._parse_tab_separated_data(self.train_set_path())

    def valid_set(self) -> List[List[str]]:
        """Get the parsed validation dataset.

        Returns
        -------
        List[List[str]
            A list of parsed triples.
        """
        return self._parse_tab_separated_data(self.valid_set_path())

    @staticmethod
    def _parse_tab_separated_data(file_path) -> List[List[str]]:
        """Parses the given file.

        Parameters
        ----------
        file_path : str
            Path to the file that shall be read. Expected format: token \tab token \tab token

        Returns
        -------
        List[List[str]]
            List of triples.
        """
        result = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.replace("\n", "")
                tokens = line.split(sep="\t")
                result.append(tokens)
        return result

    def test_set_path(self) -> str:
        """Get the test dataset path of the given dataset.

        Returns
        -------
        str
            Relative path to the test dataset.
        """
        return self.value[0]

    def train_set_path(self) -> str:
        """Get the training dataset path of the given dataset.

        Returns
        -------
        str
            Relative path to the training dataset.
        """
        return self.value[1]

    def valid_set_path(self) -> str:
        """Get the validation dataset path of the given dataset.

        Returns
        -------
        str
            Relative path to the validation dataset.
        """
        return self.value[2]

    @staticmethod
    def write_training_file_nt(data_set, file_to_write: str) -> None:
        """ File in NT format that can be parsed by jRDF2Vec (https://github.com/dwslab/jRDF2Vec).

        Parameters
        ----------
        data_set : DataSet
            The dataset that shall be persisted as NT file for vector training.
        file_to_write : str
            The file that shall be written.
        """

        with io.open(file_to_write, 'w+', encoding='utf8') as f:
            data_to_write = data_set.train_set()
            data_to_write.extend(data_set.valid_set())
            for triple in data_to_write:
                f.write("<" + triple[0] + "> <" + triple[1] + "> <" + triple[2] + "> .\n")


class ParsedSet:

    def __init__(self, file_to_be_evaluated: str, is_apply_filtering: bool = False):
        self.file_to_be_evaluated = file_to_be_evaluated
        self.is_apply_filtering = is_apply_filtering
        self.total_prediction_tasks = 0
        self.triple_predictions = {}

        # initialize lookup datastructures for filtering (contains only true statements)
        self._sp_map = {}
        self._po_map = {}

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
                self.triple_predictions[(truth[0], truth[1], truth[2])] = (heads, tails)
                self.total_prediction_tasks += 2

                if self.is_apply_filtering:
                    self._apply_filtering()

    def _apply_filtering(self) -> None:
        """Loops over self.triple_predictions and applies the filtering.
        """
        new_triple_predictions = {}
        for truth, prediction in self.triple_predictions.items():

            # processing heads
            heads = prediction[0]
            po_key = truth[1] + "_" + truth[2]
            correct_heads = self._po_map[po_key]
            new_heads = []
            for predicted_head in heads:
                if predicted_head == truth[0]:
                    new_heads.append(predicted_head)
                    continue
                if predicted_head not in correct_heads:
                    new_heads.append(predicted_head)

            # processing tails
            tails = prediction[1]
            sp_key = truth[0] + "_" + truth[1]
            correct_tails = self._sp_map[sp_key]
            new_tails = []
            for predicted_tail in tails:
                if predicted_tail == truth[2]:
                    new_tails.append(predicted_tail)
                    continue
                if predicted_tail not in correct_tails:
                    new_tails.append(predicted_tail)

            # replace with new predictions
            new_triple_predictions[truth] = (new_heads, new_tails)
        self.triple_predictions = new_triple_predictions

    def _parse_lines(self, truth_line: str, heads_line: str, tails_line) -> (List, List, List):
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

        # add to filter indices
        if self.is_apply_filtering:
            self._add_triple_to_filter_set(triple=truth)

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

    def _add_triple_to_filter_set(self, triple: List) -> None:
        """Adds the triple to the self._sp_map and self._po_map in order to apply the filtering later.

        Parameters
        ----------
        triple : List
            The triple to be added.
        """
        sp_key = triple[0] + "_" + triple[1]
        po_key = triple[1] + "_" + triple[2]
        if sp_key not in self._sp_map:
            # create new list with object
            self._sp_map[sp_key] = [triple[2]]
        else:
            # add to existing list
            self._sp_map[sp_key].append(triple[2])
        if po_key not in self._po_map:
            # create new list with subject
            self._po_map[po_key] = [triple[0]]
        else:
            self._po_map[po_key].append(triple[0])
