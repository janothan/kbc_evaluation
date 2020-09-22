import logging


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

    def calculate_hits_at(self, n: int = 10) -> int:
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
                # parse truth
                truth = truth.split(" ")
                if len(truth) != 3:
                    logging.error(f"Problem evaluating the following triple: {truth}")
                else:
                    truth[2] = truth[2].replace("\n", "")

                # parse heads
                heads_prefix = "\tHeads: "
                if not heads.startswith(heads_prefix):
                    logging.error(f"Invalid heads line: {heads}")
                else:
                    heads = heads[len(heads_prefix):]
                    heads = heads.replace("\n", "")
                    heads = heads.split(" ")

                # parse tails
                tails_prefix = "\tTails: "
                if not tails.startswith(tails_prefix):
                    logging.error(f"Invalid tails line: {tails}")
                else:
                    tails = tails[len(tails_prefix):]
                    tails = tails.replace("\n", "")
                    tails = tails.split(" ")

                # perform the actual evaluation
                if truth[0] in heads[:(n+1)]:
                    heads_hits += 1
                if truth[2] in tails[:(n+1)]:
                    tails_hits += 1

        result = heads_hits + tails_hits
        logging.info(f"Hits@{n} Heads: {heads_hits}")
        logging.info(f"Hits@{n} Tails: {tails_hits}")
        logging.info(f"Hits@{n} Total: {result}")
        return result


if __name__ == "__main__":
    evaluator = Evaluator(file_to_be_evaluated="/Users/janportisch/PycharmProjects/KBC_RDF2Vec/wn_evaluation_file.txt")
    print(evaluator.calculate_hits_at(10))
