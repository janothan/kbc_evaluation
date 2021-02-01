import logging
import os

from kbc_evaluation.dataset import DataSet, ParsedSet


logconf_file = os.path.join(os.path.dirname(__file__), "log.conf")
logging.config.fileConfig(fname=logconf_file, disable_existing_loggers=False)
logger = logging.getLogger(__name__)

# making sure that the relative path works
package_directory = os.path.dirname(os.path.abspath(__file__))


class Util:
    @staticmethod
    def write_sample_predictions(
        prediction_file: str,
        file_to_be_written: str,
        data_set: DataSet,
        is_apply_filtering: bool = True,
        top_predictions: int = 10,
        number_of_triples: int = 100,
    ) -> None:
        predictions_set = ParsedSet(
            file_to_be_evaluated=prediction_file, is_apply_filtering=is_apply_filtering
        )

        definitions_map = data_set.definitions_map()
        if definitions_map is None:
            return

        result_string = ""
        triples_processed = 0
        for triple, predictions in predictions_set.triple_predictions.items():
            result_string += f"Triple: {triple[0]}  {triple[1]}  {triple[2]}\n"

            h = triple[0]
            if triple[0] in definitions_map:
                h = definitions_map[triple[0]][0]
            l = triple[1]
            if triple[1] in definitions_map:
                l = definitions_map[triple[1]][0]
            t = triple[2]
            if triple[2] in definitions_map:
                t = definitions_map[triple[2]][0]

            result_string += f"Triple translated: {h}  {l}  {t}\n"
            result_string += f"\tHead Predictions:\n"
            head_predictions = predictions[0][: (top_predictions + 1)]
            for head_prediction in head_predictions:
                if head_prediction in definitions_map:
                    result_string += f"\t\t[{head_prediction}] {definitions_map[head_prediction][0]}   ({definitions_map[head_prediction][1]})\n"
                else:
                    result_string += head_prediction + "  (no concept link found)\n"
            result_string += f"\tTail Predictions:\n"
            tail_predictions = predictions[1][: (top_predictions + 1)]
            for tail_prediction in tail_predictions:
                if tail_prediction in definitions_map:
                    result_string += f"\t\t[{tail_prediction}] {definitions_map[tail_prediction][0]}   ({definitions_map[tail_prediction][1]})\n"
                else:
                    result_string += tail_prediction + "  (no concept link found)\n"
            triples_processed += 1
            if triples_processed >= number_of_triples:
                # we are finished, let's write the file
                break

        with open(file_to_be_written, "w+", encoding="utf-8") as f:
            f.write(result_string)


def main():
    Util.write_sample_predictions(
        prediction_file="/Users/janportisch/TMP/wn_averaged_most_similar.txt",
        file_to_be_written="./textual_predictions.txt",
        data_set=DataSet.WN18,
        is_apply_filtering=True,
        top_predictions=10,
        number_of_triples=100,
    )


if __name__ == "__main__":
    main()