# 20200903
# FLAIR NER INFER
from __future__ import absolute_import, division, print_function
# evaluate using trained model to see if my modification is correct or not
from flair.data import Sentence
from flair.models import SequenceTagger
import argparse

from flair.data import Corpus
from flair.datasets import CONLL_03


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--data_dir",
      default='./',
      type=str,
      help=
      "The parent dir of input data, should include folder named `conll_03` ")
  parser.add_argument("--model_dir",
                      default=None,
                      type=str,
                      required=True,
                      help="The model directory where model chekpoints stored")
  parser.add_argument(
      "--result_file",
      default='dev.tsv',
      type=str,
      required=True,
      help=
      "The name of prediction file, default is in the same dir of script file")
  parser.add_argument("--eval_on",
                      default='dev',
                      type=str,
                      required=True,
                      help="Whether to eval on dev set or test set")

  args = parser.parse_args()

  model_path = args.model_dir
  model = SequenceTagger.load(model_path + '/final-model.pt')

  corpus: Corpus = CONLL_03(base_path=args.data_dir)
  if args.eval_on == 'dev':
    testdata = corpus.dev
  elif args.eval_on == 'test':
    testdata = corpus.test
  else:
    raise ValueError("Invalid argument, must specify evaluation on dev or test")

  test_result, test_loss = model.evaluate(testdata, out_path=args.result_file)
  result_line = f"\t{test_loss}\t{test_result.log_line}"

  # main score is micro averaged f1 score
  # result line is precision, recall, micro averaged score

  print(f"TEST : loss {test_loss} - score {round(test_result.main_score, 4)}")
  print(f"TEST RESULT: {result_line}")
  print(test_result.detailed_results)


if __name__ == '__main__':
  main()