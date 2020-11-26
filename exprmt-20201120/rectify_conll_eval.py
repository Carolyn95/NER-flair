# FLAIR NER INFER
from __future__ import absolute_import, division, print_function
# evaluate using trained model to see if my modification is correct or not
from flair.data import Sentence
from flair.models import SequenceTagger
import argparse

from flair.data import Corpus
from flair.datasets import CONLL_03


def main(data_dir, model_path, result_file):
  # parser = argparse.ArgumentParser()
  # parser.add_argument("--data_dir", default='./', type=str, help="The parent dir of input data, should include folder named `conll_03` ")
  # parser.add_argument("--model_dir", default=None, type=str, required=True, help="The model directory where model chekpoints stored")
  # parser.add_argument("--result_file", default='dev.tsv', type=str, required=True, help="The name of prediction file, default save in current dir")
  # parser.add_argument("--eval_on", default='dev', type=str, required=True, help="Whether to eval on dev set or test set")

  # args = parser.parse_args()

  # model_path = args.model_dir
  model = SequenceTagger.load(model_path + '/final-model.pt')

  # corpus: Corpus = CONLL_03(base_path=args.data_dir)
  corpus: Corpus = CONLL_03(base_path=data_dir)
  testdata = corpus.dev

  # test_result, test_loss = model.evaluate(testdata, out_path=args.result_file)
  test_result, test_loss = model.evaluate(testdata, out_path=result_file)
  result_line = f"\t{test_loss}\t{test_result.log_line}"

  # main score is micro averaged f1 score
  # result line is precision, recall, micro averaged score

  print(f"TEST : loss {test_loss} - score {round(test_result.main_score, 4)}")
  print(f"TEST RESULT: {result_line}")
  print(test_result.detailed_results)


if __name__ == '__main__':
  parent_dir = 'exprmt-20201120/conll_frac/'
  # data_dir = ['10ptdata', '1ptdata', '3ptdata', '5ptdata', 'pt1ptdata', 'pt3ptdata', 'pt5ptdata']
  # data_dir = ['100ptdata']
  data_dir = ['10ptdata']
  data_dir = [parent_dir + d for d in data_dir]
  # models-20201124: 10e
  model_path = [directory + '/models-5e-20201124' for directory in data_dir]
  result_file = [md + '/dev_20201125.tsv' for md in model_path]
  for dd, mp, rf in zip(data_dir, model_path, result_file):
    # print(dd, mp, rf)
    main(dd, mp, rf)
    # python rectify_conll_eval.py --data_dir=conll_frac/1ptdata --model_dir=conll_frac/1ptdata/models-5e-20201124 --result_file=conll_frac/1ptdata/models-5e-20201124/dev.tsv --eval_on=dev
