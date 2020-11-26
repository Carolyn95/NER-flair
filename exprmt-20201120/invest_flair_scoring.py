from typing import List
import pdb
import pandas as pd
import csv


def summariseDebugTxt(result_file):

  columns = ['text', 'gold', 'pred']
  result_df = pd.DataFrame(columns=columns, index=None)
  with open(result_file) as f:
    result_data = csv.reader(f, delimiter='\t')

    for i, line in enumerate(result_data):
      if line:
        # pdb.set_trace()
        row = line[0].split(' ')
        result_df.loc[i] = row
      else:
        pass
  result_df.reset_index(drop=True, inplace=True)

  length = len(result_df)
  for l in range(length):
    result_df.loc[l,
                  'gold'] = result_df.loc[l,
                                          'gold'].replace('B-',
                                                          '').replace('I-', '')
    result_df.loc[l,
                  'pred'] = result_df.loc[l,
                                          'pred'].replace('B-',
                                                          '').replace('I-', '')

  print(result_df.groupby('gold'))
  # tp, fp, fn = 0, 0, 0
  # result_df = result_df[result_df.gold != 'O']
  # result_df.reset_index(drop=True, inplace=True)

  # for i, gold_span in enumerate(result_df.gold):
  #   if gold_span != result_df.loc[i, 'pred']:
  #     fn += 1

  # for i, pred_span in enumerate(result_df.pred):
  #   if pred_span == result_df.loc[i, 'gold']:
  #     tp += 1
  #   else:
  #     fp += 1

  # print('tp is {} fp is {} fn is {}'.format(tp, fp, fn))
  # prec = tp / (tp + fp)
  # rec = tp / (tp + fn)
  # f1 = 2 * prec * rec / (prec + rec)

  # print('prec is {} rec is {} f1 is {}'.format(prec, rec, f1))


if __name__ == '__main__':
  summariseDebugTxt('conll_frac/10ptdata/models-5e-20201124/dev_20201125.tsv')
