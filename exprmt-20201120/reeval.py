import pandas as pd
import pdb
import csv
import numpy as np
from sklearn.metrics import confusion_matrix
import os
from sklearn import metrics


def removeBioFromResult(result_file):
  columns = ['text', 'gold', 'pred']
  result_df = pd.DataFrame(columns=columns, index=None)
  sent_breaker = []
  with open(result_file) as f:
    result_data = csv.reader(f, delimiter='\t')
    for i, line in enumerate(result_data):
      if line:
        row = line[0].split(' ')
        result_df.loc[i] = row
      else:
        sent_breaker.append(i)
        pass
  result_df.reset_index(drop=True, inplace=True)
  # np.save('sent_breaker.npy', sent_breaker)
  return result_df


def getOrgDF(result_file):
  columns = ['text', 'gold', 'pred']
  result_df = pd.DataFrame(columns=columns, index=None)
  sent_breaker = []
  with open(result_file) as f:
    result_data = csv.reader(f, delimiter='\t')

    for i, line in enumerate(result_data):
      if line:
        row = line[0].split(' ')
        result_df.loc[i] = row
      else:
        # replace empty line with whitespaces in three columns
        result_df.loc[i] = ' ', ' ', ' '
        pass
  result_df.reset_index(drop=True, inplace=True)
  # np.save('sent_breaker.npy', sent_breaker)
  return result_df


def findSentBoundary(result_df, labels):
  if not labels:
    labels = sorted(result_df.gold.unique())
  org_df = getOrgDF('../2pt5pct/data/test.tsv')
  miscls_df = org_df.loc[org_df.gold != org_df.pred]
  full_list = org_df.text.tolist()

  sent_breaker = np.load('sent_breaker.npy', allow_pickle=True).tolist()

  for idx in miscls_df.index:
    for i, real_i in enumerate(sent_breaker):
      if i == 0 and idx < real_i:
        sent_start = 0
        sent_end = sent_breaker[0]
        print(idx, sent_start, sent_end)
        sent = ' '.join(full_list[sent_start:sent_end + 1])
        miscls_df.loc[idx, 'sent'] = sent
        continue
      else:
        if real_i <= idx and idx < sent_breaker[i + 1]:
          sent_start = real_i + 1
          sent_end = sent_breaker[i + 1]
          print(idx, sent_start, sent_end)
          sent = ' '.join(full_list[sent_start:sent_end + 1])
          miscls_df.loc[idx, 'sent'] = sent
          continue

  return miscls_df


def sklearnEval(result_df, is_bio, break_down_path=None):

  if not is_bio:
    size = len(result_df)
    for s in range(size):
      result_df.loc[s, 'gold'] = result_df.loc[s, 'gold'].replace('B-',
                                                                  '').replace(
                                                                      'I-', '')
      result_df.loc[s, 'pred'] = result_df.loc[s, 'pred'].replace('B-',
                                                                  '').replace(
                                                                      'I-', '')
      # result_df.to_csv('no_bio_result.csv')
      file_prefix = 'no_bio_'
  else:
    # result_df.to_csv('bio_result.csv')
    file_prefix = 'bio_'

  print('True labels: {}\n'.format(result_df.groupby('gold').size()))
  print('Predicted labels: {}\n'.format(result_df.groupby('pred').size()))

  # make "classification report"
  target_names = sorted(result_df.gold.unique())
  print('Labels: {}\n'.format(target_names))
  # target_names = [t for t in target_names if t != 'O']

  y_true = result_df.gold
  y_pred = result_df.pred
  classification_report = metrics.classification_report(
      y_true, y_pred, digits=4, target_names=target_names, zero_division=1)
  getBreakdowns(result_df, target_names, file_prefix, break_down_path)
  # miscls_df = findSentBoundary(result_df, target_names)
  # miscls_df.to_csv('miscls_df.csv')
  # precision    recall  f1-score       tps       fps       fns
  conf_mat = confusion_matrix(y_true, y_pred, labels=target_names)
  conf_mat_df = pd.DataFrame(conf_mat, index=target_names, columns=target_names)
  print(conf_mat_df)
  # print('TN: {}\n, FP: {}\n, FN: {}\n, TP: {}\n'.format(tn, fp, fn, tp))
  # get_breakdown_mat(conf_mat)

  beta = 1.0

  # get scores
  macro_f_score = round(
      metrics.fbeta_score(y_true, y_pred, beta=beta, average='macro'), 4)
  micro_f_score = round(
      metrics.fbeta_score(y_true, y_pred, beta=beta, average='micro'), 4)
  weighted_avg_f_score = round(
      metrics.fbeta_score(y_true, y_pred, beta=beta, average='weighted'), 4)
  accuracy_score = round(metrics.accuracy_score(y_true, y_pred), 4)

  detailed_result = ("\nResults:"
                     f"\n- F-score (macro) {macro_f_score}"
                     f"\n- F-score (micro) {micro_f_score}"
                     f"\n- F1-score (weighted) {weighted_avg_f_score}"
                     f"\n- Accuracy {accuracy_score}"
                     '\n\nBy class:\n' + classification_report)
  print(detailed_result)


def getBreakdowns(result_df, labels, file_prefix, save_folder=None):
  if save_folder and not os.path.exists(save_folder):
    os.makedirs(save_folder)

  if not labels:
    labels = sorted(y_true.unique())

  for label in labels:
    target_label = label
    other_labels = [l for l in labels if l != target_label]
    for mis_label in other_labels:
      sub_df = result_df.loc[(result_df.gold == target_label) &
                             (result_df.pred == mis_label)]
      sub_df.reset_index(drop=True, inplace=True)
      # sub_df.to_csv(save_folder + file_prefix + label + '_' + mis_label + '.csv')


def removeOCls(df):
  df = df.loc[(df.gold != 'O') & (df.pred != 'O')]
  df.reset_index(drop=True, inplace=True)
  return df


if __name__ == '__main__':
  # FlairEmbModels / WordEmbModels / TransEmbModels / WordEmbTransEmbModels / GloveFlairFwBwModels
  # result_df = removeBioFromResult('2pt5pct/GloveFlairFwBwModels/test.tsv')
  result_df = removeBioFromResult(
      'conll_frac/100ptdata/models-5e-20201124/test.tsv')
  # result_df = removeOCls(result_df)
  # is_bio=True
  sklearnEval(result_df, is_bio=True, break_down_path=None)
