"""This script is used to generate synthetic data for NER experimentations
Synthetic data format
Sentence, [word to mark], [label to word]
First define a template, containing two sections, sentence template, and entity list, separated by ==========, stored in folder `template`
execute this script, it will generate data stored in folder `synthetic-data`
synthetic data is annotated with BIO schema
"""
import pdb
from random import choice, random 
import re
from difflib import SequenceMatcher
import argparse


def annotateSpan(template, label):
  # choose one text from texts
  # replace var{} by choosing one span from span list
  # append label to the span
  with open(template) as f:
    template = f.read().split("==========")
    template_text, span_list = template[0], template[1]
    template_text = [t.strip() for t in template_text.split('\n') if t]
    span_list = [s.strip() for s in span_list.split('\n') if s]
  text, span = choice(template_text), choice(span_list)
  text = re.sub('\{.*\}', span, text)
  span_n_label = (span, label)
  print(text, span_n_label)
  return text, span_n_label


def markBIO(annotated_text):
  text, span_n_label = annotated_text[0], annotated_text[1]
  span, type = span_n_label[0], span_n_label[1]
  
  span = span.strip()
  seq_match = SequenceMatcher(None, text, span, autojunk=False)
  match = seq_match.find_longest_match(0, len(text), 0, len(span))
  # Single match only
  start = match.a 
  end = start + match.size
  print("start: {}, end: {}".format(start, end))

  temp_str = text[start:end]
  temp_str_tokens = temp_str.split()

  word_dict = {}
  pointer = 0
  for word in text.split():
    if pointer < start:
      word_dict[word] = '0'
    elif pointer >= start and pointer < end: 
      if len(temp_str_tokens) > 1:
        word_dict[temp_str_tokens[0]] = 'B-' + type
        for w in temp_str_tokens[1:]:
          word_dict[w] = 'I-' + type 
      else:
        word_dict[temp_str] = 'B-' + type
    else:
      word_dict[word] = '0'
    pointer += (len(word) + 1)
  print(word_dict)
  return word_dict

# file_name = 'text_apps.tmpl'
# tag_type = 'APPLICATION'
# file_name = 'text_devices.tmpl'
# tag_type = 'DEVICE'


def createData(n, template_file, tag_type, output_file):
  save_path = 'synthetic-data/' + template_file.split('.')[0] + '.txt' if output_file is None else output_file
  with open(save_path, 'w') as f:
    for _ in range(n):
      tmpl_file_name = 'templates/' + file_name
      annotated_text = (annotateSpan(tmpl_file_name, tag_type))
      word_dict = markBIO(annotated_text)
      for w in word_dict.keys():
        f.writelines(w + ' ' + word_dict[w] + '\n')
      f.writelines('\n')



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--tmpl_name", default='./templates/text_apps.tmpl', type=str, required=True, help="The template file defining template to generate synthetic data")
  parser.add_argument("--tag_type", default='APPLICATION', type=str, required=True, help="Label to annotate synthetic data")
  parser.add_argument("--output_file", default=None, required=False, help="The output file name")
  args = parser.parse_args()

  file_name = args.tmpl_name
  tag_type = args.tag_type
  output_file = args.output_file

  ### n refers to how many samples you wanna generate
  createData(n=50, template_file=file_name, tag_type=tag_type, output_file=output_file)
