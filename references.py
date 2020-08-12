# ============= Fine-tune idea =============
# https://github.com/flairNLP/flair/issues/53
from flair.models import SequenceTagger
tagger = SequenceTagger.load("/home/carolyn/.flair/models/en-ner-conll03-v0.4.pt")
from flair.data import Corpus
from flair.datasets import ColumnCorpus
columns = {0: 'text', 1: 'ner'}                                                                                                                                    
data_folder = './dummy-data/'                                                                                       
corpus: Corpus = ColumnCorpus(data_folder, columns)                                                                     
tag_type = 'ner'                                                                                                                                                   
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)   
from flair.trainers import ModelTrainer                                                                                                                            
trainer: ModelTrainer = ModelTrainer(tagger, corpus)                     
trainer.train('dummy-model', train_with_dev=True, max_epochs=5)              
model = SequenceTagger.load('dummy-model/final-model.pt')                                                               

# ============= Generate data idea - birthday corpus - progidy =============
import json
import re
import time
from random import choice, random
from typing import TextIO, Callable, Sequence, Tuple, Optional

import click

NAME = DATE = str
SPAN_OFFSET = Tuple[int, int]


def generate(name_factory: Callable[[], NAME], lifespan_factory: Callable[[], Tuple[DATE, DATE]]) \
        -> Tuple[str, Optional[SPAN_OFFSET], Optional[SPAN_OFFSET]]:
    def find_span(date: DATE) -> SPAN_OFFSET:
        i = text.find(date)
        j = i + len(date)
        return i, j

    name = name_factory()
    born, died = lifespan_factory()
    texts = [
        (f"{name} was born on {born}.", True, False),
        (f"{name} has a birthday on {born}.", True, False),
        (f"{name} was born on {born} and died {died}.", True, True),
        (f"On {born} {name} was born.", True, False),
        (f"On {died} {name} died.", False, True),
        (f"{name} died on {died}.", False, True),
        (f"RIP {name}: {born}-{died}.", True, True),
        (f"A skilled carpenter, {name} lived from {born} until {died}.", True, True),
        (f"{died} was the day {name} died.", False, True),
        (f"{born} was the day {name} was born.", True, False),
        (f"{name} is a skilled juggler.", False, False),
        (f"Where are you, {name}?", False, False)
    ]
    text, contains_born, contains_died = choice(texts)
    born_span = died_span = None
    if contains_born:
        born_span = find_span(born)
    if contains_died:
        died_span = find_span(died)
    return text, born_span, died_span


def name_generator(first_names: Sequence[str], last_names: Sequence[str]) -> Callable[[], NAME]:
    def factory() -> str:
        if random() < 0.5:
            return f"{choice(first_names)} {choice(last_names)}"
        else:
            return f"{choice(first_names)}"

    return factory


def lifespan_generator(start="1/1/1900", end="12/31/2010") -> Callable[[], Tuple[DATE, DATE]]:
    start = time.mktime(time.strptime(start, "%m/%d/%Y"))
    end = time.mktime(time.strptime(end, "%m/%d/%Y"))
    formats = ["%m/%d/%Y", "%B %d, %Y", "%d %B %Y"]

    def factory() -> Tuple[DATE, DATE]:
        def make_date(timestamp):
            date = time.strftime(fmt, time.localtime(timestamp))
            return re.sub(r'\b0(\d)', r'\1', date)  # Remove leading zeroes from numbers.

        born = start + (end - start) * random()
        died = born + (end - born) * random()
        fmt = choice(formats)
        return make_date(born), make_date(died)

    return factory


@click.command()
@click.option("--n", default=10000, help="number of samples to generate")
@click.option("--first-names", type=click.File(), help="list of first names, one per line")
@click.option("--last-names", type=click.File(), help="list of last names, one per line")
def birthday_corpus(n: int, first_names: Optional[TextIO], last_names: Optional[TextIO]):
    """
    Generate a corpus of texts describing birth and death dates for people.

    The texts refer to dates on which a person was born and or died. The appropriate date spans are annoated with a
    BIRTHDAY label. This is used to create a training file that can be used by Prodigy.

    If the first names or last names file is not specified, a short default list of names is used.

    See https://prodi.gy.
    """

    def annotation_span(span, accept):
        return {"text": text[span[0]:span[1]], "start": span[0], "end": span[1], "label": "BIRTHDAY", "accept": accept}

    if first_names is not None:
        first_names_list = list(name.title().strip() for name in first_names)
    else:
        first_names_list = ["Mary", "Sue", "John", "Roger"]
    if last_names is not None:
        last_names_list = list(name.title().strip() for name in last_names)
    else:
        last_names_list = ["Smith", "Jones", "Jackson", "Ruiz"]
    for _ in range(n):
        text, born_span, died_span = generate(name_generator(first_names_list, last_names_list), lifespan_generator())
        spans = []
        if born_span:
            spans.append(annotation_span(born_span, True))
        if died_span:
            spans.append(annotation_span(died_span, False))
        click.echo(json.dumps({"text": text, "spans": spans}))


if __name__ == "__main__":
    birthday_corpus()

# ============= Generate data idea - BIO schema =============

""" Sample input format
George N B-PER
Washington N I-PER
went V O
to P O 
Washington N B-LOC

Sam N B-PER 
Houston N I-PER 
stayed V O 
home N O
"""
# Tag to predict
tag_type = 'ner'
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
# Use flair embeddings
from flair.embeddings import WordEmbeddings, StackedEmbeddings
from typing import List
embedding_types: List[TokenEmbeddings] = [
    WordEmbeddings('glove'),
]
embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)
# Initialize sequence tagger (bi-LSTM, CRF)
from flair.models import SequenceTagger
tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type,
                                        use_crf=True)
print(tagger)
# Train model
from flair.trainers import ModelTrainer
trainer: ModelTrainer = ModelTrainer(tagger, corpus)
trainer.train('resources/taggers/example-ner',
              learning_rate=0.1,
              mini_batch_size=32,
              max_epochs=150)

# Use the trained model to predict
from flair.data import Sentence
from flair.models import SequenceTagger
model = SequenceTagger.load('resources/tagger/example-ner/final-model.pt')
sentence = Sentence('I love Berlin')
model.predict(sentence)
print(sentence.to_taggerd_string())

# Sentence tokenizer and create data
import pandas as pd
from tqdm import tqdm
from difflib import SequenceMatcher
import re
import pickle as plk


def matchSequence(string, pattern):
  """Return start and end index of any pattern present in the text
  """
  match_list = []
  pattern = patter.strip()
  seq_match = SequenceMatcher(None, string, pattern, autojunk=False)
  match = seq_match.find_longest_match(0, len(string), 0, len(pattern))
  if (match.size == len(pattern)):
    start = match.a
    end = match.a + match.size
    match_tup = (start, end)
    string = string.replace(pattern, "X" * len(pattern), 1)
    match_list.append(match_tup)

  return match_list, string


def markSentence(s, match_list):
  """Marks all the entities in the sentence as per the BIO scheme
  """
  word_dict = {}
  for word in s.split():
    word_dict[word] = '0'

  for start, end, e_type in match_list:
    temp_str = s[start:end]
    tmp_list = temp_str.split()
    if len(tmp_list) > 1:
      word_dict[tmp_list[0]] = 'B-' + e_type
      for w in tmp_list[1:]:
        word_dict[w] = 'I-' + e_type
    else:
      word_dict[temp_str] = 'B-' + e_type
  return word_dict


def clean(text):
  """Helper function to add a space before the puntuations for better tokenization
  """
  filters = [
      '!', '#', '$', '%', '&', '(', ')', '/', '*', '.', ':', ';', '<', '=', '>',
      '?', '@', '[', '\\', ']', '_', '`', '{', '}', '~', "'"
  ]
  for i in text:
    if i in filters:
      text = text.replace(i, ' ' + i)
  return text


def createData(df, filepath):
  """Create data in the desired format
  """
  with open(filepath, 'w') as f:
    for text, annotation in zip(df.text, df.annotation):
      text = clean(text)
      text_ = text
      match_list = []
      for i in annotation:
        a, text_ = matchSequence(text, i[0])
        match_list.append((a[0][0], a[0][1], i[1]))

      d = markSentence(text, match_list)
      for i in d.keys():
        f.writelines(i + ' ' + d[i] + '\n')
      f.writelines('\n')


def main():
  """Show a smaill piece of example
  """
  data = pd.DataFrame(
      [[
          'Horses are too tall and they pretend to care about your feelings',
          [('Horses', 'ANIMAL')]
      ], ['Who is Shaka Khan?', [('Shaka Khan', 'PERSON')]],
       [
           'I like London and Berlin.',
           [('London', 'LOCATION'), ('Berlin', 'LOCATION')]
       ],
       ['There is a banyan tree in the courtyard', [('banyan tree', 'TREE')]]])

  filepath = 'train/train.txt'
  createData(data, filepath)


if __name__ == '__main__':
  main()

