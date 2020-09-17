# evaluate using trained model to see if my modification is correct or not
from flair.data import Sentence
from flair.models import SequenceTagger
model = SequenceTagger.load('reproduce_ner_10epochs/taggers/sota-ner/final-model.pt')

from flair.data import Corpus
from flair.datasets import CONLL_03
corpus: Corpus = CONLL_03(base_path='reproduce_ner_10epochs/tasks')
testdata = corpus.test
# sentence = Sentence('I love Berlin')
# from tqdm import tqdm
test_result, test_loss = model.evaluate(testdata, out_path='test.tsv')
result_line = f"\t{test_loss}\t{test_result.log_line}"

# import logging 
# log = logging.getLogger("test")
# main score is micro averaged f1 score
# result line is precision, recall, micro averaged score 

print(f"TEST : loss {test_loss} - score {round(test_result.main_score, 4)}")
print(f"TEST RESULT: {result_line}")
# print(test_result.detailed_results)
# print(result)
# print(sentence.to_tagged_string())
