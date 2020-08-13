import pdb 
import pandas as pd 



with open('dummy-data/test_tmpl.txt') as f:
  data = f.read()
  data = data.split('\n')
  data = [eval(t.replace("],", ']').replace('"', '')) for t in data_ttt]

df = pd.DataFrame(data, columns=['text', 'annotation'])


pdb.set_trace()
print()