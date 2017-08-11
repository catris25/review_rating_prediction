import csv
import pandas as pd
import gzip

file_path = '/home/lia/Documents/the_project/dataset/Movies_and_TV_5.json.gz'
# file_path ='test.json'

def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield eval(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

output_path = '/home/lia/Documents/the_project/dataset/Movies_and_TV_review.csv'
df = getDF(file_path)
df.to_csv(output_path, sep=',', encoding='utf-8')
# df.to_csv('test_music2.csv', sep=',', encoding='utf-8', quoting=csv.QUOTE_NONE, quotechar='',escapechar='')
# print(df.head(10))
