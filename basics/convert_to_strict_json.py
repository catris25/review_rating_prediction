import json
import gzip

# file_path = '/home/lia/Documents/the_project/dataset/Digital_Music_5.json.gz'
# file_path = '/home/lia/Documents/the_project/dataset/Movies_and_TV_5.json.gz'
file_path = 'test.json.gz'

def parse(path):
  g = gzip.open(path, 'r')
  for l in g:
    yield json.dumps(eval(l))

f = open("output.strict", 'w')
for l in parse(file_path):
  f.write(l + '\n')
