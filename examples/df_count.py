texts = [['the', 'boy', 'peter'],['pete','the', 'boy'],['peter','rabbit', 'peter']]
def flatten(l):
    out = []
    for item in l:
        if isinstance(item, (list, tuple)):
            out.extend(flatten(item))
        else:
            out.append(item)
    return out

# flat = flatten(texts)
#
# print(sum(1 for text in texts if 'peter' in text))

import collections
def counts(listr, word):
    total = []
    for i in range(len(texts)):
        total.append(word in collections.Counter(listr[i]))
    return(sum(total))

print(counts(texts,'pete'))
# texts = ['fuck this fuck that fuck everything else', 'what the fuck', 'oh fuck you', 'yeah yeah yeah']
# matches = len([True for text in texts if 'fuck' in text])
#
# print(matches)
