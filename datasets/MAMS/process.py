# import xml.etree.ElementTree as ET
# tree = ET.parse('raw/test.xml')
# root = tree.getroot()
#
# dataset = []
#
# for child in root:
#     for node in child:
#         if node.tag == 'text':
#             sentence = node.text
#         else:
#             for cn in node:
#                 fro = int(cn.attrib['from'])
#                 to = int(cn.attrib['to'])
#                 polarity = cn.attrib['polarity']
#                 if polarity == 'neutral':
#                     polarity = '0'
#                 elif polarity == 'positive':
#                     polarity = '1'
#                 else:
#                     polarity = '0'
#                 dataset.append([sentence[:fro]+'$T$'+sentence[to:], sentence[fro:to], polarity])
#
# with open('test.txt', 'w') as file:
#     for data in dataset:
#         for d in data:
#             file.write(d+'\n')
#         # file.write('\n')


##############################################

import spacy, pickle
from tqdm import tqdm

nlp = spacy.load("en")

As = {}

with open('val.txt') as file:
    lines = file.readlines()
    i = -1
    for l in tqdm(lines):
        i += 1
        if i % 3 == 0:
            doc = nlp(l.strip())
            parse = []
            for j, token in enumerate(doc):
                head = token.head.i
                if j == head:
                    head = -1
                # head += 1
                parse.append(head)
            A = []
            for _ in range(len(doc)):
                A.append([0]*len(doc))
            for j, p in enumerate(parse):
                # p -= 1
                if p >= 0:
                    A[j][p] = 1
                    A[p][j] = 1
            As[i] = A

print(len(As))

pickle.dump(As, open('val.txt.graph', 'wb'))

###############################################################

# new_dataset = []
#
# with open('val.txt') as file:
#     lines = file.readlines()
#     for i, l in enumerate(lines):
#         if i % 4 == 2:
#             if l.strip() == 'neutral':
#                l = '0\n'
#             elif l.strip == 'positive':
#                 l = '1\n'
#             else:
#                 l = '-1\n'
#         new_dataset.append(l)
#
# with open('val.txt', 'w') as file:
#     for l in new_dataset:
#         file.write(l)

