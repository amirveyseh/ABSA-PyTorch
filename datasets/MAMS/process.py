# import xml.etree.ElementTree as ET
# tree = ET.parse('raw/train.xml')
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
#                 dataset.append([sentence[:fro]+'$T$'+sentence[to:], sentence[fro:to], polarity])
#
# with open('train.txt', 'w') as file:
#     for data in dataset:
#         for d in data:
#             file.write(d+'\n')
#         file.write('\n')


##############################################

# import spacy, pickle
# from tqdm import tqdm
#
# nlp = spacy.load("en")
#
# As = {}
# 
# with open('val.txt') as file:
#     lines = file.readlines()
#     i = -1
#     for l in tqdm(lines):
#         i += 1
#         if i % 4 == 0:
#             doc = nlp(l.strip())
#             parse = []
#             for i, token in enumerate(doc):
#                 head = token.head.i
#                 if i == head:
#                     head = -1
#                 # head += 1
#                 parse.append(head)
#             A = []
#             for _ in range(len(doc)):
#                 A.append([0]*len(doc))
#             for i, p in enumerate(parse):
#                 # p -= 1
#                 if p >= 0:
#                     A[i][p] = 1
#                     A[p][i] = 1
#             As[i] = A
#
# pickle.dump(As, open('val.txt.graph', 'wb'))
