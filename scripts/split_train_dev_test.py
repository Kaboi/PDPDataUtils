# -*- coding: utf-8 -*-
"""
# **Import Python Libraries**
"""

import pandas as pd
import numpy as np
import os
import csv
import re

from sklearn.model_selection import train_test_split
RANDOM = 501501

"""# **Read NER Dataset**"""


def get_sentences(filename):
    f = open(filename)
    sentences = []
    sentence = []
    for line in f:
        if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
            if len(sentence) > 0:
                sentences.append(sentence)
                sentence = []
            continue
        splits = line.split('\t')
        sentence.append([splits[0].strip(), splits[-1].strip()])
        # if splits[0].strip() not in ['"'] and splits[0].strip() not in ["'"] and splits[-1].strip() in ['O','B-GPE','B-CROP','B-ORG','I-ORG','B-DATE','I-DATE','B-PLANT_PART','I-CROP','B-LOC','I-LOC','B-PATHOGEN','I-PATHOGEN','I-GPE','B-DISEASE','I-DISEASE','I-PLANT_PART']:
        # sentence.append([splits[0].strip(),splits[-1].strip()])

    print(len(sentences))

    sentences_list = []
    for i, sent in enumerate(sentences):
        for x in sent:
            x.append(str(i))  # open to add sentence idx
            sentences_list.append(x)
    return sentences, sentences_list
    # return sentences


# sent_text, all_text = get_sentences('ciat_ner_teach-leroy_Dec_27_IOB.csv')
sent_text, all_text = get_sentences('./annotated_data/ciat_ner_v2_combined_rev_20230106_IOB_14_Jan.csv')

# split data
print('Splitting data')
train_docs, test_docs = \
    train_test_split(sent_text,
                     test_size=0.2, random_state=RANDOM,
                     shuffle=True)

test_docs, val_docs = \
    train_test_split(test_docs, test_size=0.5,
                     random_state=RANDOM,
                     shuffle=True)

print('All {} docs, Training {} docs, Testing {} docs, Development {} docs'.format(len(sent_text),
                                                                                   len(train_docs), len(test_docs), len(val_docs)))

# # test_sents[1]
# !rm -rf /content/split_data_outdir

# write data splits

save_path = 'split_data_outdir'

if not os.path.exists(save_path):
    os.makedirs(save_path)

for element in [(train_docs, 'train'), (test_docs, 'test'), (val_docs, 'dev')]:
    docs, docs_type = element

    with open(os.path.join(save_path, docs_type + '.txt'), 'w') as f_out:

        for doc in docs:
            writer = csv.writer(f_out, delimiter=' ')
            writer.writerows(doc)
            f_out.write("\n")
