# https://support.prodi.gy/t/split-a-ner-manual-dataset-into-smaller-texts/5713/3
from prodigy.components.preprocess import split_sentences
import spacy
import srsly
import csv
import re
from spacy.util import compile_infix_regex

nlp = spacy.load("en_core_web_sm")


stream = srsly.read_jsonl("./annotated_data/ciat_ner_v2_combined_rev_20230106.jsonl")
stream = split_sentences(nlp, stream, min_length=30)

examples = ((eg["text"], eg) for eg in stream)

with open('./annotated_data/ciat_ner_v2_combined_rev_20230106_IOB_14_Jan.csv', 'w') as f:
    for doc, eg in nlp.pipe(examples, as_tuples=True):
        try:
            doc.ents = [doc.char_span(s["start"], s["end"], s["label"]) for s in eg["spans"]]
            iob_tags = [f"{t.ent_iob_}-{t.ent_type_}" if t.ent_iob_ else "O" for t in doc if not t.text.isspace()]
            iob_tags = [t.strip("-") for t in iob_tags]
            tokens = [str(t) for t in doc if not t.text.isspace()]

            writer = csv.writer(f, delimiter='\t')
            writer.writerows(zip(tokens, iob_tags))
            f.write("\n")

        except:
            pass
