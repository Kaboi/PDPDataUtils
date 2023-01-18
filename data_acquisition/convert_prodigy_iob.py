# convert from prodigy format to IOB format using spacy biluo_tags_from_offsets
# %% imports
import spacy
from prodigy.components.db import connect
from prodigy.components.preprocess import split_sentences
import os
import re

# %% parameters
# db_name = "tags_output"
db_name = "ciat_ner_v2_combined_rev"

# %% load data and model for tokenization
print("loading data and model...")
nlp = spacy.load("en_core_web_lg")
db = connect()
prodigy_annotations = db.get_dataset(db_name)
prodigy_split_annotations = split_sentences(nlp, prodigy_annotations, min_length=30)
# annotations = ((ann["text"], ann) for ann in prodigy_annotations)
annotations = ((ann["text"], ann) for ann in prodigy_split_annotations)

# %% convert to IOB format and append to list
print("converting to IOB format...")
lines = []
for doc, eg in nlp.pipe(annotations, as_tuples=True):
    doc.ents = [doc.char_span(s["start"], s["end"], s["label"]) for s in eg["spans"]]
    # load the tokens and tags and strip the - from O tags
    iob_tags = [f"{t.ent_iob_}-{t.ent_type_}" if t.ent_iob_ and t.ent_type_ else f"{t.ent_iob_}" if t.ent_iob_ else "O"
                for t in doc]
    # print(doc.text, iob_tags)
    for token, tag in zip(doc, iob_tags):
        lines.append(f"{token.text}\t{tag}\n")
        # token_text = re.sub("\n+", "\n", token.text)
        # lines.append(f"{token_text}\t{tag}\n")
        # lines.append(f"{token.text}\t{tag}\n")
    lines.append(f"\n")

# %% save to file
print("saving to file...")
filename = "data/" + db_name + ".tsv"
output_file = os.path.join(os.getcwd(), filename)
with open(output_file, "w") as f:
    f.write("".join(lines))
