# convert from prodigy format to IOB format using spacy biluo_tags_from_offsets
# %% imports
import spacy
from prodigy.components.db import connect
import os

# %% parameters
db_name = "ciat_ner_v2_combined_rev"

# %% load data and model for tokenization
print("loading data and model...")
db = connect()
prodigy_annotations = db.get_dataset(db_name)
annotations = ((ann["text"], ann) for ann in prodigy_annotations)
nlp = spacy.load("en_core_web_lg")

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

# %% save to file
print("saving to file...")
filename = "data/" + db_name + ".tsv"
output_file = os.path.join(os.getcwd(), filename)
with open(output_file, "w") as f:
    f.write("".join(lines))
