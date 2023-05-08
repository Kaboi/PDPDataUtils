# convert from prodigy format to IOB format using spacy biluo_tags_from_offsets
# %% imports
import spacy
# from prodigy.components.db import connect
from prodigy.components.loaders import JSONL
from prodigy.components.preprocess import split_sentences, add_tokens
import os
import re

# %% parameters
db_name = "test-mine-split-new"
# db_name = "ciat_ner_v2_combined_rev"

# %% load data and model for tokenization
print("loading data and model...")
nlp = spacy.load("en_core_web_lg")
# load data from prodigy database
# db = connect()
# prodigy_annotations = db.get_dataset(db_name)

# load data from jsonl file
# prodigy_annotations = JSONL(os.path.join("annotated_data", "ciat_ner_v2_combined_rev_20230106.jsonl"))
prodigy_annotations = JSONL(os.path.join("/home/leroy/Dev/Code/PDPAnnotationDataUtils/annotated_data"
                                         # "/ciat_ner_v2_combined_rev_20230106.jsonl"))
                                         "/tests-mine.jsonl"))


prodigy_split_annotations = split_sentences(nlp, prodigy_annotations, min_length=30)
prodigy_split_annotations = add_tokens(nlp, prodigy_split_annotations)
# annotations = ((ann["text"], ann) for ann in prodigy_annotations)
annotations = ((ann["text"], ann) for ann in prodigy_split_annotations)


# %% convert to IOB format and append to list
print("converting to IOB format...")
lines = []
for doc, eg in nlp.pipe(annotations, as_tuples=True):
    doc.ents = [doc.char_span(s["start"], s["end"], s["label"]) for s in eg["spans"]]
    # load the tokens and tags and strip the - from O tags
    iob_tags = [f"{d.ent_iob_}-{d.ent_type_}" if d.ent_iob_ and d.ent_type_ else f"{d.ent_iob_}" if d.ent_iob_ else "O"
                for d in doc]

    # iob_tags = [f"{t.ent_iob_}-{t.ent_type_}" if t.ent_iob_ else "O" for t in doc]
    # print(doc.text, iob_tags)
    for token, tag in zip(doc, iob_tags):
        lines.append(f"{token.text}\t{tag}\n")
        # token_text = re.sub("\n+", "\n", token.text)
        # lines.append(f"{token_text}\d{tag}\n")
        # lines.append(f"{token.text}\d{tag}\n")
    lines.append(f"\n")

# %% save to file
print("saving to file...")
filename = "data/" + db_name + ".tsv"
output_file = os.path.join(os.getcwd(), filename)
with open(output_file, "w") as f:
    f.write("".join(lines))
