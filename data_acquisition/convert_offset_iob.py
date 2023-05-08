import json
import os
import spacy

# Edit paths here
file_path = "/home/leroy/Dev/Code/PDPAnnotationDataUtils/annotated_data"
in_file = "tests-mine.jsonl"
out_file = "test-mine-output-iob-tags.tsv"


# Function to convert a single text and its spans to IOB format
def convert_to_iob(text, spans):
    doc = nlp(text)
    iob_tags = []

    for token in doc:
        tag = "O"

        for span in spans:
            if token.idx == span["start"]:
                tag = f"B-{span['label']}"
                break
            elif token.idx > span["start"] and token.idx + len(token.text) <= span["end"]:
                tag = f"I-{span['label']}"
                break

        iob_tags.append((token.text, tag))

    return iob_tags

# Function to convert a list of texts and their spans to IOB format
def adjust_annotation_spans(new_sentence_start, annotations):
    adjusted_annotations = []
    for annotation in annotations:
        start = annotation["start"] - new_sentence_start
        end = annotation["end"] - new_sentence_start
        adjusted_annotations.append({"start": start, "end": end, "label": annotation["label"]})
    return adjusted_annotations


#Function to split a list of texts and their spans into sentences
def split_sentences(nlp, prodigy_annotations, min_length):
    new_annotations = []
    for annotation in prodigy_annotations:
        text = annotation["text"]
        spans = annotation["spans"]
        doc = nlp(text)
        start = 0
        for sent in doc.sents:
            if sent.end - start >= min_length:
                new_text = doc[start:sent.end].text
                new_spans = adjust_annotation_spans(doc[start].idx, spans)
                new_annotations.append({"text": new_text, "spans": new_spans})
                start = sent.end
        if start < len(doc):
            new_text = doc[start:].text
            new_spans = adjust_annotation_spans(doc[start].idx, spans)
            new_annotations.append({"text": new_text, "spans": new_spans})
    return new_annotations


# Define the input file path
file_input_path = os.path.join(file_path, in_file)
# Define the output file path
file_output_path = os.path.join(file_path, out_file)

# Load the English tokenizer from spaCy
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "textcat"])

# Load the JSONL annotations file
with open(file_input_path, "r") as f:
    annotations = [json.loads(line) for line in f]

# Process the annotations by splitting the sentences with the specified minimum length
new_annotations = split_sentences(nlp, annotations, min_length=30)

# Open the output file
with open(file_output_path, "w") as output_file:
    # Iterate through all the annotations and convert them to IOB format
    for annotation in new_annotations:
        text = annotation["text"]
        spans = annotation["spans"]
        iob_tags = convert_to_iob(text, spans)

        # Write the IOB tags to the output file
        for token, tag in iob_tags:
            output_file.write(f"{token}\t{tag}\n")
        output_file.write("\n")




# Load the JSONL annotations file
file_path = "annotations.jsonl"
with open(file_path, "r") as f:
    prodigy_annotations = [json.loads(line) for line in f]


