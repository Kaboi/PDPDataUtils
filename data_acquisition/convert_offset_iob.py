import json
import os
import spacy
from transformers import BertTokenizerFast, RobertaTokenizerFast, AutoTokenizer

# Edit paths here
transformer_model = "bert-base-uncased"  # Choose from "bert-base-uncased", "roberta-base", "allenai/scibert_scivocab_uncased"

file_path = "/home/leroy/Dev/Code/PDPAnnotationDataUtils/annotated_data"
in_file = "tests-mine.jsonl"
# in_file = "ciat_ner_v2_combined_rev_20230106.jsonl"
# out_file = "test-mine-output-iob-tags.tsv"
# out_file = "ciat_ner_v2_combined_rev_20230106-iob-tags" + transformer_model + ".tsv"
out_file = "tests-mine-output-iob-tags" + transformer_model + ".tsv"


# Function to convert a single text and its spans to IOB format
def convert_to_iob_spacy(lnlp, txt, spns):
    doc = lnlp(txt)
    iob_tgs = []

    for tkn in doc:
        tg = "O"

        for span in spns:
            if tkn.idx == span["start"]:
                tg = f"B-{span['label']}"
                break
            elif tkn.idx > span["start"] and tkn.idx + len(tkn.text) <= span["end"]:
                tg = f"I-{span['label']}"
                break

        iob_tgs.append((tkn.text, tg))

    return iob_tgs


# Function to convert a single text and its spans to IOB format using Hugging Face's tokenizer
def convert_to_iob_huggingface(tokenizer, txt, spns):
    tokens = tokenizer.tokenize(txt)
    token_offsets = tokenizer(txt, return_offsets_mapping=True, truncation=True, padding=False)["offset_mapping"]
    iob_tgs = ["O"] * len(tokens)

    for span in spns:
        start, end, label = span["start"], span["end"], span["label"]

        for idx, (start_offset, end_offset) in enumerate(token_offsets):
            if start_offset == start:
                iob_tgs[idx] = f"B-{label}"
                break

        for idx, (start_offset, end_offset) in enumerate(token_offsets):
            if start_offset > start and end_offset <= end:
                iob_tgs[idx] = f"I-{label}"
                break

    return list(zip(tokens, iob_tgs))


# Function to convert a list of texts and their spans to IOB format
def adjust_annotation_spans(new_sentence_start, anns):
    adjusted_annotations = []
    for ann in anns:
        start = ann["start"] - new_sentence_start
        end = ann["end"] - new_sentence_start
        adjusted_annotations.append({"start": start, "end": end, "label": ann["label"]})
    return adjusted_annotations


# Function to split a list of texts and their spans into sentences
def split_sentences(lnlp, anns, min_length):
    new_anns = []
    for ann in anns:
        txt = ann["text"]
        spns = ann["spans"]
        doc = lnlp(txt)
        start = 0
        for sent in doc.sents:
            if sent.end - start >= min_length:
                new_text = doc[start:sent.end].text
                new_spans = adjust_annotation_spans(doc[start].idx, spns)
                new_anns.append({"text": new_text, "spans": new_spans})
                start = sent.end
        if start < len(doc):
            new_text = doc[start:].text
            new_spans = adjust_annotation_spans(doc[start].idx, spns)
            new_anns.append({"text": new_text, "spans": new_spans})
    return new_anns


# Define the input file path
file_input_path = os.path.join(file_path, in_file)
# Define the output file path
file_output_path = os.path.join(file_path, out_file)

# Load the English tokenizer from spaCy
nlp = spacy.load("en_core_web_lg", disable=["parser", "ner", "textcat"])

# Add the "sentencizer" component to the pipeline
if not nlp.has_pipe("sentencizer"):
    nlp.add_pipe("sentencizer")

# Load the JSONL annotations file
with open(file_input_path, "r") as f:
    annotations = [json.loads(line) for line in f]

# Process the annotations by splitting the sentences with the specified minimum length
new_annotations = split_sentences(nlp, annotations, min_length=100)

# Open the output file
with open(file_output_path, "w") as output_file:
    # Iterate through all the annotations and convert them to IOB format
    for annotation in new_annotations:
        text = annotation["text"]
        spans = annotation["spans"]

        # Choose the tokenizer based on your preference
        use_huggingface = True  # Set this to True if you want to use Hugging Face's tokenizer

        if use_huggingface:
            if transformer_model == "bert-base-uncased":
                hf_tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
            elif transformer_model == "roberta-base":
                hf_tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
            elif transformer_model == "allenai/scibert_scivocab_uncased":
                hf_tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
            else:
                raise ValueError(f"Unsupported transformer model: {transformer_model}")

            iob_tags = convert_to_iob_huggingface(hf_tokenizer, text, spans)
        else:
            iob_tags = convert_to_iob_spacy(nlp, text, spans)

        # Write the IOB tags to the output file
        for token, tag in iob_tags:
            output_file.write(f"{token}\t{tag}\n")
        output_file.write("\n")


