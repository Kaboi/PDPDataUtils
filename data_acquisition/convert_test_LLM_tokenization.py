# %% imports
import json
import os
import spacy
from transformers import BertTokenizerFast, RobertaTokenizerFast, AutoTokenizer

# %% Functions
def generate_output_filename(in_file_path, transformer="None"):
    # Get the file name including base without the extension
    file_name = os.path.splitext(os.path.basename(in_file_path))[0]
    dir_path = os.path.dirname(in_file_path)
    # Define the output file name
    if transformer != "None":
        return os.path.join(dir_path, file_name + "-output-iob-tags-" + transformer + ".tsv")
    else:
        return os.path.join(dir_path, file_name + "-output-iob-tags.tsv")


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


# Function to load the JSONL annotations file
def load_annotations(in_file_path):
    with open(in_file_path, "r") as f:
        # load json objects from jsonl and remove extra whitespaces from the text using textacy
        anns = [json.loads(line) for line in f]
    return anns


def convert_to_iob_save(anns, out_file_path, transformer="None") -> int:
    # Initialize the max token length
    max_token_length = 0

    # Open the output file
    with open(out_file_path, "w") as output_file:
        # Iterate through all the annotations and convert them to IOB format
        for annotation in anns:
            text = annotation["text"]
            spans = annotation["spans"]

            if transformer != "None":
                if transformer == "bert-base-uncased":
                    hf_tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
                elif transformer == "roberta-base":
                    hf_tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
                elif transformer == "allenai/scibert_scivocab_uncased":
                    hf_tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
                else:
                    raise ValueError(f"Unsupported transformer model: {transformer}")

                iob_tags = convert_to_iob_huggingface(hf_tokenizer, text, spans)
            else:
                iob_tags = convert_to_iob_spacy(nlp, text, spans)

            # Write the IOB tags to the output file
            for token, tag in iob_tags:
                output_file.write(f"{token}\t{tag}\n")
            output_file.write("\n")

            # Update the max token length if the current token sequence is longer
            if len(iob_tags) > max_token_length:
                max_token_length = len(iob_tags)

    return max_token_length


# %% variables
# Edit paths here
file_path = "/home/leroy/Dev/Code/PDPAnnotationDataUtils/annotated_data"
in_file = "tests-mine.jsonl"
# in_file = "ciat_ner_v2_combined_rev_20230106.jsonl"

# Set the model to use or default to spacy
# Choose from "bert-base-uncased", "roberta-base", "allenai/scibert_scivocab_uncased",
# "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", None : to use spacy tokenizer
transformer_model = "None"
# transformer_model = "bert-base-uncased"
# transformer_model = "roberta-base"
# transformer_model = "allenai/scibert_scivocab_uncased"
# transformer_model = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"


# %% Main

# Define the input file path
file_input_path = os.path.join(file_path, in_file)
# Define the output file path
file_output_path = generate_output_filename(file_input_path, transformer_model)

annotations = load_annotations(file_input_path)

# Load the English tokenizer from spaCy
nlp = spacy.load("en_core_web_lg", disable=["parser", "ner", "textcat"])

# Add the "sentencizer" component to the pipeline
if not nlp.has_pipe("sentencizer"):
    nlp.add_pipe("sentencizer")

# Process the annotations by splitting the sentences with the specified minimum length
new_annotations = split_sentences(nlp, annotations, min_length=100)

longest = convert_to_iob_save(new_annotations, file_output_path, transformer_model)
print(f"Longest sequence: {longest}")


# %% quick tests

from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# %% quick tests
text = "Fusarium wilt disease , caused by the fungus Fusarium oxysporum f. sp. cubense race 4 , has been reported to be the most limiting factor in"
labels = ["I-DISEASE", "I-DISEASE", "I-DISEASE", "O", "O", "O", "O", "O", "B-PATHOGEN", "I-PATHOGEN", "I-PATHOGEN",
          "I-PATHOGEN", "I-PATHOGEN", "I-PATHOGEN", "I-PATHOGEN", "I-PATHOGEN", "O", "O", "O", "O", "O", "O", "O", "O",
          "O"]


print(text)
print(labels)

# %% quick tests
tokenized_text = []
tokenized_labels = []

for word, label in zip(text.split(), labels):
    # Tokenize the word
    word_tokens = tokenizer.tokenize(word)

    # If the word was not split into subwords
    if len(word_tokens) == 1:
        tokenized_text.append(word_tokens[0])
        tokenized_labels.append(label)
    else:
        # If the word was split into subwords
        for i, word_token in enumerate(word_tokens):
            tokenized_text.append(word_token)
            if i == 0:
                tokenized_labels.append(label)
            else:
                tokenized_labels.append("X")

# %% quick tests

print(tokenized_text)
print(tokenized_labels)

