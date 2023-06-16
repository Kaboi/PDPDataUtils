import argparse
import json
import os
import spacy
from tqdm import tqdm


def generate_output_filename(in_file_path):
    file_name = os.path.splitext(os.path.basename(in_file_path))[0]
    dir_path = os.path.dirname(in_file_path)
    return os.path.join(dir_path, file_name + "-output-iob-tags.tsv")


# TODO combine with generate_output_filename
def generate_meta_filename(in_file_path):
    file_name = os.path.splitext(os.path.basename(in_file_path))[0]
    dir_path = os.path.dirname(in_file_path)
    return os.path.join(dir_path, file_name + "-output-iob-tags-meta.txt")


def convert_to_iob(lnlp, txt, spns, doc_hash=None):
    doc = lnlp(txt)
    iob_tgs = []

    for tkn in doc:
        tg = None
        for span in spns:
            if tkn.idx == span["start"]:
                tg = f"B-{span['label']}"
                break
            elif tkn.idx > span["start"] and tkn.idx + len(tkn.text) <= span["end"]:
                tg = f"I-{span['label']}"
                break

        # Edge cases if not well preprocessed. If token is a newline character, assign 'O' tag
        # This increase sequence/chunk size
        # TODO - add a logger to log these edge cases for now print an warning message
        if tkn.text.strip() == '' and not tg:
            # Log a warning if multiple newline characters are found
            if tkn.text.count("\n") > 1:
                tqdm.write(f"Found multiple newline characters in _input_hash {doc_hash} text at position {tkn.idx}.")

            # Add each newline character to iob_tgs with an 'O' tag
            for newline in tkn.text:
                iob_tgs.append(('', 'O'))
            continue

        if not tg:
            tg = "O"

        iob_tgs.append((tkn.text, tg))

    return iob_tgs


def adjust_annotation_spans(new_sentence_start, anns):
    adjusted_annotations = []
    for ann in anns:
        start = ann["start"] - new_sentence_start
        end = ann["end"] - new_sentence_start
        adjusted_annotations.append({"start": start, "end": end, "label": ann["label"]})
    return adjusted_annotations


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


def load_annotations(in_file_path):
    with open(in_file_path, "r") as f:
        anns = [json.loads(line) for line in f]
    return anns


def convert_to_iob_save(lnlp, anns, in_file_path, out_file_path, meta_file_path):
    max_text_sequence_length = 0
    total_number_of_tokens = 0
    total_number_of_annotated_documents = 0

    with open(out_file_path, "w") as output_file:
        for annotation in tqdm(anns, desc="Processing annotations"):
            if annotation["answer"] == "accept":
                total_number_of_annotated_documents += 1
                text = annotation["text"]
                spans = annotation["spans"]
                input_hash = annotation["_input_hash"]
                iob_tags = convert_to_iob(lnlp, text, spans, input_hash)
                for token, tag in iob_tags:
                    output_file.write(f"{token}\t{tag}\n")
                    total_number_of_tokens += 1
                output_file.write("\n")
                if len(iob_tags) > max_text_sequence_length:
                    max_text_sequence_length = len(iob_tags)
            else:
                # TODO - add a logger to log exclusion of reject and ignore print an warning message
                tqdm.write(f"Excluding annotation with _input_hash {annotation['_input_hash']} due to answer: "
                           f"{annotation['answer']}")

    # Write the metadata to a file
    with open(meta_file_path, "w") as meta_file:
        meta_file.write(f"Input file: {in_file_path}\n")
        meta_file.write(f"Output file: {out_file_path}\n")
        meta_file.write(f"Total number of annotation documents: {total_number_of_annotated_documents}\n")
        meta_file.write(f"Total number of tokens: {total_number_of_tokens}\n")
        meta_file.write(f"Longest sequence of tokens: {max_text_sequence_length}\n")


def main(file_path, language_model="en_core_web_lg", min_length=None):
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist.")
        return

    file_output_path = generate_output_filename(file_path)
    file_meta_path = generate_meta_filename(file_path)

    annotations = load_annotations(file_path)
    nlp = spacy.load(language_model, disable=["parser", "ner", "textcat"])

    if min_length:
        if not nlp.has_pipe("sentencizer"):
            nlp.add_pipe("sentencizer")

        annotations = split_sentences(nlp, annotations, min_length=min_length)

    # TODO - Do this more elegantly
    convert_to_iob_save(nlp, annotations, file_path, file_output_path, file_meta_path)


# # %% Testing
# # Edit paths here
# input_file_path = "/home/leroy/Dev/Code/PDPAnnotationDataUtils/annotated_data/tests-mine_2.jsonl"
# # in_file = "ciat_ner_v2_combined_rev_20230106.jsonl"
#
# # Set the model to use or default to spacy
# # Choose from "bert-base-uncased", "roberta-base", "allenai/scibert_scivocab_uncased",
# # "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", None : to use spacy tokenizer
# # transformer_model = "None"
# # transformer_model = "bert-base-uncased"
# # transformer_model = "roberta-base"
# # transformer_model = "allenai/scibert_scivocab_uncased"
# # transformer_model = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
#
# main(input_file_path)
# exit()


# %% Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert jsonl annotations to IOB format.')
    parser.add_argument('-f', '--file_path', required=True, help='Input file name with path.')
    parser.add_argument('-sm', '--spacy_model', default="en_core_web_lg", help='Spacy language model.')
    parser.add_argument('-l', '--min_length', default=None, type=int, help='Minimum sentence length, default is the '
                                                                           'document length')
    args = parser.parse_args()

    main(args.file_path, args.spacy_model, args.min_length)
