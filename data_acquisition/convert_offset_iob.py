import argparse
import json
import os
import spacy


def generate_output_filename(in_file_path):
    file_name = os.path.splitext(os.path.basename(in_file_path))[0]
    dir_path = os.path.dirname(in_file_path)
    return os.path.join(dir_path, file_name + "-output-iob-tags.tsv")


def convert_to_iob(lnlp, txt, spns):
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


def convert_to_iob_save(lnlp, anns, out_file_path):
    max_token_length = 0
    with open(out_file_path, "w") as output_file:
        for annotation in anns:
            text = annotation["text"]
            spans = annotation["spans"]
            iob_tags = convert_to_iob(lnlp, text, spans)
            for token, tag in iob_tags:
                output_file.write(f"{token}\t{tag}\n")
            output_file.write("\n")
            if len(iob_tags) > max_token_length:
                max_token_length = len(iob_tags)
    return max_token_length


def main(file_path, language_model="en_core_web_lg", min_length=None):
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist.")
        return
    file_output_path = generate_output_filename(file_path)
    annotations = load_annotations(file_path)
    nlp = spacy.load(language_model, disable=["parser", "ner", "textcat"])

    if min_length:
        if not nlp.has_pipe("sentencizer"):
            nlp.add_pipe("sentencizer")

        annotations = split_sentences(nlp, annotations, min_length=min_length)

    longest_chunk = convert_to_iob_save(nlp, annotations, file_output_path)
    print(f"Longest sequence: {longest_chunk}")


# # %% Testing
# # Edit paths here
# input_file_path = "/home/leroy/Dev/Code/PDPAnnotationDataUtils/annotated_data/tests-mine.jsonl"
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



