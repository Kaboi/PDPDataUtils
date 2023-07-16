import argparse
import json
import os
import spacy
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import csv
from collections import defaultdict

def generate_output_filename(in_file_paths, min_sentence_length):
    file_name = os.path.splitext(os.path.basename(in_file_paths[0]))[0]
    dir_path = os.path.dirname(in_file_paths[0])
    return os.path.join(dir_path, file_name + "-output-iob-tags-" +
                        (str(min_sentence_length) if min_sentence_length else "full_doc") + ".txt")


def generate_meta_filename(in_file_paths, min_sentence_length):
    file_name = os.path.splitext(os.path.basename(in_file_paths[0]))[0]
    dir_path = os.path.dirname(in_file_paths[0])
    return os.path.join(dir_path, file_name + "-output-iob-tags-" +
                        (str(min_sentence_length) if min_sentence_length else "full_doc") + ".meta.txt")


def adjust_annotation_spans(new_sentence_start, anns):
    adjusted_annotations = []
    for ann in anns:
        start = ann["start"] - new_sentence_start
        end = ann["end"] - new_sentence_start
        adjusted_annotations.append({"start": start, "end": end, "label": ann["label"]})
    return adjusted_annotations


def split_sentences(lnlp, anns, min_length):
    if isinstance(anns, tuple):
        return tuple(split_sentences(lnlp, group, min_length) for group in anns)
    else:
        new_anns = []
        for ann in anns:
            txt = ann["text"]
            spns = ann["spans"]
            doc = lnlp(txt)
            start = 0
            input_hash = ann["_input_hash"]
            for sent in doc.sents:
                if sent.end - start >= min_length:
                    new_text = doc[start:sent.end].text
                    new_spans = adjust_annotation_spans(doc[start].idx, spns)
                    new_anns.append({"text": new_text, "spans": new_spans, "_input_hash": input_hash})
                    start = sent.end
            if start < len(doc):
                new_text = doc[start:].text
                new_spans = adjust_annotation_spans(doc[start].idx, spns)
                new_anns.append({"text": new_text, "spans": new_spans, "_input_hash": input_hash})
        return new_anns


def load_annotations(in_file_paths):
    anns = []
    for in_file_path in in_file_paths:
        with open(in_file_path, "r") as f:
            anns.extend([json.loads(line) for line in f])

    return anns, len(anns)


# function that gets the annotations and removes all annotations that the answer is not accept
def filter_annotations(anns):
    filtered_anns = []
    num_of_docs = 0
    for ann in anns:
        if ann["answer"] == "accept":
            filtered_anns.append(ann)
            num_of_docs += 1
        else:
            # TODO - add a logger to log exclusion of reject and ignore print an warning message
            print(f"Excluding annotation with _input_hash {ann['_input_hash']} due to answer: {ann['answer']}")
    return filtered_anns, num_of_docs


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

            # Add each newline character as a space to iob_tgs with an 'O' tag
            # for _ in tkn.text:
            #     iob_tgs.append((' ', 'O'))

            # add only one newline character as a space to iob_tgs with an 'O' tag
            iob_tgs.append((' ', 'O'))
            continue

        if not tg:
            tg = "O"

        iob_tgs.append((tkn.text, tg))

    return iob_tgs


def convert_to_iob_save(lnlp, anns, maximum_sentence_length, in_file_paths, out_file_path, meta_file_path, n_docs, separator=','):

    label_counts = defaultdict(int)
    combined_label_counts = defaultdict(int)

    if isinstance(anns, tuple):
        data_types = ["train", "test", "validate"]
        return tuple(convert_to_iob_save(lnlp, group, maximum_sentence_length, in_file_paths,
                                         f"{os.path.splitext(out_file_path)[0]}_{data_types[i]}{os.path.splitext(out_file_path)[1]}",
                                         f"{os.path.splitext(meta_file_path)[0]}_{data_types[i]}{os.path.splitext(meta_file_path)[1]}",
                                         n_docs[i], separator) for i, group in enumerate(anns))
    else:
        max_text_sequence_length = 0
        total_number_of_tokens = 0
        current_combined_label = None

        with open(out_file_path, "w", newline='') as output_file:
            if separator == ',':
                csv_writer = csv.writer(output_file, delimiter=separator, quoting=csv.QUOTE_MINIMAL)
            else:
                csv_writer = csv.writer(output_file, delimiter=separator, quoting=csv.QUOTE_NONE, quotechar='',
                                        escapechar='\\', lineterminator='\n')

            for annotation in tqdm(anns, desc="Processing annotations"):
                text = annotation["text"]
                spans = annotation["spans"]
                input_hash = annotation["_input_hash"]
                iob_tags = convert_to_iob(lnlp, text, spans, input_hash)

                tokens_in_current_sequence = 0
                for token, tag in iob_tags:
                    csv_writer.writerow([token, tag])
                    total_number_of_tokens += 1
                    tokens_in_current_sequence += 1

                    label_counts[tag] += 1  # count individual label

                    # for combined labels, consider only B- and I- tags
                    if tag.startswith("B-") or tag.startswith("I-"):
                        label_name = tag[2:]
                        combined_label_counts[label_name] += 1

                    if maximum_sentence_length is not None and tokens_in_current_sequence >= maximum_sentence_length:
                        if (token.isspace() or token in [",", "."]) and tag == "O":
                            csv_writer.writerow([])
                            if tokens_in_current_sequence > max_text_sequence_length:
                                max_text_sequence_length = tokens_in_current_sequence
                            tokens_in_current_sequence = 0

                if tokens_in_current_sequence > 0:
                    csv_writer.writerow([])
                if tokens_in_current_sequence > max_text_sequence_length:
                    max_text_sequence_length = tokens_in_current_sequence

        with open(meta_file_path, "w") as meta_file:
            for in_file_path in in_file_paths:
                meta_file.write(f"Input file: {in_file_path}\n")
            meta_file.write(f"Output file: {out_file_path}\n")
            meta_file.write(f"Total number of annotation documents: {n_docs}\n")
            meta_file.write(f"Total number of tokens: {total_number_of_tokens}\n")
            meta_file.write(f"Longest sequence of tokens: {max_text_sequence_length}\n")
            meta_file.write("Label Counts:\n")
            for label, count in label_counts.items():
                meta_file.write(f"{label}: {count}\n")

            meta_file.write("\nCombined Label Counts:\n")
            for label, count in combined_label_counts.items():
                meta_file.write(f"{label}: {count}\n")



# def convert_to_iob_save(lnlp, anns, maximum_sentence_length, in_file_paths, out_file_path, meta_file_path, n_docs):
#     if isinstance(anns, tuple):
#         data_types = ["train", "test", "validate"]
#         return tuple(convert_to_iob_save(lnlp, group, maximum_sentence_length, in_file_paths,
#                                          f"{os.path.splitext(out_file_path)[0]}_{data_types[i]}"
#                                          f"{os.path.splitext(out_file_path)[1]}",
#                                          f"{os.path.splitext(meta_file_path)[0]}_{data_types[i]}"
#                                          f"{os.path.splitext(meta_file_path)[1]}",
#                                          n_docs[i]) for i, group in enumerate(anns))
#     else:
#         max_text_sequence_length = 0
#         total_number_of_tokens = 0
#
#         with open(out_file_path, "w") as output_file:
#             for annotation in tqdm(anns, desc="Processing annotations"):
#                 text = annotation["text"]
#                 spans = annotation["spans"]
#                 input_hash = annotation["_input_hash"]
#                 iob_tags = convert_to_iob(lnlp, text, spans, input_hash)
#
#                 tokens_in_current_sequence = 0
#                 for token, tag in iob_tags:
#                     output_file.write(f"{token}\t{tag}\n")
#                     total_number_of_tokens += 1
#                     tokens_in_current_sequence += 1
#
#                     # Check if a break should be added
#                     if maximum_sentence_length is not None and tokens_in_current_sequence >= maximum_sentence_length:
#                         if (token.isspace() or token in [",", "."]) and tag == "O":
#                             output_file.write("\n")
#
#                             # Update max_text_sequence_length
#                             if tokens_in_current_sequence > max_text_sequence_length:
#                                 max_text_sequence_length = tokens_in_current_sequence
#
#                             tokens_in_current_sequence = 0
#
#                 # Separate different annotations with a newline
#                 if tokens_in_current_sequence > 0:
#                     output_file.write("\n")
#                 # Update max_text_sequence_length
#                 if tokens_in_current_sequence > max_text_sequence_length:
#                     max_text_sequence_length = tokens_in_current_sequence
#
#         # Write the metadata to a file
#         with open(meta_file_path, "w") as meta_file:
#             for in_file_path in in_file_paths:
#                 meta_file.write(f"Input file: {in_file_path}\n")
#             meta_file.write(f"Output file: {out_file_path}\n")
#             meta_file.write(f"Total number of annotation documents: {n_docs}\n")
#             meta_file.write(f"Total number of tokens: {total_number_of_tokens}\n")
#             meta_file.write(f"Longest sequence of tokens: {max_text_sequence_length}\n")


def split_data_into_train_val_test(annotations, split_ratios=(0.7, 0.15, 0.15), random_seed=33):
    train_ratio, validation_ratio, test_ratio = split_ratios

    # First, split the data into training and temporary test
    train_annotations, temp_test_annotations = train_test_split(
        annotations, train_size=train_ratio, random_state=random_seed)

    # Then, split the temporary test data into validation and test
    temp_ratio = test_ratio / (1 - train_ratio)
    validation_annotations, test_annotations = train_test_split(
        temp_test_annotations, test_size=temp_ratio, random_state=random_seed)

    # Get the sizes
    train_size = len(train_annotations)
    validation_size = len(validation_annotations)
    test_size = len(test_annotations)

    return (train_annotations, validation_annotations, test_annotations), (train_size, validation_size, test_size)


def main(file_paths, min_length=None, max_length=None, split=False, language_model="en_core_web_lg", separator=','):
    if min_length and not max_length:
        max_length = min_length

    nlp = spacy.load(language_model, disable=["parser", "ner", "textcat"])
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"Error: File {file_path} does not exist.")
            return

    annotations, num_of_docs = load_annotations(file_paths)
    print("number of docs: ", num_of_docs)
    annotations, num_of_docs = filter_annotations(annotations)
    print("number of docs after filter: ", num_of_docs)

    if split:
        print(f"Splitting data into train, validation and test sets...")
        annotations, num_of_docs = split_data_into_train_val_test(annotations)

    if min_length:
        if not nlp.has_pipe("sentencizer"):
            nlp.add_pipe("sentencizer")

        print(f"Splitting into sentences with a minimum length of {min_length}...")
        annotations = split_sentences(nlp, annotations, min_length=min_length)

    # TODO - Do this more elegantly
    file_output_path = generate_output_filename(file_paths, min_length)
    file_meta_path = generate_meta_filename(file_paths, min_length)
    print(f"Converting to IOB format and saving...")
    convert_to_iob_save(nlp, annotations, max_length, file_paths, file_output_path, file_meta_path, num_of_docs,
                        separator=separator)


# %% Quick test runs
# # Edit paths here
# input_file_path = "/home/leroy/Dev/Code/PDPAnnotationDataUtils/annotated_data/tests-mine_4.jsonl"
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
# filepaths = []
# filepaths.append(input_file_path)
# main(filepaths)
# exit()
#

# %% Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert jsonl annotations to IOB format.')
    parser.add_argument('-f', '--file_paths', nargs='+', required=True, help='Input file names, can be multiple space'
                                                                             ' separated files.')
    parser.add_argument('-min', '--min_length', default=None, type=int, help='Minimum sentence length, default is the '
                                                                             'document length')
    parser.add_argument('-max', '--max_length', default=None, type=int, help='Maximum ballpark sentence length, default'
                                                                             ' is the document length')
    # add an argument for splitting the data into train, validation and test
    parser.add_argument('-s', '--split', default=False, action='store_true', help='Split the data into train, test '
                                                                                  'and validation sets.')
    parser.add_argument('-sm', '--spacy_model', default="en_core_web_lg", help='Spacy language model.')
    # add an argument for the separator
    parser.add_argument('-sep', '--separator', default=',', help='Separator for the output file. Allowed values are ","'
                                                                 ' (comma), \\t (tab) and \s (space) Default is comma.')

    args = parser.parse_args()

    if args.separator == '\\t':
        args.separator = '\t'
    elif args.separator == '\\s':
        args.separator = ' '

    assert args.separator in [',', '\t', ' '], "Separator must be either a comma, tab (\\t), or space (\\s)"

    # Assert that min_length is less than max_length
    assert args.min_length is None or args.max_length is None or args.min_length <= args.max_length, \
        "min_length must be less than or equal to max_length"

    main(args.file_paths, args.min_length, args.max_length, args.split, args.spacy_model, args.separator)

