import re
from textacy import preprocessing


# define a normalization function
def normalize_text(text):
    original_text_remove = text
    # join words split by a hyphen or line break
    text = preprocessing.normalize.hyphenated_words(text)

    # remove any unnecessary white spaces
    text = preprocessing.normalize.whitespace(text)

    # Replace three or more consecutive line breaks (accounting for spaces) with two
    text = re.sub(r'((\r\n|\r|\n)\s*){3,}', '\n\n', text)

    # subsitute fancy quatation marks with an ASCII equivalent
    text = preprocessing.normalize.quotation_marks(text)
    # normalize unicode characters in text into canonical forms
    text = preprocessing.normalize.unicode(text)
    # remove any accents character in text by replacing them with ASCII equivalents or removing them entirely
    text = preprocessing.remove.accents(text)

    return text
