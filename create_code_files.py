# %% Load libraries
import pandas as pd

# %% functions
def create_code_text_line(data, element):
    # {"label": "DISEASE", "pattern": [{"lower": "cassava"}, {"lower": "scale"}]}

    word_out: str = ''
    output: str = ''
    for word in data.split():
        if len(word_out) == 0:
            word_out = word_out + '{"lower":"' + word + '"}'
        else:
            word_out = word_out + ',{"lower":"' + word + '"}'

    if element == 'disease':
        output = '{"label":"DISEASE","pattern":[' + word_out + ']}'
    elif element == 'pathogen':
        output = '{"label":"PATHOGEN","pattern":[' + word_out + ']}'

    return output


# %% Load data
sweetpotato_df = pd.read_csv("data/Patterns-raw-sweet-potato.csv")
potato_df = pd.read_csv("data/Patterns-raw-potato.csv")
musa_df = pd.read_csv("data/Patterns-raw-Musa.csv")

# %% view data
# sweetpotato_df.info()
# potato_df.info()
# musa_df.info()

# %% process data

musa_disease_code = musa_df['Disease'].dropna().apply(
    lambda cropdf: create_code_text_line(cropdf, 'disease'))
potato_disease_code = potato_df['Disease'].dropna().apply(
    lambda cropdf: create_code_text_line(cropdf, 'disease'))
sweetpotato_disease_code = sweetpotato_df['Disease'].dropna().apply(
    lambda cropdf: create_code_text_line(cropdf, 'disease'))
musa_pathogen_code = musa_df['Pathogen'].dropna().apply(
    lambda cropdf: create_code_text_line(cropdf, 'pathogen'))
potato_pathogen_code = potato_df['Pathogen'].dropna().apply(
    lambda cropdf: create_code_text_line(cropdf, 'pathogen'))
sweetpotato_pathogen_code = sweetpotato_df['Pathogen'].dropna().apply(
    lambda cropdf: create_code_text_line(cropdf, 'pathogen'))

# %% create back the code

with open("data/output.jsonl", "w+") as f:
    # sweetpotato pathogens
    for lines in sweetpotato_pathogen_code:
        f.write(lines)
        f.write('\n')
    # sweetpotato diseases
    for lines in sweetpotato_disease_code:
        f.write(lines)
        f.write('\n')
    # Potato pathogens
    for lines in potato_pathogen_code:
        f.write(lines)
        f.write('\n')
    # Potato diseases
    for lines in potato_disease_code:
        f.write(lines)
        f.write('\n')
    # Musa pathogens
    for lines in musa_pathogen_code:
        f.write(lines)
        f.write('\n')
    # Musa diseases
    for lines in musa_disease_code:
        f.write(lines)
        f.write('\n')