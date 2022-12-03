#%% Load libraries
import pandas as pd


def create_disease_line(disease):
    # {"label": "DISEASE", "pattern": [{"lower": "cassava"}, {"lower": "scale"}]}
    word_out = ''
    for word in disease.split():
        if len(word_out) == 0:
            word_out = word_out + '{"lower":"' + word + '"}'
        else:
            word_out = word_out + ',{"lower":"' + word + '"}'
    output = '{"label":"DISEASE","pattern":[' + word_out + ']}'
    return output


print(create_disease_line("Pseudocercospora leaf spot"))


#%% Load data
sweetpotato_df = pd.read_csv("data/Patterns-raw-sweet-potato.csv")
potato_df = pd.read_csv("data/Patterns-raw-potato.csv")
musa_df = pd.read_csv("data/Patterns-raw-Musa.csv")


#%% view data
sweetpotato_df.info()
potato_df.info()
musa_df.info()

#%% process data
# musa_disease = musa_df['Disease']

# for disease in musa_df['Disease'].items():
#     # if not disease.isnull():
#     #     print(disease)
#     print(type(disease))
#     print(disease)

musa_diseases = musa_df['Disease'].dropna()
print(musa_diseases)

#%% learn some string stuff

