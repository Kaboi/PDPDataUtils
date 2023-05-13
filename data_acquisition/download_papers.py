# %% Load libraries
import re
import pandas as pd
from semanticscholar import SemanticScholar
from textacy import preprocessing
from tqdm import tqdm


# %% functions
def search_semantic_scholar(search, sfields, page_size):
    if page_size > 100:
        page_size = 100

    scholar = SemanticScholar()
    results = scholar.search_paper(search, fields=sfields, limit=page_size)
    return results


def populate_article_df(search_articles, search_limit, page_size, search_crop):
    # create dataframe
    num_of_skipped_records = 0
    data_list = list()
    if page_size > 100:
        page_size = 100

    # -(a // -b) for ceil division rather than floor
    display_counter: int = -((search_articles.total if search_limit > search_articles.total else search_limit) // -page_size)


    # Initialize the tqdm progress bar
    pbar = tqdm(total=display_counter, initial=0, desc="Processing articles", unit="pages", position=0, leave=True, ncols=100,
            bar_format='{l_bar}{bar}|{n_fmt}/{total_fmt}')

    for i in range(1, display_counter+1):
        if search_articles.next > search_limit:
            pbar.update(display_counter - pbar.n)  # update the remaining iterations
            break

        for item in search_articles.items[search_articles.offset:search_articles.next or None]:
            # print(item)
            if (item.abstract is not None) and (len(item.externalIds) > 0):
                keys = item.externalIds.keys()
                if 'DOI' in keys:
                    doi_value = item.externalIds['DOI']
                else:
                    key = list(keys)[0]
                    doi_value = key + ":" + str(item.externalIds[key])

                # doi = item.externalIds["DOI"]
                row = {"Crop": search_crop,
                       "DOI": doi_value,
                       "URL": item.url,
                       "Year": item.year,
                       "Title": item.title,
                       "Abstract": normalize(item.abstract)}
                data_list.append(row)
            else:
                num_of_skipped_records = num_of_skipped_records + 1

        if search_limit > search_articles.next != 0:
            search_articles.next_page()
        else:
            break

        # Update the progress bar at the end of each loop
        pbar.update()

    # If loop finished prematurely, make sure progress bar is at 100%
    if pbar.n < display_counter:
        pbar.update(display_counter - pbar.n)

    pbar.close()

    if num_of_skipped_records > 0:
        print(num_of_skipped_records, " empty abstracts or no identifier left out.")

    return pd.DataFrame.from_records(data_list)


# define a normalization function
def normalize(text):
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

# %% add search parameters
searchCrop = "Cassava"
searchString = "First report of Cassava"
searchFields = ['url', 'externalIds', 'year', 'title', 'abstract']
# searchLimit ideally should be multiple of pagesize and > than pagesize
# max pagesize is 100
pageSize = 10
searchLimit = 50

# %% search for the papers
print("searching for the papers...")
articles = search_semantic_scholar(searchString, searchFields, pageSize)

# %% load papers into Data Frame
print("populating data to a limit of ", searchLimit)
articles_dataframe = populate_article_df(articles, searchLimit, pageSize, searchCrop)
# print(articles_dataframe)

# %% save data frame as CSV
filename = "data/" + searchCrop + "_Output.xlsx"
print("saving the file to ", filename)
articles_dataframe.to_excel(filename, index=False, engine='xlsxwriter')
