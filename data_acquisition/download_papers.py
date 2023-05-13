# %% Load libraries
import pandas as pd
from semanticscholar import SemanticScholar
from textacy import preprocessing


# %% functions
def search_semantic_scholar(search, sfields, page_size):
    if page_size > 100:
        page_size = 100

    scholar = SemanticScholar()
    results = scholar.search_paper(search, fields=sfields, limit=page_size)
    return results


def populate_article_df(search_articles, search_limit, search_crop):
    # create dataframe
    num_of_skipped_records = 0
    data_list = list()
    while search_articles.next <= search_limit:
        for item in search_articles.items[search_articles.offset:search_articles.next:]:
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
                       "Abstract": preprocessing.normalize.whitespace(item.abstract)}
                data_list.append(row)
            else:
                num_of_skipped_records = num_of_skipped_records + 1

        if search_limit > search_articles.next and search_articles.next != 0:
            search_articles.next_page()
        else:
            break

    if num_of_skipped_records > 0:
        print(num_of_skipped_records, " empty abstracts or no identifier left out.")

    return pd.DataFrame.from_records(data_list)


# %% add search parameters
searchCrop = "Test"
searchString = 'Blueberry and Banana Consumption Mitigate Arachidonic'

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
articles_dataframe = populate_article_df(articles, searchLimit, searchCrop)
# print(articles_dataframe)

# %% save data frame as CSV
filename = "data/" + searchCrop + "_Output.xlsx"
print("saving the file to ", filename)
articles_dataframe.to_excel(filename, index=False, engine='xlsxwriter')
