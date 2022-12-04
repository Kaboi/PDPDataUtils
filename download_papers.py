# %% Load libraries
import pandas as pd
from semanticscholar import SemanticScholar


# %% functions
def search_semantic_scholar(search, sfields, page_size):
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
                       "Year": item.year,
                       "Title": item.title,
                       "Abstract": item.abstract}
                data_list.append(row)
            else:
                num_of_skipped_records = num_of_skipped_records + 1

        if search_limit > search_articles.next:
            search_articles.next_page()
        else:
            break

    if num_of_skipped_records > 0:
        print(num_of_skipped_records, " empty abstracts or no identifier left out.")

    return pd.DataFrame.from_records(data_list)


# %% add search parameters
searchCrop = "Cassava"
searchString = '"first report" cassava'
searchFields = ['externalIds', 'year', 'title', 'abstract']
# searchLimit ideally should be multiple of pagsize max is 100
pageSize = 100
searchLimit = 500

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
