# %% Load libraries
import pandas as pd
from semanticscholar import SemanticScholar


# %% functions
def search_semantic_scholar(search, sfields, pageSize):
    scholar = SemanticScholar()
    results = scholar.search_paper(search, fields=sfields, limit=pageSize)
    return results


def populate_article_df(search_articles, search_limit):
    # create dataframe
    data = pd.DataFrame(columns=["DOI", "Year", "Title", "Abstract"])
    data_list = list()
    while search_articles.next <= search_limit:
        for item in search_articles.items[search_articles.offset:search_articles.next:]:
            # print(item)
            row = {"DOI": item.externalIds["DOI"],
                   "Year": item.year,
                   "Title": item.title,
                   "Abstract": item.abstract}
            data_list.append(row)

        if search_limit > search_articles.next:
            search_articles.next_page()
        else:
            break

    data = pd.DataFrame.from_records(data_list)
    return data


# %% add search parameters
searchString = 'first report cassava'
searchFields = ['externalIds', 'year', 'title', 'abstract']
# searchLimit ideally should be multiple of pagsize
pageSize = 5
searchLimit = 5

# %% search for the papers

articles = search_semantic_scholar(searchString, searchFields, pageSize)
articles_dataframe = populate_article_df(articles, searchLimit)
print(articles_dataframe)

# for item in articles.items:
#     print(item)

# %% load papers into Data Frame

