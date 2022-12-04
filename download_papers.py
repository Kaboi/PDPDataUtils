# %% Load libraries
import pandas as pd
from semanticscholar import SemanticScholar


# %% functions
def search_semantic_scholar(search, sfields, pageSize):
    scholar = SemanticScholar()
    results = scholar.search_paper(search, fields=sfields, limit=pageSize)
    return results


def populate_article_df(search_articles, search_limit):
    i = 1
    while search_articles.next <= search_limit:
        for item in search_articles.items[search_articles.offset:search_articles.next:]:
            print(i)
            i = i + 1
            print(item)
        search_articles.next_page()

# %% add search parameters
searchString = 'first report cassava'
searchFields = ['externalIds', 'year', 'title', 'abstract']
pageSize = 10
searchLimit = 15

# %% search for the papers

articles = search_semantic_scholar(searchString, searchFields, pageSize)
populate_article_df(articles, searchLimit)
print(articles.total)


# for item in articles.items:
#     print(item)

# %% load papers into Data Frame


# %% Learning
# # data = pd.DataFrame()
# data = pd.DataFrame(columns=["paperId", "Title", "Abstract"])
# data = data.append({"paperId": paper.paperId,
#                     "Title": paper.title,
#                     "Abstract": paper.abstract}, ignore_index=True)
# # data = data.append(pd.DataFrame([[paper.paperId, paper.title, paper.abstract]]))
# data.info()
#
# print(data)
