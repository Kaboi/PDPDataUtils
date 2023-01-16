# %% Load libraries
from GoogleNews import GoogleNews
from newspaper import Article
from newspaper import Config
import pandas as pd
import nltk

# %% functions

def initial_config():
    nltk.download('punkt')
    USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 ' \
                 'Safari/537.36 Edg/108.0.1462.76'
    config = Config()
    config.browser_user_agent = USER_AGENT
    config.request_timeout = 10
    return


def search_google_news(search, search_start_date, search_end_date, no_pages):
    googlenews = GoogleNews(start=search_start_date, end=search_end_date)
    googlenews.search(search)
    result = googlenews.result()
    df = pd.DataFrame(result)

    for i in range(2, no_pages):
        googlenews.getpage(i)
        result = googlenews.result()
        df = df.append(result, ignore_index=True)

    return result


def populate_def_df(news_items_df, search_crop):
    # create dataframe
    data_list = list()
    for index, row in news_items_df.iterrows():
        # print(row)
        article = Article(row['link'], config=config)
        article.download()
        article.parse()
        article.nlp()
        row = {"Crop": search_crop,
               "Date": row['date'],
               "Title": row['title'],
               "Summary": article.summary,
               "Text": article.text,
               "Keywords": article.keywords}
        data_list.append(row)
    return pd.DataFrame.from_records(data_list)


# %% add search parameters
searchCrop = "Musa"
searchString = 'banana OR plantain + crop disease'
startDate = '01/01/2021'
endDate = '16/01/2023'
pageSize = 10

# %% initialize
initial_config()

# %% search for the papers
print("searching for news...")
news_items = search_google_news(searchString, startDate, endDate, pageSize)
print("search complete.")

# %% populate the dataframe
print("populating dataframe with news items...")
news_df = populate_def_df(news_items, searchCrop)
print("dataframe populated.")

# %% save the dataframe
print("saving dataframe...")
filename = "data/" + searchCrop + "News_Output.xlsx"
print("saving the dataframe to file...", filename)
news_df.to_excel(filename, index=False, engine='xlsxwriter')

