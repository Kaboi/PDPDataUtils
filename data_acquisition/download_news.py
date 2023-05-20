# %% Load libraries
import time

from GoogleNews import GoogleNews
from newspaper import Article
from newspaper import Config
import pandas as pd
import nltk
from tqdm import tqdm
from utilities import normalize_text as normalize


# %% functions
def initial_config():
    nltk.download('punkt')
    user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 ' \
                 'Safari/537.36 Edg/108.0.1462.76'
    config = Config()
    config.browser_user_agent = user_agent
    config.request_timeout = 10
    return config


# add time.sleep to delay the requests for getpage(i)
def search_google_news(search, search_start_date, search_end_date, no_pages):
    df = None
    googlenews = GoogleNews(start=search_start_date, end=search_end_date)
    for i in tqdm(range(1, no_pages+1), desc="Getting news pages", unit="pages", position=0, leave=True, ncols=100,
                  bar_format='{l_bar}{bar}|{n_fmt}/{total_fmt}'):
        if i == 1:
            googlenews.search(search)
        else:
            googlenews.getpage(i)
        result = googlenews.result()
        if len(result) == 0:
            print("No more news found")
            break
        else:
            df = pd.DataFrame(result)
        time.sleep(5)

    return df if df is not None else None


def populate_def_df(news_items_df, search_crop, config):
    # create dataframe
    data_list = list()
    for index, row in tqdm(news_items_df.iterrows(), desc="Getting news articles", unit="articles",
                           total=len(news_items_df), position=0, leave=True, ncols=100,
                           bar_format='{l_bar}{bar}|{n_fmt}/{total_fmt}'):
        # enclose the code block below in a try-except block to catch any errors
        try:
            article = Article(row['link'], config=config)
            print("Getting article: ", row['link'])
            article.download()
            article.parse()
            article.nlp()
            row = {"Crop": search_crop,
                   "Date": row['date'],
                   "URL": row['link'],
                   "Title": row['title'],
                   "Summary": article.summary,
                   "Text": normalize(article.text),
                   "Keywords": article.keywords}
            data_list.append(row)
        except:
            print("Error getting article: ", row['link'])
            continue
    return pd.DataFrame.from_records(data_list)


# %% add search parameters
searchCrop = "Banana Plantain"
searchString = 'banana or plantain crop disease'
startDate = '01/01/2009'
endDate = '16/01/2023'
pageSize = 30

# %% initialize
search_config = initial_config()

# %% search for the papers
print("searching for ", searchCrop, " news...")
news_items = search_google_news(searchString, startDate, endDate, pageSize)
print("search complete.")

# %% populate the dataframe
print("populating dataframe with news items...")
# check if news_items is not None then populate
news_df = None
if news_items is not None:
    news_df = populate_def_df(news_items, searchCrop, search_config)
    print("dataframe populated.")


# %% save the dataframe
if news_df is not None:
    print("saving dataframe...")
    filename = "data/" + searchCrop + "_News_Output.xlsx"
    print("saving the dataframe to file...", filename)
    news_df.to_excel(filename, index=False, engine='xlsxwriter')
