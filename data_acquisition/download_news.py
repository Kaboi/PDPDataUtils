# %% Load libraries
import time

from GoogleNews import GoogleNews
from newspaper import Article
from newspaper import Config
import pandas as pd
import nltk
from tqdm import tqdm
from utilities import normalize_text as normalize
from fake_useragent import UserAgent


# %% functions
def initial_config():
    nltk.download('punkt')


def get_article(url):
    ua = UserAgent()
    config = Config()
    config.browser_user_agent = ua.random
    config.request_timeout = 10
    article = Article(url, config=config)
    article.download()
    article.parse()
    article.nlp()
    return article


def search_google_news(search, search_start_date, search_end_date, no_pages):
    df = None
    googlenews = GoogleNews(start=search_start_date, end=search_end_date, lang='en')
    googlenews.enableException(True)
    # error_count = 0
    for i in tqdm(range(1, no_pages+1), desc="Getting news pages", unit="pages", position=0, leave=True, ncols=100,
                  bar_format='{l_bar}{bar}|{n_fmt}/{total_fmt}'):

        try:
            result = None
            if i == 1:
                googlenews.search(search)
                result = googlenews.result()
            else:
                result = googlenews.page_at(i)

            if result is None:
                tqdm.write("No more results returned")
                break
            elif len(result) == 0:
                tqdm.write("No more news found")
                break
            else:
                if df is None:
                    df = pd.DataFrame(result)
                else:
                    df = pd.concat([df, pd.DataFrame(result)], ignore_index=True)

            # add time.sleep to delay the requests for getpage(i)
            time.sleep(5)
        # except Exception as e:
        except Exception:
            tqdm.write("It seems the pages are over, exiting...")
            break
            # print(type(e))
            # print(str(e))
            # error_count += 1
            # if error_count > 2:
            #     print ("\nToo many errors. Exiting...")
            #     break

    return df if df is not None else None


def populate_def_df(news_items_df, search_crop):
    # create dataframe
    data_list = list()
    for index, row in tqdm(news_items_df.iterrows(), desc="Getting news articles", unit="articles",
                           total=len(news_items_df), position=0, leave=True, ncols=100,
                           bar_format='{l_bar}{bar}|{n_fmt}/{total_fmt}'):
        # enclose the code block below in a try-except block to catch any errors
        try:
            if row['title'] and not row['title'].isspace() and row['title'] != '"':
                tqdm.write(f"Getting article: {row['link']}")
                article = get_article(row['link'])
                row = {"Crop": search_crop,
                       "Date": row['date'],
                       "URL": row['link'],
                       "Title": row['title'],
                       "Summary": article.summary,
                       "Text": normalize(article.text),
                       "Keywords": article.keywords}
                data_list.append(row)
        except Exception:
            print("Error getting article: ", row['link'])
            continue
    return pd.DataFrame.from_records(data_list)


# %% add search parameters
searchCrop = "Banana Plantain"
searchString = 'banana or plantain crop disease'
startDate = '01/01/2002'
endDate = '16/01/2023'
pageSize = 30

# %% initialize
initial_config()

# %% search for the papers
print("searching for news relating to ", searchCrop, "...\n")
news_items = search_google_news(searchString, startDate, endDate, pageSize)
print("search complete.")

# %% populate the dataframe
print("populating dataframe with news items...\n")
# check if news_items is not None then populate
news_df = None
if news_items is not None:
    news_df = populate_def_df(news_items, searchCrop)
    print("dataframe populated.")


# %% save the dataframe
if news_df is not None:
    print("saving dataframe...\n")
    filename = "data/" + searchCrop + "_News_Output.xlsx"
    print("saving the dataframe to file...", filename)
    news_df.to_excel(filename, index=False, engine='xlsxwriter')
