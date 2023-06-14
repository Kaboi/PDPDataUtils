# %% Load libraries
from utilities import normalize_text as normalize
import time
import argparse
import nltk
import pandas as pd
from newspaper import Article, ArticleException, Config
from tqdm import tqdm
from GoogleNews import GoogleNews
from fake_useragent import UserAgent
from datetime import datetime, timedelta


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
    countdown = no_pages
    # error_count = 0
    for i in tqdm(range(1, no_pages+1), desc="Getting news pages", unit="pages", position=0, leave=True, ncols=100,
                  bar_format='{l_bar}{bar}|{n_fmt}/{total_fmt} Pages'):

        try:
            result = None
            tqdm.write(f"Getting page {i} of possible {no_pages} pages...")
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
            rest = 5
            countdown -= 1
            tqdm.write(f"Giving Google a {rest} seconds pause ....{countdown} pages left")
            time.sleep(rest)
        # except Exception as e:
        except Exception:
            tqdm.write("We have an error either no more pages or you are blocked. Exiting...")
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
    num_of_failures = 0
    total_articles = len(news_items_df)
    for index, row in tqdm(news_items_df.iterrows(), desc="Getting news articles", unit="articles",
                           total=total_articles, position=0, leave=True, ncols=100,
                           bar_format='{l_bar}{bar}|{n_fmt}/{total_fmt} Articles'):
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
        except ArticleException as ae:
            # Handle specific ArticleException
            tqdm.write(f"**Error : {str(ae)}")
            num_of_failures += 1
        except Exception as e:
            tqdm.write(f"**Error getting : {row['link']} - Error Message:{str(e)} - Error Class:{type(e)}")
            num_of_failures += 1
            continue

    print((
        f"{total_articles - num_of_failures} out of {total_articles} "
        f"({round((total_articles - num_of_failures) / total_articles * 100, 2)}%) "
        "articles downloaded. "
        f"{num_of_failures} downloads failed."
    ))

    return pd.DataFrame.from_records(data_list)


# %% main
def main(search_crop, search_string, start_date, end_date, page_size):
    # initialize
    initial_config()

    # search for the papers
    print("\nsearching for news relating to ", search_crop, "...\n")
    news_items = search_google_news(search_string, start_date, end_date, page_size)
    print("search complete.")

    # populate the dataframe
    print("populating dataframe with news items...")
    # check if news_items is not None then populate
    news_df = None
    if news_items is not None:
        news_df = populate_def_df(news_items, search_crop)
        print("dataframe populated.")

    # save the dataframe
    if news_df is not None:
        print("saving dataframe...")
        filename = "data/" + search_crop + "_News_Output.xlsx"
        print("saving the dataframe to file...", filename)
        news_df.to_excel(filename, index=False, engine='xlsxwriter')


# %% test
# # %% add search parameters
# searchCrop = "Sweetpotato"
# searchString = '"sweet potato" and sweetpotato crop disease'
# startDate = '01/01/1992'
# endDate = '22/05/2023'
# pageSize = 50
# main("Sweetpotato", '"sweet potato" and sweetpotato crop disease', '01/01/1992', '22/05/2023', 2)

# %% call main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download news articles from Google News')
    parser.add_argument('-c', '--crop', type=str, required=True, help='Crop name')
    parser.add_argument('-s', '--search', type=str, required=True, help='Search string')
    parser.add_argument('-sd', '--startdate', type=str, help='Start date',
                        default=(datetime.today() - timedelta(days=30)).strftime("%d/%m/%Y"))
    parser.add_argument('-ed', '--enddate', type=str, help='End date', default=datetime.today().strftime("%d/%m/%Y"))
    parser.add_argument('-p', '--pagesize', type=int, help='Page size', default=1)
    args = parser.parse_args()

    main(args.crop, args.search, args.startdate, args.enddate, args.pagesize)
