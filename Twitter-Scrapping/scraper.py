# -*- coding: utf-8 -*-
"""scraper.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Gr3q4iQQepNB0IH_dVkQZAiYOtkKy26i

Necessary Libraries
"""

import tweepy
import json
import pandas as pd
import numpy as np

"""Upload credential file"""

from google.colab import files
uploaded = files.upload()

"""Read keys from Keys.json and establish connection """

# Credentials JSON file:
credentials = "keys.json"
with open(credentials, "r") as keys:
    api_tokens = json.load(keys)

# Grab the API keys:
consumer_key = api_tokens["api_key"]
consumer_secret = api_tokens["api_secret"]
access_token = api_tokens["access_token"]
access_token_secret = api_tokens["access_secret"]

# Setting up Tweepy authorization
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)

"""# Function for scrapping tweeter by passing parameters as arguments and returns a Pandas dataframe containing tweet data."""

def tweet_scraper(query=None, lang="en", tweet_mode="extended", count=100, tweet_limit=500):
    """
    :query: keyword
    :lang: default: English
    :tweet_mode: choose whether to extend tweets to full.
    :count: the number of tweets to return per page
    :tweet_limit: the maximum number of tweets to return (default 500).
    """

    # Creating a dictionary that will store our tweet data.

    data = {
        "user_id": [], 
        "screen_name": [],
        "created_at": [],
        "full_text": [],
        "retweet_count": [],
        "favorite_count": [],
        "followers_count": [],
        "friends_count": [],
        "location": [],
    }

    # Search the tweets

    for tweet in tweepy.Cursor(api.search, q=query, tweet_mode=tweet_mode, count=count).items(tweet_limit):
      
        # User Information
        # ---------------------------------------------------------------
        # User ID:
        data["user_id"].append(tweet.user.id)
        # Screen name:
        data["screen_name"].append(tweet.user.screen_name)
        # Date:
        data["created_at"].append(tweet.created_at)
        # Full text of tweet:
        data["full_text"].append(tweet.full_text)

        # Performance Matrics
        #------------------------------------------------------------------
        # Get retweet count:
        data["retweet_count"].append(tweet.retweet_count)
        # Get favorite count:
        data["favorite_count"].append(tweet.favorite_count)
        # Get followers count:
        data["followers_count"].append(tweet.user.followers_count)
        # Get friends count:
        data["friends_count"].append(tweet.user.friends_count)
        # Retrieve location
        data["location"].append(tweet.user.location)
        
    #save data dictionary into a Pandas dataframe
    df = pd.DataFrame(data)

    return df

"""# Collect 5000 tweets"""

# Set the function parameters:

search_keyword = "#BlackHistoryMonth OR #blackhistorymonth OR #Black_History_Month"
# Remove all retweets
query = search_keyword + " -filter:retweets"
lang = "en"
tweet_mode = "extended"
count = 100 
tweet_limit = 5000

# Call the function using parameters:
df = tweet_scraper(query=query, lang=lang, tweet_mode=tweet_mode, count=count, tweet_limit=tweet_limit)
#Remove tweets that have less than 5 words
df_full= df[df['full_text'].str.split().str.len().gt(5)]

df_full.iloc[[50]]

"""# Randomly collect 300 tweets """

df_sample = df_full.sample(n = 300)

"""# Save tweets data in json files

Full Dataset after removing retweets and tweets less than 5 words
"""

df_full.to_json("data full.json")
files.download("data full.json")

"""Sample Dataset with 300 data"""

df_sample.to_json("data sample.json")
files.download("data sample.json")
