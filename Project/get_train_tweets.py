import sys
import time
from TwitterAPI import TwitterAPI
import networkx as nx
import csv
import re
import pandas as pd


consumer_key = 'kSv03dEhZpOUyq2kVQ0fj2Hdt'
consumer_secret = 'rBBKNoHOCf9UXHqHvtQqYG1JWdxAwXqbrjLprlG6keeJ6QCeEa'
access_token = '1516682912-jHXVb56RCpXRbgO7klOS2IYCiEaXn4H6yYLjq8A'
access_token_secret = 'n92V9vEnoqd2AFaSUBOCWruTMqt49QEnHkazCO1hu2DVj'


# This method is done for you.
def get_twitter():
    """ Construct an instance of TwitterAPI using the tokens you entered above.
    Returns:
      An instance of TwitterAPI.
    """
    return TwitterAPI(consumer_key, consumer_secret, access_token, access_token_secret)


def robust_request(twitter, resource, params, max_tries=5):
    """ If a Twitter request fails, sleep for 15 minutes.
    Do this at most max_tries times before quitting.
    Args:
      twitter .... A TwitterAPI object.
      resource ... A resource string to request; e.g., "friends/ids"
      params ..... A parameter
       dict for the request, e.g., to specify
                   parameters like screen_name or count.
      max_tries .. The maximum number of tries to attempt.
    Returns:
      A TwitterResponse object, or None if failed.
    """
    for i in range(max_tries):
        request = twitter.request(resource, params)
        print(request.status_code)        
        if request.status_code == 200:
            return request
        else:
        	print('Got error %s \nsleeping for 15 minutes.' % request.text)
        	sys.stderr.flush()
        	time.sleep(61 * 15)    

def get_tweets(twitter,location,f):
    tweets = robust_request(twitter,'search/tweets', {'q': 'Trump','geocode':location,'lang':'en','count':100},5)          
    
    #f.write(columnTitleRow)

    for i in tweets:                
        string = i['text']
        string = re.sub(',', '', string)
        string = re.sub('\n', '', string)
        #f.write(string+'\n')        
        tweet = string
        user = i['user']['screen_name']       
        row = user + "," + tweet + "\n"
        print(row)
        f.write(row)
        #print(i)
        #print(string)
    return get_tweets

def tokenize(string, lowercase, keep_punctuation,
             collapse_urls, collapse_mentions):
    """ Split a tweet into tokens."""
    if not string:
        return []
    if lowercase:
        string = string.lower()
    tokens = []
    if collapse_urls:
        string = re.sub('http\S+', 'THIS_IS_A_URL', string)
    if collapse_mentions:
        string = re.sub('@\S+', 'THIS_IS_A_MENTION', string)
    if keep_punctuation:
        tokens = string.split()
    else:
        tokens = re.sub('\W+', ' ', string).split()    
    return tokens

def main():
    
    twitter = get_twitter()    
    #screen_names = read_screen_names('candidates.txt')    
    print('Established Twitter connection.')    
    f = open('train_tweets1.csv', 'w',encoding='utf-8')
    u = get_tweets(twitter,"47.010391,-120.391957,50mi",f) #washington tweets (democratic state)
    u = get_tweets(twitter,"36.129197,-119.381215,50mi",f) #California tweets (democratic state)
    u = get_tweets(twitter,"38.150363,-105.834915,50mi",f) #colorado tweets (democratic state)
    u = get_tweets(twitter,"40.838932,-89.190785,50mi",f) #Illinois tweets (democratic state)
    u = get_tweets(twitter,"42.670802,-72.864353,50mi",f) #Massachusetts tweets (democratic state)
    u = get_tweets(twitter,"46.488078,-111.146070,50mi",f) #Montana tweets (Republican state)
    u = get_tweets(twitter,"38.088192,-97.610914,50mi",f) #Kansas tweets (Republican state)
    u = get_tweets(twitter,"32.497126,-97.083570,50mi",f) #Texas tweets (Republican state)
    u = get_tweets(twitter,"33.382250,-87.635328,50mi",f) #Alabama tweets (Republican state)
    u = get_tweets(twitter,"26.338781,-81.230757,50mi",f) #Florida tweets (Republican state)

if __name__ == '__main__':
    main()
