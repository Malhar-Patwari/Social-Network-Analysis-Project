"""
classify.py
"""
import networkx as nx
import matplotlib.pyplot as plt
import csv
import re
import pandas as pd
from collections import defaultdict
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
import numpy as np
from itertools import product
from scipy.sparse import lil_matrix
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
import pickle



def read_Afinn():
	url = urlopen('http://www2.compute.dtu.dk/~faan/data/AFINN.zip')
	zipfile = ZipFile(BytesIO(url.read()))
	afinn_file = zipfile.open('AFINN/AFINN-111.txt')

	afinn = dict()

	for line in afinn_file:
	    parts = line.strip().split()
	    if len(parts) == 2:
	        afinn[parts[0].decode("utf-8")] = int(parts[1])

	return afinn

def read_Afinn_test_tweets(filename):
	tweets = pd.read_csv(filename,header=None,names=['user', 'text'])
	return tweets

def read_train_tweets(filename):
	tweets = pd.read_csv(filename,header=None,names=['user', 'text','polarity'])
	return tweets

def read_ML_test_tweets(filename):
	tweets = pd.read_csv(filename,header=0,names=['Index','user', 'text','Afinn'])
	#print(tweets.head())
	#print(tweets.tail())
	#print("lenngth:",len(tweets))
	return tweets


def using_Afinn(tweets,afinn):
	ind = -1
	sentiment = []
	for i in tweets['text']:
		ind+=1
		tokens = tokenize(i,True,True,True,True,False,True)
		sent = 0
		for j in tokens:
			if j in afinn:
				sent+=afinn[j]
		sentiment.append(sent)
	#sentiment = pd.DataFrame(np.array(sentiment))
	#print,encoding='utf-8'(sentiment)
	tweets['Afinn_sentiment'] = pd.Series(sentiment, tweets.index)
	#print(tweets)
	#tweets.append(loc = 2,column='sentiment',value = sentiment)		
	return tweets

def write_to_file(tweets,filename):
	tweets.to_csv(filename, sep=',',encoding='utf-8')
	print("Sentiment written to file %s" %filename)


def tokenize(string, lowercase, keep_punctuation,collapse_urls, collapse_mentions,collapse_hashtags,collapse_retweets):    
    if not string:
        return []
    if lowercase:
        string = string.lower()
    tokens = []
    if collapse_urls:
        string = re.sub('http\S+', 'THIS_IS_A_URL', string)
    if collapse_mentions:
        string = re.sub('@\S+', 'THIS_IS_A_MENTION', string)
    if collapse_hashtags:
    	string = re.sub('#\S+', 'THIS_IS_A_HASHTAG', string)
    if collapse_retweets:
    	string = re.sub('RT', '', string)	
    if keep_punctuation:
        tokens = string.split()
    else:
        tokens = re.sub('\W+', ' ', string).split()

    return tokens

def make_vocabulary(tokens_list):
    vocabulary = defaultdict(lambda: len(vocabulary))
    for tokens in tokens_list:
        for token in tokens:
            vocabulary[token]  
    return vocabulary    

def make_feature_matrix(tokens_list, vocabulary):
    X = lil_matrix((len(tokens_list), len(vocabulary)))
    for i, tokens in enumerate(tokens_list):
        for token in tokens:
            j = vocabulary[token]
            X[i,j] += 1
    return X.tocsr()

def do_cross_val(X, y, nfolds):
    """ Compute average cross-validation acccuracy."""
    cv = KFold(n_splits=nfolds, random_state=42, shuffle=True)
    accuracies = []
    for train_idx, test_idx in cv.split(X):
        clf = LogisticRegression()
        clf.fit(X[train_idx], y[train_idx])
        predicted = clf.predict(X[test_idx])
        acc = accuracy_score(y[test_idx], predicted)
        accuracies.append(acc)
    avg = np.mean(accuracies)
    #print(np.std(accuracies))
    #print(accuracies)
    return avg


def run_all(tweets,lowercase,keep_punctuation,collapse_urls, collapse_mentions, collapse_hashtags, collapse_retweets):
    
    tokens_list = [tokenize(t,lowercase,keep_punctuation,collapse_urls, collapse_mentions,collapse_hashtags,collapse_retweets)for t in tweets['text']]
    vocabulary = make_vocabulary(tokens_list)
    X = make_feature_matrix(tokens_list, vocabulary)
    y= np.array(tweets['polarity'])
    acc = do_cross_val(X, y, 5)
    #print('acc=', acc)
    return acc    

def predict_sentiment(train_tweets,test_tweets,lowercase,keep_punctuation,collapse_urls, collapse_mentions, collapse_hashtags, collapse_retweets):
	
	vectorizer = CountVectorizer(tokenizer=lambda text:tokenize(text,lowercase,keep_punctuation,collapse_urls, collapse_mentions, collapse_hashtags, collapse_retweets),min_df=1)
	train_X = vectorizer.fit_transform(train_tweets['text'])
	test_X=vectorizer.transform(test_tweets['text'])
	train_y = np.array(train_tweets['polarity'])
	clf = LogisticRegression()
	clf.fit(train_X,train_y)
	predicted = clf.predict(test_X)
	#print(predicted)
	test_tweets['ML_Sentiment'] = pd.Series(predicted, test_tweets.index)
	return test_tweets



def main():	

	afinn = read_Afinn()
	test_Afinn_tweets = read_Afinn_test_tweets('test_tweets.csv')
	test_Afinn_tweets = using_Afinn(test_Afinn_tweets,afinn)
	write_to_file(test_Afinn_tweets,'Afinn_sentiment.csv')
	#print(tweets)

	train_ML_tweets = read_train_tweets('train_tweets.csv')

	lowercase_opts = [True, False]
	keep_punctuation_opts = [True, False]	
	url_opts = [True, False]
	mention_opts = [True, False]
	Hashtags = [True,False]
	retweet = [True,False]
	
	option_iter = product(lowercase_opts,keep_punctuation_opts,url_opts,mention_opts,Hashtags,retweet)
	argnames = ['lower', 'punct','url', 'mention','Hashtags','RT']
	results = []
	for options in option_iter:		
	    acc = run_all(train_ML_tweets, *options)
	    results.append((acc, options))
	
	results = sorted(results,reverse=True)	
	
	highest_settings = {}	
	highest_accuracy =results[0]

	for i in range(len(argnames)):
		highest_settings[argnames[i]]=results[0][1][i]


	#START Predicting 
	test_ML_tweets = read_ML_test_tweets('Afinn_sentiment.csv')
	test_ML_tweets = predict_sentiment(train_ML_tweets,test_ML_tweets,highest_settings['lower'],
													highest_settings['punct'],highest_settings['url'],highest_settings['mention'],highest_settings['Hashtags'],highest_settings['RT'])
	write_to_file(test_ML_tweets,"Sentiment_Final.csv")

	with open("test_AFINN_pik", "wb") as f:
		pickle.dump(test_Afinn_tweets, f)
	with open("train_ML_pik", "wb") as f:
		pickle.dump(train_ML_tweets, f)
	with open("test_ML_pik", "wb") as f:
		pickle.dump(test_ML_tweets, f)
	with open("Accuracy_Setting_pik", "wb") as f:
		pickle.dump(highest_settings, f)
	with open("Highest_Accuracy_pik", "wb") as f:
		pickle.dump(highest_accuracy, f)	



if __name__ == '__main__':
    main()
