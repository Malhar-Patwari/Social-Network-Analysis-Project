"""
collect.py
"""
import sys
import time
from TwitterAPI import TwitterAPI
import networkx as nx
import csv
import re
import pandas as pd

consumer_key = ''
consumer_secret = ''
access_token = ''
access_token_secret = ''


def get_twitter():
    """ 
      An instance of TwitterAPI.
    """
    return TwitterAPI(consumer_key, consumer_secret, access_token, access_token_secret)

def read_screen_names(filename):
    
    fname = open(filename)
    rows = fname.read()        
    return rows.split('\n')

def robust_request(twitter, resource, params, max_tries=5):
    
    for i in range(max_tries):
        request = twitter.request(resource, params)
        #print(request.status_code)        
        if request.status_code == 200:
            return request
        else:
        	print('Got error %s \nsleeping for 15 minutes.' % request.text)
        	sys.stderr.flush()
        	time.sleep(61 * 15)    

    
def get_friends(twitter, screen_name):

    friends = robust_request(twitter,"friends/list",{'screen_name':screen_name,'count':200,'cursor':-1},5)    
    names = []
    for r in friends:
    	for i in r['users']:
    		names.append(i["screen_name"])
    
    #print(len(names))
    #print(names)    
    return names

def get_names(twitter,ids):
	user_objects = robust_request(twitter,"users/lookup",{'ids':ids},5)
	#print(type(names))
	names = [r['screen_name'] for r in user_objects]
	#print(names)
	return names

def get_tweets(twitter):
    tweets = robust_request(twitter,'search/tweets', {'q': 'Trump','lang':'en','count':100},5)          
    f = open('test_tweets.csv', 'w',encoding='utf-8')
    #f.write(columnTitleRow)

    for i in tweets:
        string = i['text']                
        string = re.sub(',', '', string)
        string = re.sub('\n', '', string)
        #f.write(string+'\n')
        #print(string)        
        user = i['user']['screen_name']
        row = user + "," + string + "\n"
        f.write(row)
        #print(i)
        #print(string)
    print("Tweets written on file test_tweets.csv")    
    


    
def get_friends_ids(twitter,screen_names):
	frnd_dict ={}
	for i in screen_names:
		name = i
		#print(i)
		list_friends =[]
		list_friends = get_friends(twitter,i)
		list_friends = sorted(list_friends)
		frnd_dict[i]=list_friends
		for j in list_friends[:14]:
			#print(j)
			list_fof=[]
			list_fof = get_friends(twitter,j)
			frnd_dict[j]=list_fof
	#print(frnd_dict)		
	return frnd_dict


def write_name(users_friends):
	f = open('friends.txt', 'w')	
	for i in users_friends:		
		for j in users_friends[i]:					
			f.write(i+"\t"+j+"\n")
	print("Friends names are written on file named friends.txt")		



def main():
    
    #This script will take approx 5 seconds to run
    twitter = get_twitter()    
    screen_names = read_screen_names('candidates.txt')  
    print('Established Twitter connection.')
    print('Read screen names: %s' % screen_names)
    print("getting friends")
    users_friends = get_friends_ids(twitter, screen_names)    
    write_name(users_friends)

    get_tweets(twitter)

if __name__ == '__main__':
    main()
