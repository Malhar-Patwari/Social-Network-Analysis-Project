DESCRIPTION

Overview: In this project, I tried to collect David Cameron's Friends and his friends of friends. I created graph between the users who are connected
on Twitter. Applied Girwan newman algorithm to get communities in the graph. Then I Collected Tweets about Donald Trump and applied sentiment analysis on it using
both AFINN and Logistic regression approach.

The below listed is the file wise description of the project.

1) candidates.txt:- 

	Name of the user list. (Here only David Cameron's screenname (David_Cameron))

2)Data Collection(collect.py)-

	This file takes name form the candidates file. Then It takes first 15 friends of David Cameron and get 200 friends of all his 15 friends and put it into "friends.txt".
	I chose these numbers to avoid twitter rate limits. Then the script takes 100 tweets which has word "trump" in it. It dumped those tweets into "test_tweets.csv"

3)Clustering (cluster.py):-

	The script generate the graph from "friends.txt" file. Then it removes the nodes which has degree less then 2. It applies GirwanNewman algorithm to create 5
	communities in the graph. That graph is saved in "Final_Graph.png" file. The graph is stored in pickle object.

4)get training dataset (get_train_tweets.py):- (Do not run as it will overwrite the classified tweets)
	To train my logistic Regression model, I used this script to get 450 tweets about Trump. I gathered these tweets from 5 democratic states and 5 republican states.
	So that training dataset does not bias towards positive or negative tweets. I dumped these tweets into "train_tweets.csv" and manually classified it into -1,0,1. where 1 being positive, 0 being neutral and -1 being negative sentiment towards Trump.

5)Classification(classify.py):-

	For classification, I used 2 approaches,
	i)AFINN:- Used AFINN to classify tweets in "test_tweets.csv" file and dumped it into the "Afinn_sentiment.csv".
	ii) Logistic Regression:-  I took tweets from "train_tweets.csv" applied model on the tweets. I permuted different settings to filter tweets and then applied 							 model on every permutations of settings. I got best setting(77% accuracy) and applied on unseen "Afinn_sentiment.csv" tweets. Then 						   dumped the Final output in the "Sentiment_Final.csv".
							   Dumped final "Sentiment_Final.csv" file into pickle object.
	
6) summary (summary.py):- 
	
	Reads all the output files of above scripts and creates summary.txt which has all the required analysis.	

	
Conclusions-

-For clustering, I used David Cameron's friends and his friends of friends. I removed 1 degree nodes(users) just to make graph dense (and discovering communities 
 on graph with 1 degree is taking long time). SO the users in the graph are connected with minimum of two other users. Applying girwan newman on this dense graph will give connected communities. (You can comment line#59 in "cluster.py" if want to perform girwanNewman on original graph). These communities are other politicians having their own community with important people.

-AFINN just weigh on words.It does not see the conext of the text. While Machine learning does consider sentence structure and tries to predict sentiment based 
 on training data. So machine learning is better approach then AFINN. So I used both approach to compare both algorithms and find similarity between them. I used
 best setting by crossvalidation accuracy, and applied on previously unseen data. I got 77% (best model) crosvalidation accuracy. I printed the differences of both algorithms in summary.txt file. There were many sarcasms and ironies in the training dataset, so machine learning could not give accurate prediction as I expected.