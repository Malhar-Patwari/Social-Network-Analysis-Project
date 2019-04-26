"""
sumarize.py
"""
import pickle
import networkx as nx
import matplotlib.pyplot as plt
import sys
import time
import csv
import pandas as pd
import pickle


def count_Afinn(x):
	if x <0:
		return -1
	elif x==0:
		return 0
	else:
		return 1		


def main():
	file = open('summary.txt', 'w')

	with open("nodes_pik", "rb") as f:
		n = pickle.load(f)
	
	file.write("***********************************************************************CLUSTERING********************************************************************************************\n")
	file.write("Number of users collected: %d\n" %n)
	
	with open("graph_pik", "rb") as f:
		graph = pickle.load(f)

	sub_graphs = nx.connected_component_subgraphs(graph)
	sub_graphs = [i for i in sub_graphs]

	file.write("\nnumber of communities discovered: %d \n" %len(sub_graphs))
	
	for i in range(len(sub_graphs)):
		file.write("Subgraph: %d consists of %d\n" %(i,len(sub_graphs[i].nodes())))

	average = 	len(graph.nodes())/len(sub_graphs)	
	file.write("\nAverage Number of users per community %f\n"%average)


	file.write("\n**************************************************************SENTIMENT ANALYSIS Using AFINN****************************************************************\n")


	with open("test_AFINN_pik", "rb") as f:
		test_Afinn_tweets = pickle.load(f)

	test_Afinn_tweets['Afinn_polarity'] = test_Afinn_tweets['Afinn_sentiment'].apply(count_Afinn)	

	file.write("Number of tweets collected to apply AFINN : %d\n" %len(test_Afinn_tweets))

	Afinn_count_classes = test_Afinn_tweets.groupby(['Afinn_polarity']).count()	
	#file.write(Afinn_count_classes)
	file.write("Number of instances per class Afinn predicted for testing dataset:\t Negative tweets: %d \t neutral tweets: %d \t positive tweets: %d \n" %(Afinn_count_classes.text[-1],Afinn_count_classes.text[0],Afinn_count_classes.text[1]))

	file.write("\n--------------------------------------------EXAMPLES-----------------------------------\n")	
	file.write("These are the Examples of Negative class , Neutral class, Positive class respectively: %s \n\n" %test_Afinn_tweets.groupby(['Afinn_polarity']).text.head(1))	


	file.write("\n************************************************SENTIMENT ANALYSIS Using Logistic Regression Algorithm********************************************************\n")
	

	with open("train_ML_pik", "rb") as f:
		train_ML_tweets = pickle.load(f)
	
	file.write("Number of tweets collected to train the model: %d\n" %len(train_ML_tweets)) #Hard coded because if I run script to collect train data, I will loose my manually defined class labels
	with open("test_ML_pik", "rb") as f:
		test_ML_tweets = pickle.load(f)

	file.write("Number of tweets collected to predict class labels: %d\n" %len(test_ML_tweets))	
    
	count_classes = train_ML_tweets.groupby(['polarity']).count()	
	file.write("Number of instances per class for training dataset:\t class -1: %d \t class 0: %d \t class 1: %d \n" %(count_classes.text[-1],count_classes.text[0],count_classes.text[1]))

	with open("Accuracy_Setting_pik", "rb") as f:
		highest_accuracy_settings = pickle.load(f)

	with open("Highest_Accuracy_pik", "rb") as f:
		highest_accuracy = pickle.load(f)

	file.write("Setting which is having highest accuracy on training dataset: %s\n" %highest_accuracy_settings)
	file.write("Highest Cross Valdation Accuracy achieved by this setting : %s \n" %highest_accuracy[0])
	
	file.write("By applying these settings to test dataset: \n")	
	count_classes = test_ML_tweets.groupby(['ML_Sentiment']).count()
	file.write("Predicted Number of instances per class for testing dataset:\t class -1: %d \t class 0: %d \t class 1: %d" %(count_classes.text[-1],count_classes.text[0],count_classes.text[1]))

	#Examples
	file.write("\n--------------------------------------------EXAMPLES-----------------------------------\n")	
	file.write("These are the Examples of class -1, class 0, class 1 respectively:\n")
	file.write("%s \n\n" %test_ML_tweets.groupby(['ML_Sentiment']).text.head(1))
	

	file.write("\n******************************************************************************************************************************\n")
	neg=0
	pos=0
	neu=0
	for i in (test_ML_tweets.index):		
		if(test_ML_tweets.Afinn[i]==0 and test_ML_tweets.ML_Sentiment[i] ==0):						
			neu+=1			
		if(test_ML_tweets.Afinn[i]<0 and test_ML_tweets.ML_Sentiment[i] <0):
			neg+=1			
		if(test_ML_tweets.Afinn[i]>0 and test_ML_tweets.ML_Sentiment[i] >0):
			pos+=1			
	file.write("\nNumber of tweets classified as positive by both algorithms:%d\n"%pos)
	file.write("\nNumber of tweets classified as Negative by both algorithms:%d\n"%neg)
	file.write("\nNumber of tweets classified as Neutral by both algorithms:%d\n"%neu)
	file.write("\nNumber of tweets classified same by both algorithms: %d\n"%(pos+neg+neu))
	print("Summary printed to file summary.txt")
	file.close()

if __name__ == '__main__':
	main()