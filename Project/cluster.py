"""
cluster.py
"""
import networkx as nx
import matplotlib.pyplot as plt
import sys
import time
import csv
import pandas as pd
import pickle

def read_graph():
    
    """
    Returns:
      A networkx undirected graph.
    """
    return nx.read_edgelist('friends.txt', delimiter='\t')

def remove_nodes(graph,d):
	for i in graph.nodes():
		if graph.degree(i) <d:
			graph.remove_node(i)
	#print(graph.degree())
	return graph

def draw_graph(graph,filename):
	pos=nx.spring_layout(graph)
	nx.draw_networkx(graph,pos,with_labels=False,node_color='blue',node_size=50,alpha=0.50,edge_color='r')
	plt.axis('off')
	plt.savefig(filename,format="PNG",frameon=None,dpi=500)
	plt.show()   

def calculate_betweenness(graph):
	return nx.edge_betweenness_centrality(graph, normalized=False)


def get_community(graph,k):
	components= nx.number_connected_components(graph)
	while k > components:
		#print(components)		
		betweenness = sorted(sorted(calculate_betweenness(graph).items()), key=lambda x: (-x[1],x[0]))		
		#print(betweenness[0][0])
		graph.remove_edge(*betweenness[0][0])
		components= nx.number_connected_components(graph)
	return graph

def main():
	# this script takes 10 minutes to run on my computer
	graph = read_graph()
	print('Original graph has %d nodes and %d edges' %
          (graph.order(), graph.number_of_edges()))
	print("generating original graphs")
	draw_graph(graph,"original_graph.png")	
	with open("nodes_pik", "wb") as f:
		pickle.dump(graph.order(), f)

	graph = remove_nodes(graph,2)
	#print('graph has %d nodes and %d edges' %
     #     (graph.order(), graph.number_of_edges()))
	draw_graph(graph,"after_removing_edges.png")
	print("girwan newman in progress")
	graph = get_community(graph,4)
	print('Final graph has %d nodes and %d edges' %
          (graph.order(), graph.number_of_edges()))
	draw_graph(graph,"Final_Graph.png")

	with open("graph_pik", "wb") as f:
		pickle.dump(graph, f)


if __name__ == '__main__':
	main()