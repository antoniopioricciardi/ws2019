import re
import unicodedata
from typing import Dict

import networkx as nx
from networkx import Graph, find_cliques, pagerank
import plotly.graph_objs as go

import src.constants as constants


def parse_ds1():
	""""
	Parse dataset ds-1.tsv
	:return: graph version of the dataset ds-1
	"""
	print(constants.DS1_PATH.__str__())
	assert constants.DS1_PATH.exists()

	graphs = {}

	with open(constants.DS1_PATH.__str__()) as f:
		for line in f:
			line = line.strip()
			year, k1, k2, authors_dict = line.split('\t')
			year = int(year)
			k1 = unicodedata.normalize("NFKD", k1.lower()).strip()
			k2 = unicodedata.normalize("NFKD", k2.lower()).strip()
			authors_dict = eval(authors_dict)  # convert string to dictionary
			for key in authors_dict:
				assert len(key) == 40  # SHA1 length

			# Create the graph for the specific year, if missing
			graphs.setdefault(year, nx.Graph())
			graph: Graph = graphs[year]

			# Add the nodes
			graph.add_node(k1)
			graph.add_node(k2)

			# And their edge
			graph.add_edge(k1, k2, authors=authors_dict, weight=sum(authors_dict.values()))

	# TODO: Skipping the year not in the proposal interval [2000, 2018] ?
	graphs = {year: graphs[year] for year in sorted(graphs) if year in range(2000, 2019)}

	return graphs


def parse_ds2():
	"""
	Parse dataset ds-2.tsv
	:return: graph version of the dataset ds-2
	"""
	print(constants.DS2_PATH.__str__())
	assert constants.DS2_PATH.exists()

	graphs = {}

	with open(constants.DS2_PATH.__str__()) as f:
		for line in f:
			line = line.strip()
			year, a1, a2, shared_papers = line.split('\t')
			year = int(year)
			shared_papers = int(shared_papers)
			a1 = unicodedata.normalize("NFKD", a1.lower()).strip()
			a2 = unicodedata.normalize("NFKD", a2.lower()).strip()

			# Create the graph for the specific year, if missing
			graphs.setdefault(year, nx.Graph())
			graph: Graph = graphs[year]

			# Add the nodes
			graph.add_node(a1)
			graph.add_node(a2)

			previous_edge = graph.get_edge_data(a1, a2)
			if previous_edge is not None:
				# need this cover case where we have both A - B and B - A
				# sum the number of shared papers in (A,B) and (B,A)
				shared_papers += previous_edge['shared_papers']
			# And their edge
			graph.add_edge(a1, a2, shared_papers=shared_papers)  # if edge already exists this just updates edge data

	# TODO: Skipping the year not in the proposal interval [2000, 2018] ?
	# TODO: Save other years to use them to modify weights in T1.
	graphs = {year: graphs[year] for year in sorted(graphs) if year in range(2000, 2019)}

	return graphs


def plot_keywords_dist(graphs: Dict[int, Graph]):
	keywords_number = [(year, len(graph.nodes)) for year, graph in graphs.items()]
	years, keywords = zip(*keywords_number)

	fig = go.Figure()
	fig.add_trace(go.Bar(x=years,
						 y=keywords,
						 name='Keywords dist',
						 marker_color='rgb(55, 83, 109)',
						 text=keywords,
						 textposition='auto'
						 ))

	fig.update_layout(
		title='Keywords number across years',
		xaxis=dict(
			title='Year',
			titlefont_size=20,
			tickfont_size=14,
		),
		yaxis=dict(
			title='#Keywords',
			titlefont_size=20,
			tickfont_size=14,
		),
		barmode='group',
		bargap=0.15,  # gap between bars of adjacent location coordinates.
		bargroupgap=0.1  # gap between bars of the same location coordinate.
	)
	fig.show()


def get_top_topics_by_cliqueness(graphs: Graph, k: int, keyword_weights=None, weighting=True):
	"""
	Give a score to each keyword based on the number of cliques they belong to.
	If weighting is True, then weight each score with weights provided for each keyword
	:param graphs: set of graphs on which to perform the search, one for each year
	:param k: the number of topics to return
	:param keyword_weights: A dictionary of keywords weights, in the format {year: {kw: weight}}
	:param weighting: whether to perform weighting or not
	:return: A dictionary of dictionaries in the format {year: {keyword: score}}
	where {keyword: score} are keywords sorted by highest score
	"""
	if weighting:
		assert keyword_weights is not None
	results = {}
	unknown_k_re = re.compile('\\?+')
	for year, graph in graphs.items():
		graph: Graph
		cliques = list(set(x) for x in find_cliques(graph))  # find cliques
		# Get only the keywords showing up in cliques
		keywords = set(keyword for clique in cliques for keyword in clique)
		# Build the ranking dictionary
		keywords_rank = {keyword: 0 for keyword in keywords}

		'''Every found clique is represented as a networkx graph'''
		for clique in cliques:
			clique_weight = 1
			for keyword in clique:
				if unknown_k_re.match(keyword) is not None:  # ignore unknown keywords
					continue
				keywords_rank[keyword] += clique_weight
		if weighting:
			for kw in keywords_rank.keys():
				keyword_weights_year = keyword_weights.get(year)
				keyword_score = keyword_weights_year.get(kw)
				if keyword_score is None:
					keyword_score = 0.01  # forfeit value
				keywords_rank[kw] = keywords_rank[kw] * keyword_score
		results[year] = sorted(keywords_rank.items(), key=lambda x: x[1], reverse=True)[:k]  # top-k results for each year
	return results


def get_co_authorship_ranks(graphs_ds2: dict):
	"""
	Only used when we do not use pagerank, gets authors score based on the number of co-authorships they have in every
	year
	:param graphs_ds2:
	:return: dictionary of dictionaries {year: {author: rank}}
	"""
	year_co_auth = {}  # will contain co authorship
	for year, graph in graphs_ds2.items():
		graph: Graph
		auth_n = {}  # total co-authorsips of a certain author
		for author in graph.nodes:
			tot_co_auth = 0
			co_autorships_edges_list = graph.edges(author)
			for auth_1, auth_2 in co_autorships_edges_list:
				co_auth_n = graph.get_edge_data(auth_1, auth_2).get('shared_papers')  # get number of shared papers between the two authors
				tot_co_auth += co_auth_n
			auth_n[author] = tot_co_auth
		year_co_auth[year] = auth_n
	return year_co_auth


def calc_author_pagerank(graphs_ds2: dict):
	"""
	Given co-authorship graphs for each year, compute pagerank on it to obtain a score for each author
	:param graphs_ds2:
	:return: Authors pagerank for every year, as a dictionary {year: (list_of_ranks, min_rank, max_rank)}
	"""
	year_co_auth = {}
	for year, graph in graphs_ds2.items():
		graph: Graph
		auth_ranks = pagerank(graph)  # compute pagerank on the graph to obtain author ranks
		ranks_list = list(auth_ranks.values())
		# get min and max score ranks
		min_val = min(ranks_list)
		max_val = max(ranks_list)
		year_co_auth[year] = (auth_ranks, min_val, max_val)
	return year_co_auth


def get_keywords_weight(graphs_ds1: dict, graphs_ds2: dict, method='pagerank'):
	"""
	Computes keywords scores in a given year, according to one of two metrics:
		- Author Pagerank if method='pagerank' is given as a parameter <- HIGHLY RECOMMENDED
		- The score of a keyword is given by summing, for each edge linked to that word and for each author on that edge,
			the number of times an author used that word divided by the author's score. The score for an author is obtained
			by summing all the times he co-authorshipped a paper that year.
	Given the set of adjacent edges of a keyword kw_i that connect kw_i to other keywords,
	keyword score is given by the sum, for each edge,
	of the sum of the weight of each author on the edge weighted by the pagerank score of the author.
	:param graphs_ds1: Words co-occurrence graph
	:param graphs_ds2: Co-authorship graph
	:param method: which method to use to perform scoring: Author Pagerank or simply the number of publications of that author
	:return: a dictionary of dictionaries {year: {keyword: keyword_rank}}
	"""

	co_authorship_ranks = None
	if method == 'pagerank':
		co_authorship_ranks = calc_author_pagerank(graphs_ds2)
	else:
		co_authorship_ranks = get_co_authorship_ranks(graphs_ds2)
	values_per_year = {}
	for year, graph in graphs_ds1.items():
		authors_rank_year, pr_min_val, pr_max_val = co_authorship_ranks.get(year)  # get the rank of all the authors in a certain year
		graph: Graph
		nodes = graph.nodes
		keywords_rank = {node: 0 for node in nodes}
		for edge in graph.edges.items():
			keywords_value = 0  # both keywords of this edge will have this value
			kw0, kw1 = edge[0]
			'''dictionary {author_i: n} where n is the num times author_i used those keywords together'''
			edge_authors_list = edge[1]
			authors = list(edge_authors_list['authors'].keys())  # list of authors in edge

			# get all the authors for a pair of keywords (that is, an edge)
			for i in range(len(authors)):
				auth = authors[i]
				# get the number of times an author used these two keywords together
				kw_auth_weight = edge_authors_list.get('authors').get(auth)
				author_rank = authors_rank_year.get(auth)  # get all the co-authorship of author i
				# if we have no rank for that author, then assign a default value
				# (default value is the min possible author score rank in that year, if method = 'pagerank')
				if author_rank is None:
					if method == 'pagerank':
						author_rank = pr_min_val
					else:
						author_rank = 1
				if method == 'pagerank':  # report refers to this method
					keywords_value += kw_auth_weight * author_rank
				else:
					keywords_value += kw_auth_weight/author_rank
			# divide the value of each keyword by the number of authors in the current edge
			keywords_value = keywords_value/(i+1)
			# update ranks for both keywords of that edge by summing the new computed value
			keywords_rank[kw0] += keywords_value
			keywords_rank[kw1] += keywords_value
		values_per_year[year] = keywords_rank
	return values_per_year


def create_weighted_graph(old_graphs, scores):
	"""
	Given a dictionary [year]: undirected_graph,
	create a dictionary [year]: (directed_graph, graph_avg_degree) where the weight of a directed edge (n_i, n_j)
	is given by rank(n_i)/sum_incoming_ranks(n_j)
	where sum_incoming_ranks(n_j) is the sum of all the ranks of the nodes having an edge pointing towards n_j.
	Keyword ranks are first normalized in the [0.01, 1] range
	:param old_graphs: the graph to convert into a directed graph
	:param scores: keyword scores
	:return: dictionary [year]: ((directed_graph, graph_avg_degree)
	"""
	graphs_year = {}
	for year, old_graph in old_graphs.items():
		old_graph: Graph
		keywords_rank = scores.get(year)
		values_list = list(keywords_rank.values())
		max_val = max(values_list)
		min_val = min(values_list)
		avg_degree = 0
		# normalize values in [0.01, 1]
		for kw in keywords_rank.keys():
			keywords_rank[kw] = (1-0.01)*(keywords_rank[kw] - min_val) / (max_val - min_val)+0.01
			avg_degree += len(old_graph.edges(kw))
		# and compute the average degree of the graph
		avg_degree = avg_degree / len(keywords_rank.keys())

		d_graph = old_graph.to_directed()
		d_graph: nx.DiGraph
		weighted_graph = nx.DiGraph()

		for node in d_graph.nodes:
			incoming_edges = d_graph.in_edges(node)
			incoming_edges_tot_sum = 0
			for edge in incoming_edges:
				incoming_node = edge[0]
				incoming_edges_tot_sum += keywords_rank[incoming_node]
			for edge in incoming_edges:
				incoming_node = edge[0]
				weighted_graph.add_node(node)
				weighted_graph.add_node(incoming_node)
				weight = keywords_rank[incoming_node]/incoming_edges_tot_sum
				weighted_graph.add_edge(incoming_node, node, weight=weight)
		graphs_year[year] = (weighted_graph, avg_degree)
	return graphs_year


def check_if_active(graph, scores, influenced_nodes, node, in_edges):
	"""
	Check whether a node n_i can be considered active, that is the sum of the normalised weight of
	its neighbours (incoming edges)
	is larger than the score of the node.
	:param graph: a directed graph, to get edge weights
	:param scores: node scores, to check whether the node n_i has to be activated or not
	:param influenced_nodes: influenced nodes to a certain time
	:param node: the node on which to check the activation
	:param in_edges: set of incoming edges of the node
	:return: True if node activates
	"""
	node_score = scores.get(node)
	incoming_sum = 0
	for in_node, node in in_edges:
		if in_node in influenced_nodes.keys():
			incoming_sum += graph.get_edge_data(in_node, node).get('weight')
	return incoming_sum >= node_score


def calc_spreading_of_influence(graph: nx.DiGraph, starting_node, keyword_scores):
	"""
	Computes the spreading of influence algorithm over a weighted digraph, given a starting node,
	keyword scores and average degree to compute the activation function of each node
	:param graph: a weighted graph
	:param starting_node: a node where to start the spreading of influence algorithm from
	:param keyword_scores: scores for every node
	:return: a dictionary containing influenced nodes, where the key is the distance from the source
	"""

	'''to_visit will contain nodes to visit during each iteration, as keys.
	Values will be distances from source'''
	to_visit = dict()

	'''active nodes at each iteration of the algorithm (list in position 1 will have nodes activated
	during the first iteration, in position 2 during the second iteration ...)
	In position 0 there's the root'''
	influenced_nodes = dict()  # will contain active nodes.
	influenced_nodes[starting_node] = 0

	out_edges = graph.out_edges(starting_node)  # get all the outgoing edges from a node
	for edge in out_edges:  # an edge is a pair (node, adjacent_node)
		adj_node = edge[1]  # edge[1] is the node connected to curr_node
		to_visit[adj_node] = 1

	while to_visit:  # while the list of nodes to visit is not empty
		curr_node = list(to_visit.keys())[0]
		curr_iteration_num = to_visit.pop(curr_node)
		in_edges = graph.in_edges(curr_node)
		active = check_if_active(graph, keyword_scores, influenced_nodes, curr_node, in_edges)
		if active:
			# If the node is active, then save it as a key into influenced nodes. The value is its distance from the
			# starting node
			influenced_nodes[curr_node] = curr_iteration_num
			out_edges = graph.out_edges(curr_node)
			for edge in out_edges:
				other_node = edge[1]  # edge[1] is the node connected to curr_node
				if other_node not in influenced_nodes.keys() and other_node not in to_visit.keys():  # if it is not yet active
					to_visit[other_node] = curr_iteration_num + 1
	return influenced_nodes


def get_influenced_nodes_per_year(weighted_graphs, top_k_keywords, keyword_scores):
	"""
	:param weighted_graphs: 
	:param top_k_keywords: 
	:param keyword_scores: 
	:return: 
	"""
	# compute the Spreading of Influence algorithm, for every graph in every year.
	influenced_nodes = dict()  # contains influenced nodes of every year
	for year, graph_avg_deg in weighted_graphs.items():
		graph = graph_avg_deg[0]

		influencing_node = dict()
		for top_k in top_k_keywords[year]:
			starting_node = top_k[0]
			'''compute the spreading of influence for graph, given a starting node and a dictionary of scores for
			keywords in a certain year'''
			influenced_nodes_year = calc_spreading_of_influence(graph, starting_node, keyword_scores.get(year))

			'''To trace keyword generating topics over years,
			we need to distinguish a keyword with the same in different years.
			To do so, simply append the year to the keyword name (e.g: algorithm_2018)'''
			starting_node = str(year) + '_' + starting_node
			influencing_node[starting_node] = influenced_nodes_year
		influenced_nodes[year] = influencing_node  # dict of dicts {keyword_year: {nodes: visit order}}
	return influenced_nodes
