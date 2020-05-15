import os
from src.utils.paths import create_clean_folders
from src.graph import *
from src.topics import *
import matplotlib as mpl
import matplotlib.pylab as plt
import numpy as np
import src.file_writing

if __name__ == '__main__':
	constants.GRAPH_PLOTS_DIR_PATH.mkdir(exist_ok=True, parents=True)
	paths = [constants.RES_PATH, constants.TASK_1_PATH, constants.TASK_2_PATH, constants.DOTS_PATH, constants.IMAGES_PATH]
	print(constants.DOTS_PATH)
	create_clean_folders(paths)  # clean the directory of plot graph images

	# parse graphs
	graphs_ds1 = parse_ds1()
	graphs_ds2 = parse_ds2()

	unknown_k_re = re.compile('\\?+')

	# plot_keywords_dist(graphs_ds1)

	# get keyword scores
	keyword_scores = get_keywords_weight(graphs_ds1, graphs_ds2, method='pagerank')
	print("-------------------\n")

	# return top-k topics for every year
	k = 20
	# {year: {keyword: score}}
	top_results = get_top_topics_by_cliqueness(graphs_ds1, k, keyword_scores, weighting=True)

	# create a weighted graph to compute the spreading of influence algorithm on it
	weighted_graphs = create_weighted_graph(graphs_ds1, keyword_scores)

	influenced_nodes = get_influenced_nodes_per_year(weighted_graphs, top_results, keyword_scores)

	co = 0
	merged_topics_year = dict()
	topics_association_year = dict()
	merge_tracking_year = dict()
	for year, _ in weighted_graphs.items():
		influenced_nodes_year = influenced_nodes[year]
		'''merged_topics contains the actual influenced nodes (topics) merged and is in the form
		{keyword that generated topic: set of keywords that made the topic}'''
		merged_topics, topic_associations, merge_tracking = merge_topics(influenced_nodes_year)
		merged_topics_year[year] = merged_topics
		topics_association_year[year] = topic_associations
		merge_tracking_year[year] = merge_tracking

	graphs_dict = topic_tracing_year(merged_topics_year)
	graphs_dict = {k: v for k, v in sorted(graphs_dict.items(), key=lambda x: len(x[1][0]), reverse=True)}
	c = 0

	'''######### SAVE RESULTS ON FILES ##########'''
	vmin = -1
	vmax = 0
	graph_to_save_year = dict()
	for year, graph_couple in weighted_graphs.items():
		labels = dict()
		colors = []
		graph = graph_couple[0]
		influenced_nodes_year = influenced_nodes[year]
		# we are going to plot only the biggest set of influenced nodes, for each year
		influenced_nodes_year = sorted(influenced_nodes_year.items(), key=lambda x: len(x[1]), reverse=True)
		first_influenced_nodes_year = influenced_nodes_year[0]
		starting_node = first_influenced_nodes_year[0].replace(str(year) + '_', '')
		und_graph = graph.to_undirected()
		conn_components = nx.connected_components(und_graph)
		dir_graph = None
		for cc in conn_components:
			cc = und_graph.subgraph(cc)
			cc: Graph
			if cc.has_node(starting_node):
				dir_graph = cc.to_directed()
				break
		graph = dir_graph
		for node in graph.nodes:
			if node in first_influenced_nodes_year[1].keys():
				labels[node] = node
				colors.append(first_influenced_nodes_year[1][node])
			else:
				colors.append(-1)
		vmax = max(vmax, max(colors))
		graph_to_save_year[year] = (graph, starting_node, colors, labels)

	cmap = plt.get_cmap('jet', vmax-vmin)  # get a discrete colormap with vmax-vmin steps
	cmaplist = [cmap(i) for i in range(cmap.N)]
	# force the first color entry to be grey
	cmaplist[0] = (.7, .7, .7, .5)

	# create the new map
	cmap = mpl.colors.LinearSegmentedColormap.from_list(
		'Custom cmap', cmaplist, cmap.N+1)  # +1 because we need to include a colour for -1, too
	print(vmax - vmin)

	'''########## SAVE SPREADING OF INFLUENCE GRAPH ON DISK ##########'''
	for year, top_results_year in top_results.items():
		year_path = constants.TASK_1_PATH + str(year) + '/'
		create_clean_folders([year_path])

		graph, starting_node, colors, labels = graph_to_save_year[year]
		pos = nx.spring_layout(graph, k=0.9, iterations=100)  # positions for all nodes)
		graph: nx.Graph
		print(starting_node)

		ec = nx.draw_networkx_edges(graph.to_undirected(), pos, alpha=0.2)
		nc = nx.draw_networkx_nodes(graph, pos, nodelist=graph.nodes, node_color=colors,
									with_labels=True, node_size=100, cmap=cmap, vmin=vmin, vmax=vmax)
		lc = nx.draw_networkx_labels(graph, pos, labels, font_size=5)
		cb = plt.colorbar(nc)
		cb.set_label('Spreading of influence iteration (-1 if not influenced)')
		plt.axis('off')
		plt.savefig(year_path + str(starting_node), dpi=500)
		plt.close()
		influenced_path = year_path + 'spreading of influence/'
		create_clean_folders([influenced_path])
		for el in top_results_year:
			topic = el[0]
			score = el[1]
			topic_path = influenced_path + str(score) + '_' + topic + '/'
			create_clean_folders([topic_path])
			infl_nodes = influenced_nodes[year][str(year) + '_' + topic]
			f = open(topic_path + 'influenced_nodes.txt', "a")
			f.write(str(infl_nodes))
			f.close()

	'''########## SAVE TOPIC MERGES ##########'''
	for year in merged_topics_year.keys():
		year_path = constants.TASK_1_PATH + str(year) + '/'
		merging_path = year_path + 'merges/'
		create_clean_folders([merging_path])
		for keyword, merged_topics in merged_topics_year[year].items():
			keyword_path = merging_path + keyword + '/'
			create_clean_folders([keyword_path])
			f = open(keyword_path + 'merged_topics.txt', "a")
			f.write(str(merged_topics))
			f.close()

			merge_track = merge_tracking_year[year][keyword]
			f = open(keyword_path + 'merge_tracking.txt', "a")
			f.write(str(merge_track))
			f.close()

	# saving image files
	for keyword in graphs_dict.keys():
		c += 1
		graph = graphs_dict[keyword][0]
		topics = graphs_dict[keyword][1]

		print('Saving graph as png image for the topic identified by the keyword', keyword)
		graph_dot = nx.nx_pydot.to_pydot(graph)
		year = keyword[:keyword.index('_')]
		year_dots_path = constants.DOTS_PATH + year + '/'
		if not os.path.exists(year_dots_path):
			os.mkdir(year_dots_path)
		graph_dot.write(year_dots_path + keyword + '.dot')
		year_images_path = constants.IMAGES_PATH + year + '/'
		if not os.path.exists(year_images_path):
			os.mkdir(year_images_path)
		os.system('dot -n2 -Tpng ' + '\"' + year_dots_path + keyword + '.dot' + '\"' + '>' + '\"' + year_images_path
				+ keyword + '.png' + '\"')

	print('###############')
	'''##### BIGGEST YEAR MERGES #####'''
	remarkable_merges_path = constants.TASK_2_PATH + 'remarkable_merges/'
	create_clean_folders([remarkable_merges_path])
	c = 0
	for keyword in graphs_dict.keys():
		graph = graphs_dict[keyword][0]
		graph_dot = nx.nx_pydot.to_pydot(graph)
		merged_topics = graphs_dict[keyword][1]
		keyword_path = remarkable_merges_path + str(c) + '.' + keyword + '/'
		create_clean_folders([keyword_path])
		print(keyword)
		graph_dot.write(keyword_path + 'tracing.dot')
		os.system('dot -n2 -Tpng ' + '\"' + keyword_path + 'tracing.dot' + '\"' + '>' + '\"' + keyword_path + 'tracing.png' + '\"')
		f = open(keyword_path + 'merged_topics.txt', "a")
		f.write(str(merged_topics))
		f.close()
		c += 1
		if c == 5:
			break
		# print(f'{len(graph.nodes)} keywords for the year {year}:\t{graph.nodes}\n')

