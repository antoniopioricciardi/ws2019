from src.file_writing import *

if __name__ == '__main__':
	k = int(input("How many topics per year would you like to generate? Please insert an integer\n"))
	assert isinstance(k, int)

	similarity_score = 0.7  # threshold for merging (Task 1)
	tracing_similarity_score = 0.6  # threshold for merging over years (Task 2)

	# 	'''prepare environment variables'''
	constants.GRAPH_PLOTS_DIR_PATH.mkdir(exist_ok=True, parents=True)
	paths = [constants.RES_PATH, constants.TASK_1_PATH, constants.TASK_2_PATH, constants.DOTS_PATH, constants.IMAGES_PATH]
	create_clean_folders(paths)  # clean the directory from old plot graph images

	'''parse graphs'''
	graphs_ds1 = parse_ds1()
	graphs_ds2 = parse_ds2()

	print('##########')
	'''get keyword scores'''
	print("Computing keywords weight")
	keyword_scores = get_keywords_weight(graphs_ds1, graphs_ds2, method='pagerank')
	print("Done\n")

	print('##########')
	print('Computing top {} keywords'.format(k))
	'''return top-k topics for every year'''
	'''top_results is in the format {year: {keyword: score}}'''
	top_results = get_top_topics_by_cliqueness(graphs_ds1, k, keyword_scores, weighting=True)
	print("Done")

	# create a weighted graph to compute the spreading of influence algorithm on it
	weighted_graphs = create_weighted_graph(graphs_ds1, keyword_scores)

	print('##########')
	print('Getting topics')
	influenced_nodes = get_influenced_nodes_per_year(weighted_graphs, top_results, keyword_scores)
	print('Done')
	co = 0
	merged_topics_year = dict()
	topics_association_year = dict()
	merge_tracking_year = dict()

	print('##########')
	print('Merging topic for every year')
	for year, _ in weighted_graphs.items():
		influenced_nodes_year = influenced_nodes[year]
		'''merged_topics contains the actual influenced nodes (topics) merged and is in the form
		{keyword that generated topic: set of keywords that made the topic}'''
		merged_topics, topic_associations, merge_tracking = merge_topics(influenced_nodes_year, similarity_score)
		merged_topics_year[year] = merged_topics
		topics_association_year[year] = topic_associations
		merge_tracking_year[year] = merge_tracking
	print('Done')


	print('##########')
	print('Performing topic tracing and merging over years')
	graphs_dict = topic_tracing_year(merged_topics_year, tracing_similarity_score)
	graphs_dict = {k: v for k, v in sorted(graphs_dict.items(), key=lambda x: len(x[1][0]), reverse=True)}
	c = 0
	print('Done')

	print('##########')
	print('Saving results in directories')
	'''######### SAVE RESULTS ON FILES ##########'''
	save_spreading_of_influence(weighted_graphs, influenced_nodes, top_results)

	'''########## SAVE TOPIC MERGES ##########'''
	save_topic_merges(merged_topics_year, merge_tracking_year)

	'''########## SAVE TOPIC TRACING ##########'''
	save_tracing_graphs(graphs_dict)

	print('##############')
	'''##### SAVE TOP N MERGES #####'''
	save_remarkable_merges(graphs_dict, 10)
	# print(f'{len(graph.nodes)} keywords for the year {year}:\t{graph.nodes}\n')
	print('All Done!')


