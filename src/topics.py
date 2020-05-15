from networkx import DiGraph


def merge_topics(influenced_keywords_list, similarity_score):
    """
    Merge groups of similar topics into one.
    :param influenced_keywords_list: a list of topics
    :param similarity_score: similarity that two topics must have in order to merge them
    :return: (merged topics - dictionary{keyword: merged_topics}. The keyword is the keyword containing the bigger topic
    when merging
    · topic_associations - {keyword_i: (keyword_j, score)}. This tells that keyword_i has been merged to keyword_j with that score
    we use this to decide whether to skip a topic merge or to de-merge and re-merge topics in favour of those with higher similarity

    · merge tracking - Will be used to perform the opposite of topic_associations. This will contain a list of topics
    merged to the key
    """

    '''
    sort the topic list in reverse order, so that we can start including topics into bigger ones
    we check from biggest to smallest topics so that if
    given topics a, b, c  with similarity(a,b) == similarity(b,c), we have already matched b to a if a is bigger
    then we're okay.
    '''
    influenced_keywords_list = {keyword: v for keyword, v in sorted(influenced_keywords_list.items(), key=lambda item: len(item[1]), reverse=True)}
    # key j is a keyword, value (i, s) is the keyword i to which j is merged, with score s
    topic_associations = {keyword: None for keyword in influenced_keywords_list.keys()}
    merge_tracking = {keyword: [] for keyword in influenced_keywords_list.keys()}
    already_merged = []  # list of nodes that don't need to be checked anymore (already matched)
    for i, keyword_i in enumerate(influenced_keywords_list):
        if keyword_i in already_merged:  # if the keyword generating the topic has already been inserted
            continue
        kw_set_i = set(influenced_keywords_list[keyword_i].keys())

        for j, keyword_j in enumerate(influenced_keywords_list):
            if j <= i:  # no need to compare same topics twice
                continue
            kw_set_j = set(influenced_keywords_list[keyword_j].keys())
            intersection = kw_set_i & kw_set_j
            if len(kw_set_j) <= len(kw_set_i):  # check whether at least 70% of the smaller set is into the other
                score = len(intersection) / len(kw_set_j)
                '''We don't merge immediately, because we want to check whether original topic contains another,
                and not merged (thus bigger) topic contains another. Merging would be too easy!
                For now simply save associations, saying which topic merges to which.'''
                if score >= similarity_score:
                    # if j has never been merged to another topic, then we just merge
                    if topic_associations[keyword_j] == None:
                        topic_associations[keyword_j] = (keyword_i, score)
                        already_merged.append(keyword_i)
                        already_merged.append(keyword_j)
                    else:
                        # if j belongs to an already merged topic but with smaller score,
                        # then forget that merge and replace it with a merge to to the new i
                        if topic_associations[keyword_j][1] < score:
                            # remove the keyword that will be merged to a new topic, from merge_tracking
                            merge_tracking[topic_associations[keyword_j][0]].remove(keyword_j)
                            # and merge it to the new topic
                            topic_associations[keyword_j] = (keyword_i, score)
                            already_merged.append(keyword_i)
                            already_merged.append(keyword_j)
                    merge_tracking[keyword_i].append(keyword_j)
    '''Finally, perform merging'''
    merged_topics = dict()
    for keyword in topic_associations.keys():
        merged_to = topic_associations[keyword]
        if merged_to is not None:
            merged_to = merged_to[0]  # remember: topic_associations[keyword] = (another keyword: score)
            merge = set(influenced_keywords_list[keyword].keys()) | set(influenced_keywords_list[merged_to].keys())
            if merged_topics.get(merged_to) is None:
                merged_topics[merged_to] = merge
            else:
                merged_topics[merged_to] = merged_topics[merged_to] | merge
        else:
            # unmerged topics are still topics, so add them to the dictionary
            if keyword not in merged_topics.keys():
                merged_topics[keyword] = set(influenced_keywords_list[keyword].keys())
    return merged_topics, topic_associations, merge_tracking


def topic_tracing_year(merged_topics_year, similarity_score):
    """
    Trace topics over years and merge them if they are similar enough.
    Given a topic, to trace it means to follow its evolution over the years.
    :param merged_topics_year: A dictionary of topics for each year
    :param similarity_score: Similarity threshold for merging
    :return: a dictionary with a keyword as key, that contains tuples (graph, topic), where graph is the graph created
            by tracing a topic over the years and topic is the list of keywords merged in the tracing process.
    """

    '''
    NOTE: The structure of topic_dictionary is keyword: topic.
    Meaning that a topic (that is the set of words representing that topic) is identified by a keyword, that is the 
    keyword that generated that topic and has had smaller topics merged in previous steps.
    '''

    graphs_dict = dict()
    '''
    when a keyword is added to a certain graph, we will add it to this dictionary as a key. 
    The value of that key will be the root of that graph
    '''
    merged_nodes_root = dict()

    '''topics_dict is a dictionary of topics for a given year, with a keyword as key and the topic as value'''
    for year, topics_dict in merged_topics_year.items():
        next_year_topics_dict = dict()
        '''We are going to analyse topic by consecutive years'''
        if year < 2018:
            next_year_topics_dict = merged_topics_year[year+1]
        for topic_keyword in topics_dict.keys():
            '''
            If the keyword has not been merged, create a graph with keyword as a node,
            to track merged topics in that branch and add the keyword as the first element (root) of merged_nodes_root
            '''
            if topic_keyword not in merged_nodes_root.keys():
                merged_nodes_root[topic_keyword] = [topic_keyword]
                topic_tracing_graph = DiGraph()
                topic_tracing_graph.add_node(topic_keyword)
                # add the tuple (tracing_graph, topic generated by that graph) to the final result
                graphs_dict[topic_keyword] = (topic_tracing_graph, topics_dict[topic_keyword])
            '''Given a topic node, we are going to analyse all of the graphs containing that node'''
            if year == 2018:
                continue
            for keyword_root in merged_nodes_root[topic_keyword]:
                topic = graphs_dict[keyword_root][1]
                for next_year_topic_keyword in next_year_topics_dict.keys():
                    next_topic = next_year_topics_dict[next_year_topic_keyword]
                    intersection = topic & next_topic
                    if len(topic) < len(next_topic):
                        score = len(intersection) / len(topic)
                    else:
                        score = len(intersection) / len(next_topic)
                    if score > similarity_score:
                        '''
                        update the tuple (graph, merged_topic) in graphs_dict with the new tuple
                        where we added a node/edge and merged a new topic
                        '''
                        graph = graphs_dict[keyword_root][0]
                        graph.add_node(next_year_topic_keyword)
                        graph.add_edge(topic_keyword, next_year_topic_keyword)
                        # merged_topic = graphs_dict[keyword_root][1]
                        merged_topic = topic | next_topic
                        graphs_dict[keyword_root] = (graph, merged_topic)
                        if merged_nodes_root.get(next_year_topic_keyword) is None:
                            merged_nodes_root[next_year_topic_keyword] = []
                        merged_nodes_root[next_year_topic_keyword].append(keyword_root)
    return graphs_dict
