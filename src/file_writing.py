import os
from src.utils.paths import create_clean_folders
from src.graph import *
from src.topics import *
import matplotlib as mpl
import matplotlib.pylab as plt


def save_spreading_of_influence(weighted_graphs, influenced_nodes, top_results):
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

    cmap = plt.get_cmap('jet', vmax - vmin)  # get a discrete colormap with vmax-vmin steps
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # force the first color entry to be grey
    cmaplist[0] = (.7, .7, .7, .5)

    # create the new map
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'Custom cmap', cmaplist, cmap.N + 1)  # +1 because we need to include a colour for -1, too

    '''########## SAVE SPREADING OF INFLUENCE GRAPH ON DISK ##########'''
    print('writing in {} for every year. . .'.format(constants.TASK_1_PATH))
    for year, top_results_year in top_results.items():
        year_path = constants.TASK_1_PATH + str(year) + '/'
        create_clean_folders([year_path])

        graph, starting_node, colors, labels = graph_to_save_year[year]
        pos = nx.spring_layout(graph, k=0.9, iterations=100)  # positions for all nodes)
        graph: nx.Graph

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


def save_topic_merges(merged_topics_year, merge_tracking_year):
    merged_topics_year = {k: v for k, v in sorted(merged_topics_year.items(), key=lambda x: len(x[1]), reverse=True)}
    for year in merged_topics_year.keys():
        year_path = constants.TASK_1_PATH + str(year) + '/'
        merging_path = year_path + 'merges/'
        create_clean_folders([merging_path])
        c = 1
        for keyword, merged_topics in merged_topics_year[year].items():
            keyword_path = merging_path + str(c) + '.' + keyword + '/'
            create_clean_folders([keyword_path])
            f = open(keyword_path + 'merged_topics.txt', "a")
            f.write(str(merged_topics))
            f.close()

            merge_track = merge_tracking_year[year][keyword]
            f = open(keyword_path + 'merge_tracking.txt', "a")
            f.write(str(merge_track))
            f.close()
            c += 1

def save_tracing_graphs(graphs_dict):
    print('writing in {}. . .'.format(constants.TASK_2_PATH))
    for keyword in graphs_dict.keys():
        graph = graphs_dict[keyword][0]
        topics = graphs_dict[keyword][1]
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
        f = open(year_images_path + keyword + '.txt', "a")
        f.write(str(topics))
        f.close()


def save_remarkable_merges(graphs_dict, num):
    remarkable_merges_path = constants.TASK_2_PATH + 'remarkable_merges/'
    create_clean_folders([remarkable_merges_path])
    c = 1
    for keyword in graphs_dict.keys():
        graph = graphs_dict[keyword][0]
        graph_dot = nx.nx_pydot.to_pydot(graph)
        merged_topics = graphs_dict[keyword][1]
        keyword_path = remarkable_merges_path + str(c) + '.' + keyword + '/'
        create_clean_folders([keyword_path])
        graph_dot.write(keyword_path + 'tracing.dot')
        os.system(
            'dot -n2 -Tpng ' + '\"' + keyword_path + 'tracing.dot' + '\"' + '>' + '\"' + keyword_path + 'tracing.png' + '\"')
        f = open(keyword_path + 'merged_topics.txt', "a")
        f.write(str(merged_topics))
        f.close()
        if c == num:
            break
        c += 1
