import operator
import sys
from random import sample
import networkx as nx
import numpy as np
from sklearn import svm, cross_validation, grid_search, metrics
from math import ceil
import scipy
from collections import defaultdict
import pickle

def hits(G,max_iter=100,tol=1.0e-6):
    M=nx.adjacency_matrix(G,nodelist=G.nodes())
    (n,m)=M.shape # should be square
    A=M.T*M # authority matrix
    x=scipy.ones((n,1))/n  # initial guess
    # power iteration on authority matrix
    i=0
    while True:
        xlast=x
        x=A*x
        x=x/x.sum()
        # check convergence, l1 norm
        err=scipy.absolute(x-xlast).sum()
        if err < tol:
            break
        if i>max_iter:
            raise NetworkXError(\
            "HITS: power iteration failed to converge in %d iterations."%(i+1))
        i+=1

    a=np.asarray(x).flatten()
    h=np.asarray(M*a).flatten()
    hubs=dict(zip(G.nodes(),h/h.sum()))
    authorities=dict(zip(G.nodes(),a/a.sum()))
    return hubs,authorities

def calc_ratio(G, node):
    in_count = 0.0
    out_count = 0.0
    for u,v,d in G.in_edges_iter(node, data=True):
        in_count += d['weight']

    for u,v,d in G.out_edges_iter(node, data=True):
        out_count += d['weight']

    return in_count / out_count if out_count != 0.0 else 0.0

def normalize_feature(feature_dict):
    mean = np.mean(feature_dict.values())
    std_dev = np.std(feature_dict.values())
    return {k: (v - mean) / std_dev for k, v in feature_dict.iteritems()}

if __name__ == "__main__":
    known_input = sys.argv[1]
    goal_input = sys.argv[2]
    kernel = sys.argv[3]
    best_params = pickle.load(open('output/params', 'r'))

    retweet_graph = nx.DiGraph()
    nx.read_weighted_edgelist('data/higgs-retweet_network.edgelist', create_using=retweet_graph)
    reply_graph = nx.DiGraph()
    nx.read_weighted_edgelist('data/higgs-reply_network.edgelist', create_using=reply_graph)
    mention_graph = nx.DiGraph()
    nx.read_weighted_edgelist('data/higgs-mention_network.edgelist', create_using=mention_graph)

    graphs = [retweet_graph, reply_graph, mention_graph]

    # load random N
    training_set = []
    with open('output/random_' + known_input + '_nodes', 'r') as f:
        for line in f:
            training_set.append(line.split())

    # calculate h/a for each graph
    features = []
    for g in graphs:
        hubs, authorities = hits(g)
        hubs = defaultdict(lambda: 0.0, hubs)
        authorities = defaultdict(lambda: 0.0, authorities)
        # features.append(hubs)
        features.append(authorities)

    # normalize feature data
    features = [normalize_feature(x) for x in features]

    # get set of unique nodes in all graphs
    training_nodes = set(map(lambda x: x[0], training_set))
    # sample "guess" nodes from social data
    all_nodes = set(retweet_graph.nodes()) | set(reply_graph.nodes()) | set(mention_graph.nodes())
    testing_nodes = all_nodes - training_nodes

    # convert features to dictionary
    features = {node:map(lambda f: f[node] if node in f else 0.0, features) for node in all_nodes}

    # populate training
    training_X = np.empty([len(training_nodes), len(features.values()[0])])
    training_Y = np.empty(len(training_nodes))
    empty_features = [0.0] * len(features.values()[0])
    for index, n_v_tuple in enumerate(training_set):
        node = n_v_tuple[0]
        value = n_v_tuple[1]
        training_X[index] = features[node] if node in features else empty_features
        training_Y[index] = value

    # populate testing
    X = np.zeros([len(testing_nodes), len(features.values()[0])])
    ordered_test_nodes = [None] * len(testing_nodes)
    for index, node in enumerate(testing_nodes):
        X[index] = features[node]
        ordered_test_nodes[index] = node

    clf = svm.SVR(**(best_params[kernel]))
    clf.fit(training_X, training_Y)
    predictions = clf.predict(X)

    # sort by predicted edges score
    with open('output/calculated_top_' + goal_input + '_' + kernel, 'w+') as f:
        count = 0
        for i,p in sorted(enumerate(predictions), key=operator.itemgetter(1), reverse=True):
            if count >= int(goal_input):
                break
            f.write(str(ordered_test_nodes[i]))
            f.write(' ')
            f.write(str(p))
            f.write("\n")
            count += 1
