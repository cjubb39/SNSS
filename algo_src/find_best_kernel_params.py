import operator
import sys
from random import sample
import networkx as nx
import numpy as np
from sklearn import svm, cross_validation, grid_search
from math import ceil
import scipy
from collections import defaultdict
import pickle
# import matplotlib.pyplot as plt

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
    random_input_data_name = sys.argv[2]
    params_out_name = sys.argv[3]

    retweet_graph = nx.DiGraph()
    nx.read_weighted_edgelist('data/higgs-retweet_network.edgelist', create_using=retweet_graph)
    reply_graph = nx.DiGraph()
    nx.read_weighted_edgelist('data/higgs-reply_network.edgelist', create_using=reply_graph)
    mention_graph = nx.DiGraph()
    nx.read_weighted_edgelist('data/higgs-mention_network.edgelist', create_using=mention_graph)

    graphs = [retweet_graph, reply_graph, mention_graph]

    # load random N
    random_n = []
    with open(random_input_data_name, 'r') as f:
        for line in f:
            random_n.append(line.split())

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
    training_set = sample(random_n, int(ceil(int(known_input) * 0.2)))
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

    # perform cross-validation to identify best params
    sss = cross_validation.LeavePOut(len(training_Y), p=int(float(len(training_Y)) * .4))
    best_params_histogram = {
        'linear': defaultdict(list),
        'rbf': defaultdict(list),
        'poly': defaultdict(list)
    }
    num_iterations = 0
    max_iterations = 100
    for train_index, test_index in sss:
        if num_iterations > max_iterations:
            break
        train_features = training_X[train_index]
        train_labels = training_Y[train_index]
        test_features = training_X[test_index]
        test_labels = training_Y[test_index]

        # cross validation
        svr = svm.SVR()
        param_grid = [
          {'C': [.01, .1, 1, 10, 100,1000,10000], 'kernel': ['linear']},
          {'C': [.01, .1, 1, 10, 100,1000,10000], 'gamma': [1, .1, .01, 0.001, 0.0001, .00001], 'kernel': ['rbf']},
          {'degree': [2, 3, 4, 5], 'kernel': ['poly']}
        ]

        for kernel in param_grid:
            clf = grid_search.GridSearchCV(svr, [kernel], scoring='mean_squared_error', iid=True)
            clf.fit(train_features, train_labels)
            for params, mean_score, scores in clf.grid_scores_:
                best_params_histogram[params['kernel']][(params['C'] if 'C' in params else None, params['gamma'] if 'gamma' in params else None, params['degree'] if 'degree' in params else None)].extend(scores)

        num_iterations += 1
        # print "Detailed regression report:"
        # y_true, y_pred = test_labels, clf.predict(test_features)
        # print metrics.mean_squared_error(y_true, y_pred)

    best_params = {}
    for kernel in best_params_histogram:
        temp_params_tuple = max(best_params_histogram[kernel].iteritems(), key=lambda x: np.mean(x[1]))[0]
        best_params[kernel] = {}
        if temp_params_tuple[0] != None: best_params[kernel]['C'] = temp_params_tuple[0]
        if temp_params_tuple[1] != None: best_params[kernel]['gamma'] = temp_params_tuple[1]
        if temp_params_tuple[2] != None: best_params[kernel]['degree'] = temp_params_tuple[2]

    pickle.dump(best_params, open(params_out_name, 'w+'))
