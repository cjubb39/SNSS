import operator
import sys
from random import sample
import networkx as nx
import numpy as np
from sklearn.svm import SVR
from math import ceil
import scipy
from collections import defaultdict

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

if __name__ == "__main__":
    n_input = sys.argv[1]

    retweet_graph = nx.DiGraph()
    nx.read_weighted_edgelist('data/higgs-retweet_network.edgelist', create_using=retweet_graph)
    reply_graph = nx.DiGraph()
    nx.read_weighted_edgelist('data/higgs-reply_network.edgelist', create_using=reply_graph)
    mention_graph = nx.DiGraph()
    nx.read_weighted_edgelist('data/higgs-mention_network.edgelist', create_using=mention_graph)

    # load top N
    validation_top_n = []
    with open('output/top_' + n_input + '_nodes', 'r') as f:
        for line in f:
            validation_top_n.append(line.split())

    # load random N
    random_n = []
    with open('output/random_' + n_input + '_nodes', 'r') as f:
        for line in f:
            random_n.append(line.split())

    # calculate h/a for each graph
    rt_h, rt_a = hits(retweet_graph)
    rt_h = defaultdict(lambda: 0.0, rt_h)
    rt_a = defaultdict(lambda: 0.0, rt_a)
    rep_h, rep_a = hits(reply_graph)
    rep_h = defaultdict(lambda: 0.0, rep_h)
    rep_a = defaultdict(lambda: 0.0, rep_a)
    mention_h, mention_a = hits(mention_graph)
    mention_h = defaultdict(lambda: 0.0, mention_h)
    mention_a = defaultdict(lambda: 0.0, mention_a)

    # get set of unique nodes in all graphs
    training_set = sample(random_n, int(ceil(int(n_input) * 0.2)))
    training_nodes = set(map(lambda x: x[0], training_set))
    # sample "guess" nodes from social data
    testing_nodes = set(retweet_graph.nodes()) | set(reply_graph.nodes()) | set(mention_graph.nodes()) - training_nodes

    # populate training
    training_X = np.empty([len(training_nodes), 6])
    training_Y = np.empty(len(training_nodes))
    for index, n_v_tuple in enumerate(training_set):
        node = n_v_tuple[0]
        value = n_v_tuple[1]
        training_X[index] = [rt_h[node], rt_a[node], rep_h[node], rep_a[node], mention_h[node], mention_a[node]]
        training_Y[index] = value

    clf = SVR(C=1.0, epsilon=0.2)
    clf.fit(training_X, training_Y)

    # populate testing
    X = np.zeros([len(testing_nodes), 6])
    ordered_test_nodes = [None] * len(testing_nodes)
    for index, node in enumerate(testing_nodes):
        X[index] = [rt_h[node], rt_a[node], rep_h[node], rep_a[node], mention_h[node], mention_a[node]]
        ordered_test_nodes[index] = node

    predictions = clf.predict(X)

    # sort by rt_h score
    with open('output/calculated_top_' + n_input + '.txt', 'w+') as f:
        count = 0
        for i,p in sorted(enumerate(predictions), key=operator.itemgetter(1), reverse=True):
            if count >= int(n_input):
                break
            f.write(str(ordered_test_nodes[i]))
            f.write(' ')
            f.write(str(p))
            f.write("\n")
            count += 1
