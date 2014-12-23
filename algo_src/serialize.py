import operator
import networkx as nx
import numpy as np
from sklearn import svm
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
    in_count, out_count = 0.0
    for u,v,d in G.in_edges_iter(node, data=True):
        in_count += d['weight']

    for u,v,d in G.out_edges_iter(node, data=True):
        out_count += d['weight']

    return in_count / out_count

if __name__ == "__main__":
    retweet_graph = nx.DiGraph()
    nx.read_weighted_edgelist('data/higgs-retweet_network.edgelist', create_using=retweet_graph)
    # reply_graph = nx.DiGraph()
    # nx.read_weighted_edgelist('data/higgs-reply_network.edgelist', create_using=reply_graph)
    # mention_graph = nx.DiGraph()
    # nx.read_weighted_edgelist('data/higgs-mention_network.edgelist', create_using=mention_graph)

    # load top N
    # validation_top_n = defaultdict(lambda: 0)
    # with open('output/validation_top_n.txt', 'r') as f:
    #     for line in f:
    #         validation_top_n.append(line)

    # calculate h/a for each graph
    print "made it"
    rt_h, rt_a = hits(retweet_graph)
    # rep_h, rep_a = hits(reply_graph)
    # mention_h, mention_a = hits(mention_graph)
    #
    # # get set of unique nodes in all graphs
    # all_nodes = set(retweet_graph.nodes()) | set(reply_graph.nodes()) | set(mention_graph.nodes())
    #
    # # populate X
    # X = numpy.zeros([len(all_nodes), 9])
    # for index, node in enumerate(all_nodes):
    #     X[index] = [rt_h[node], rt_a[node], rep_h[node], rep_a[node], mention_h[node], mention_a[node], calc_ratio(retweet_graph, node), calc_ratio(reply_graph, node), calc_ratio(mention_graph, node)]

    # sort by rt_h score
    with open('output/rt_h_top_n.txt', 'w+') as f:
        print "made it"
        for node,h_score in sorted(rt_h.items(), key=operator.itemgetter(1)):
            f.write(node)
            f.write("\n")


    # training_nodes = extract_p(.2)
