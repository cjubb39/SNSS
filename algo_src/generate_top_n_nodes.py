from __future__ import print_function
import sys
import re
from operator import itemgetter
import networkx as nx

data_file = sys.argv[1]
n_top_n = int(sys.argv[2])
output_file = sys.argv[3]

#setup
G = nx.DiGraph()
f = open(data_file, 'r')
out = open(output_file, 'w')

while True:
	line_in = f.readline()
	if line_in.startswith("#"):
		continue
	if line_in == '':
		break

	# add to graph
	m = re.search("(\d*) (\d*)\n", line_in)
	node1 = int(m.group(1))
	node2 = int(m.group(2))
	if node1 == node2:
		G.add_node(node1)
	else:
		G.add_edge(node1, node2)

# what a beautiful graph
top_nodes = sorted(G.in_degree_iter(), key=itemgetter(1), reverse=True)[0:n_top_n]

for n in top_nodes:
	print("%d %d" % (n[0], n[1]), file=out)
