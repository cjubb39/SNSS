import sys

n_input = sys.argv[1]

results = []
with open('output/calculated_top_' + n_input + '.txt', 'r') as f:
    for line in f:
        results.append(line.split()[0])

solution = []
with open('output/top_' + n_input + '_nodes', 'r') as f:
    for line in f:
        solution.append(line.split()[0])

print len(set(solution) & set(results))
