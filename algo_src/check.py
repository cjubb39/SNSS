import sys

calculated_top = sys.argv[1]
actual_top = sys.argv[2]

results = []
with open(calculated_top, 'r') as f:
    for line in f:
        results.append(line.split()[0])

solution = []
with open(actual_top, 'r') as f:
    for line in f:
        solution.append(line.split()[0])

print str(len(set(solution) & set(results))) + " of " + str(len(results))
