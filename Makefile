NETWORK_DATA ?= data/higgs-social_network.edgelist
N ?= 20
OUTPUT_FILE ?= top_$(N)_nodes

default: generate_top_nodes

help:
	@echo "Environment Variables will override test parameters."
	@echo "Defaults are as follows:\n"
	@head -n3 Makefile

generate_top_nodes:
	python generate_top_nodes.py $(NETWORK_DATA) $(N) $(OUTPUT_FILE)

clean:
	find . -regex './top_[0-9]+_nodes' -delete
