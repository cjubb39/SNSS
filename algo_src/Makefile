NETWORK_DATA ?= data/higgs-social_network.edgelist
N ?= 20
OUTPUT_FILE ?= output/top_$(N)_nodes

default: $(OUTPUT_FILE)

help:
	@echo "Environment Variables will override test parameters."
	@echo "Defaults are as follows:\n"
	@head -n3 Makefile

$(OUTPUT_FILE): $(NETWORK_DATA) generate_top_n_nodes.py
	@mkdir -p output
	python generate_top_n_nodes.py $(NETWORK_DATA) $(N) $(OUTPUT_FILE)

$(NETWORK_DATA):

clean:
	find . -regex './output/top_[0-9]+_nodes' -delete
