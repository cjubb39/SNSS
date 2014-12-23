NETWORK_DATA ?= data/higgs-social_network.edgelist
N ?= 20
OUTPUT_FILE ?= top_$(N)_nodes

default: $(OUTPUT_FILE)

help:
	@echo "Environment Variables will override test parameters."
	@echo "Defaults are as follows:\n"
	@head -n3 Makefile

$(OUTPUT_FILE): $(NETWORK_DATA)
	python generate_top_nodes.py $(NETWORK_DATA) $(N) $(OUTPUT_FILE)

$(NETWORK_DATA):

clean:
	find . -regex './top_[0-9]+_nodes' -delete
