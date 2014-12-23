NETWORK_DATA ?= data/higgs-social_network.edgelist
N ?= 20
OUTPUT_FILE ?= output/top_$(N)_nodes

PYTHON_SCRIPT = sort_network_by_degree.py
SORTED_NETWORK = output/sorted_network

create_sorted_$(N): $(OUTPUT_FILE)

help:
	@echo "Environment Variables will override test parameters."
	@echo "Defaults are as follows:\n"
	@head -n3 Makefile

$(OUTPUT_FILE): $(SORTED_NETWORK)
	head -n$(N) $(SORTED_NETWORK) > $(OUTPUT_FILE)

$(SORTED_NETWORK): $(NETWORK_DATA) $(PYTHON_SCRIPT)
	@mkdir -p output
	python $(PYTHON_SCRIPT) $(NETWORK_DATA) $(SORTED_NETWORK)

$(NETWORK_DATA):

clean:
	find . -regex './output/top_[0-9]+_nodes' -delete
	rm -f $(SORTED_NETWORK)
	@rmdir output > /dev/null 2>&1 && echo "output folder deleted" \
		|| echo "output folder not deleted"
