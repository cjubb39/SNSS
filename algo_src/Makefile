NETWORK_DATA ?= data/higgs-social_network.edgelist
SORTED_NETWORK ?= output/full_sorted_network

KNOWN_N ?= 20
GOAL_N ?= 100
TOP_OUTPUT_FILE ?= output/top_nodes_$(GOAL_N)
RAND_OUTPUT_FILE ?= output/random_nodes_$(KNOWN_N)
USE_KERNEL_NAME ?= linear
KERNELS_NAME ?= output/best_kernel_$(KNOWN_N)
GUESS_OUTPUT_FILE ?= output/calculated_nodes_$(KNOWN_N)_$(USE_KERNEL_NAME)

SORT_DATA_SCRIPT = sort_network_by_degree.py
GENERATE_KERNEL_SCRIPT = find_best_kernel_params.py
GENERATE_DATA_SCRIPT = serialize.py
CHECK_SCRIPT = check.py

check_$(GOAL_N)_with_$(KNOWN_N): check

reset:
	touch $(SORTED_NETWORK)

help:
	@echo "Environment Variables will override test parameters."
	@echo "Defaults are as follows:\n"
	@head -n10 Makefile
	@echo
	@echo "'rand' target produced random nodes with degrees"

check: $(GUESS_OUTPUT_FILE) $(TOP_OUTPUT_FILE)
	python $(CHECK_SCRIPT) $(GUESS_OUTPUT_FILE) $(TOP_OUTPUT_FILE)

$(GUESS_OUTPUT_FILE): $(GENERATE_DATA_SCRIPT) $(RAND_OUTPUT_FILE) $(KERNELS_NAME) $(TOP_OUTPUT_FILE)
	python $(GENERATE_DATA_SCRIPT) $(KNOWN_N) $(GOAL_N) $(USE_KERNEL_NAME) \
		$(KERNELS_NAME) $(RAND_OUTPUT_FILE) $(GUESS_OUTPUT_FILE)

$(KERNELS_NAME): $(GENERATE_KERNEL_SCRIPT) $(RAND_OUTPUT_FILE)
	python $(GENERATE_KERNEL_SCRIPT) $(KNOWN_N) $(RAND_OUTPUT_FILE) $(KERNELS_NAME)

$(RAND_OUTPUT_FILE): $(SORTED_NETWORK)
	cat $(SORTED_NETWORK) | shuf | head -n$(KNOWN_N) > $(RAND_OUTPUT_FILE)

$(TOP_OUTPUT_FILE): $(SORTED_NETWORK)
	head -n$(GOAL_N) $(SORTED_NETWORK) > $(TOP_OUTPUT_FILE)

$(SORTED_NETWORK): $(NETWORK_DATA) $(SORT_DATA_SCRIPT)
	@mkdir -p output
	python $(SORT_DATA_SCRIPT) $(NETWORK_DATA) $(SORTED_NETWORK)

$(NETWORK_DATA):


light_clean:
	find . -regex './output/top_nodes_[0-9]+' -delete
	find . -regex './output/random_nodes_[0-9]+' -delete
	find . -regex './output/calculated_nodes_[0-9]+_[A-Za-z]+' -delete

clean: light_clean
	find . -regex './output/best_kernel_[0-9]+' -delete
	rm -f $(SORTED_NETWORK)
	@rmdir output > /dev/null 2>&1 && echo "output folder deleted" \
		|| echo "output folder not deleted"
