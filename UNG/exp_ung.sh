#!/bin/bash

# Create a temporary directory under /home and set environment variables
export TMPDIR="/data/fxy/FilterVector/build"
mkdir -p "$TMPDIR"

# Check if jq is installed
if ! command -v jq &> /dev/null; then
    echo "Error: jq is not installed. Please install jq first (https://stedolan.github.io/jq/)"
    exit 1
fi


# Read JSON configuration
CONFIG_FILE="../../FilterVectorCode/UNG/experiments.json"


# Iterate through all experiments
cat "$CONFIG_FILE" | jq -c '.experiments[]' | while read -r experiment; do
    dataset=$(echo "$experiment" | jq -r '.dataset')
    echo -e "============================================"
    echo "Starting to process dataset: $dataset"
    echo "============================================"
    ./run.sh \
        --dataset "$dataset" \
        --data_dir "$(echo "$experiment" | jq -r '.data_dir')" \
        --output_dir "$(echo "$experiment" | jq -r '.output_dir')" \
        --num_query_sets "$(echo "$experiment" | jq -r '.num_query_sets')" \
        --max_degree "$(echo "$experiment" | jq -r '.max_degree')" \
        --Lbuild "$(echo "$experiment" | jq -r '.Lbuild')" \
        --alpha "$(echo "$experiment" | jq -r '.alpha')" \
        --num_cross_edges "$(echo "$experiment" | jq -r '.num_cross_edges')" \
        --num_entry_points "$(echo "$experiment" | jq -r '.num_entry_points')" \
        --Lsearch_start "$(echo "$experiment" | jq -r '.Lsearch_start')" \
        --Lsearch_end "$(echo "$experiment" | jq -r '.Lsearch_end')" \
        --Lsearch_step "$(echo "$experiment" | jq -r '.Lsearch_step')" \
        --build_dir "/data/fxy/FilterVector/build/build_$dataset" \
        --num_threads "$(echo "$experiment" | jq -r '.num_threads')" \
        --K "$(echo "$experiment" | jq -r '.K')" \
        --num_repeats "$(echo "$experiment" | jq -r '.num_repeats')" \
        --is_new_trie_method "$(echo "$experiment" | jq -r '.trie_param.is_new_trie_method')" \
        --is_rec_more_start "$(echo "$experiment" | jq -r '.trie_param.is_rec_more_start')" \
        --generate_query "$(echo "$experiment" | jq -r '.generate_query')" \
        --generate_query_task "$(echo "$experiment" | jq -r '.generate_query_task')" \
        --method1_high_coverage_p "$(echo "$experiment" | jq -r '.method1_high_coverage_p')" \
        --is_ung_more_entry "$(echo "$experiment" | jq -r '.ung_more_entry_param.is_ung_more_entry')" \
        --ung_and_acorn "$(echo "$experiment" | jq -r '.ung_and_acorn_param.ung_and_acorn')" \
        --new_edge_policy "$(echo "$experiment" | jq -r '.ung_and_acorn_param.new_edge_policy')" \
        --R_in_add_new_edge "$(echo "$experiment" | jq -r '.ung_and_acorn_param.R_in_add_new_edge')" \
        --W_in_add_new_edge "$(echo "$experiment" | jq -r '.ung_and_acorn_param.W_in_add_new_edge')" \
        --M_in_add_new_edge "$(echo "$experiment" | jq -r '.ung_and_acorn_param.M_in_add_new_edge')" \
        --layer_depth_retio "$(echo "$experiment" | jq -r '.ung_and_acorn_param.layer_depth_retio')" \
        --query_vector_ratio "$(echo "$experiment" | jq -r '.ung_and_acorn_param.query_vector_ratio')" \
        --root_coverage_threshold "$(echo "$experiment" | jq -r '.ung_and_acorn_param.root_coverage_threshold')" \
        --M "$(echo "$experiment" | jq -r '.ung_and_acorn_param.acorn_in_ung.M')" \
        --M_beta "$(echo "$experiment" | jq -r '.ung_and_acorn_param.acorn_in_ung.M_beta')" \
        --gamma "$(echo "$experiment" | jq -r '.ung_and_acorn_param.acorn_in_ung.gamma')" \
        --efs "$(echo "$experiment" | jq -r '.ung_and_acorn_param.acorn_in_ung.efs')" \
        --compute_recall "$(echo "$experiment" | jq -r '.ung_and_acorn_param.acorn_in_ung.compute_recall')"
    echo "Dataset $dataset processing complete"
    echo "============================================"
done

echo "All experiments are complete, cleaning up temporary build directory..."
if [ -d "$TMPDIR" ]; then
    rm -rf "$TMPDIR"
    echo "Temporary directory $TMPDIR has been successfully deleted."
fi
