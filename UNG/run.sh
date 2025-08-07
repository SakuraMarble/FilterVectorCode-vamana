#!/bin/bash

IS_BUILD_AND_GT="true"

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) # Get the absolute path of the script's directory

# Step1: Parse command-line arguments
while [[ $# -gt 0 ]]; do
    if [[ $1 == --* ]]; then
        # Convert --key-name to KEY_NAME
        key=$(echo "$1" | sed 's/--//' | tr '[:lower:]-' '[:upper:]_')
        # 检查参数后面是否有值
        if [ -z "$2" ]; then
            echo "Error: Missing value for parameter $1"
            exit 1
        fi
        # Assign value: e.g., DATASET="$2"
        declare "$key"="$2"
        shift 2
    else
        echo "Unknown parameter: $1"
        exit 1
    fi
done

# Step2: Delete the old build folder
if [ -d "$BUILD_DIR" ]; then
    echo "Deleting $BUILD_DIR folder and its contents..."
    rm -rf "$BUILD_DIR"
fi


# Step3: Create build directory and compile the code
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR" || exit
cmake -DCMAKE_BUILD_TYPE=Release  "${SCRIPT_DIR}/codes/"
make -j
# make VERBOSE=1 -j
cd .. || exit


# Step3.5: Construct the output directory path with Lsearch parameters
if [[ "$UNG_AND_ACORN" == "false" ]]; then
    OUTPUT_DIR="${OUTPUT_DIR}/UNG/${DATASET}/${DATASET}_query${NUM_QUERY_SETS}_sep${IS_SELECT_ENTRY_GROUPS}_th${NUM_THREADS}_M${MAX_DEGREE}_LB${LBUILD}_alpha${ALPHA}_C${NUM_CROSS_EDGES}_EP${NUM_ENTRY_POINTS}_Ls${LSEARCH_START}_Le${LSEARCH_END}_Lp${LSEARCH_STEP}_REPEATs${NUM_REPEATS}"
elif [[ "$UNG_AND_ACORN" == "true" ]]; then
    OUTPUT_DIR="${OUTPUT_DIR}/UNG+ACORN/${DATASET}/${DATASET}_query${NUM_QUERY_SETS}_sep${IS_SELECT_ENTRY_GROUPS}_th${NUM_THREADS}_${NEW_EDGE_POLICY}_aefs${EFS}_R${R_IN_ADD_NEW_EDGE}_W${W_in_add_new_edge}_M${M_IN_ADD_NEW_EDGE}_l${LAYER_DEPTH_RETIO}_q${QUERY_VECTOR_RATIO}_r${ROOT_COVERAGE_THRESHOLD}_M${MAX_DEGREE}_LB${LBUILD}_alpha${ALPHA}_C${NUM_CROSS_EDGES}_EP${NUM_ENTRY_POINTS}_Ls${LSEARCH_START}_Le${LSEARCH_END}_Lp${LSEARCH_STEP}_REPEATs${NUM_REPEATS}"
    ACORN_IN_UNG_OUTPUT_DIR="$OUTPUT_DIR/acorn_in_ung_output"
    mkdir -p "$ACORN_IN_UNG_OUTPUT_DIR"
else
    echo "Error: The value of UNG_AND_ACORN must be \"true\" or \"false\""
    exit 1
fi

OTHER_DIR="$OUTPUT_DIR/others"
RESULT_DIR="$OUTPUT_DIR/results"
if [ "$IS_BUILD_AND_GT" == "true" ]; then
    mkdir -p "$OUTPUT_DIR"
    mkdir -p "$OTHER_DIR"
    mkdir -p "$RESULT_DIR"
fi


if [ "$IS_BUILD_AND_GT" == "true" ]; then
    # Step4: Convert base data format
    "$BUILD_DIR"/tools/fvecs_to_bin --data_type float --input_file "$DATA_DIR/${DATASET}_base.fvecs" --output_file "$DATA_DIR/${DATASET}_base.bin"


    # Step5: Build index + Generate query task file
    "$BUILD_DIR"/apps/build_UNG_index \
       --data_type float --dist_fn L2 --num_threads "$NUM_THREADS" --max_degree "$MAX_DEGREE" --Lbuild "$LBUILD" --alpha "$ALPHA" --num_cross_edges "$NUM_CROSS_EDGES"\
       --base_bin_file "$DATA_DIR/${DATASET}_base.bin" \
       --base_label_file "$DATA_DIR/base_${NUM_QUERY_SETS}/${DATASET}_base_labels.txt" \
       --base_label_info_file "$DATA_DIR/base_${NUM_QUERY_SETS}/${DATASET}_base_labels_info.log" \
       --base_label_tree_roots "$DATA_DIR/base_${NUM_QUERY_SETS}/tree_roots.txt" \
       --index_path_prefix "$OUTPUT_DIR/index_files/" \
       --result_path_prefix "$RESULT_DIR/" \
       --scenario general \
       --generate_query "$GENERATE_QUERY"  --generate_query_task "$GENERATE_QUERY_TASK" --method1_high_coverage_p "$METHOD1_HIGH_COVERAGE_P" \
       --query_file_path "$DATA_DIR/query_${NUM_QUERY_SETS}" \
       --dataset "$DATASET" \
       --ung_and_acorn "$UNG_AND_ACORN"  --new_edge_policy "$NEW_EDGE_POLICY" \
       --R_in_add_new_edge "$R_IN_ADD_NEW_EDGE"  --W_in_add_new_edge "$W_IN_ADD_NEW_EDGE"  --M_in_add_new_edge "$M_IN_ADD_NEW_EDGE" \
       --layer_depth_retio "$LAYER_DEPTH_RETIO"  --query_vector_ratio "$QUERY_VECTOR_RATIO"  --root_coverage_threshold "$ROOT_COVERAGE_THRESHOLD" \
       --acorn_in_ung_output_path "$ACORN_IN_UNG_OUTPUT_DIR" \
       --M "$M"  --M_beta "$M_BETA"  --gamma "$GAMMA"  --efs "$EFS"  --compute_recall "$COMPUTE_RECALL" > "$OTHER_DIR/${DATASET}_build_index_output.txt" 2>&1


    # Step6: Convert query data format
    for ((i=NUM_QUERY_SETS; i<=NUM_QUERY_SETS; i++))
    do
       INPUT_FILE="$DATA_DIR/query_${NUM_QUERY_SETS}/${DATASET}_query.fvecs"
       OUTPUT_FILE="$DATA_DIR/query_${NUM_QUERY_SETS}/${DATASET}_query.bin"
       echo "Processing set $i: $INPUT_FILE -> $OUTPUT_FILE"
       "$BUILD_DIR"/tools/fvecs_to_bin --data_type float --input_file "$INPUT_FILE" --output_file "$OUTPUT_FILE"
    done


    # Step7: Generate ground truth
    for ((i=NUM_QUERY_SETS; i<=NUM_QUERY_SETS; i++))
    do
       "$BUILD_DIR"/tools/compute_groundtruth \
          --data_type float --dist_fn L2 --scenario containment --K "$K" --num_threads "$NUM_THREADS" \
          --base_bin_file "$DATA_DIR/${DATASET}_base.bin" \
          --base_label_file "$DATA_DIR/base_${NUM_QUERY_SETS}/${DATASET}_base_labels.txt" \
          --query_bin_file "$DATA_DIR/query_${NUM_QUERY_SETS}/${DATASET}_query.bin" \
          --query_label_file "$DATA_DIR/query_${NUM_QUERY_SETS}/${DATASET}_query_labels.txt" \
          --gt_file "$DATA_DIR/query_${NUM_QUERY_SETS}/${DATASET}_gt_labels_containment.bin"
       if [ $? -ne 0 ]; then
          echo "Error generating GT for set $i"
          exit 1
       fi
    done
    echo -e "All ground truth files generated successfully!"
fi

# Step8: Execute search
# Generate the sequence of Lsearch values
LSEARCH_VALUES=$(seq "$LSEARCH_START" "$LSEARCH_STEP" "$LSEARCH_END" | tr '\n' ' ')
echo "Generated Lsearch values to test: $LSEARCH_VALUES"

for ((i=NUM_QUERY_SETS; i<=NUM_QUERY_SETS; i++))
do    
    echo -e "\nRunning query$i..."
    QUERY_DIR="$DATA_DIR/query_${NUM_QUERY_SETS}"
    "$BUILD_DIR"/apps/search_UNG_index \
        --data_type float --dist_fn L2 --num_threads "$NUM_THREADS" --K "$K" --is_new_method true --is_ori_ung true  --is_select_entry_groups "$IS_SELECT_ENTRY_GROUPS"  --num_repeats "$NUM_REPEATS" \
        --base_bin_file "$DATA_DIR/${DATASET}_base.bin" \
        --base_label_file "$DATA_DIR/base_${NUM_QUERY_SETS}/${DATASET}_base_labels.txt" \
        --query_bin_file "$QUERY_DIR/${DATASET}_query.bin" \
        --query_label_file "$QUERY_DIR/${DATASET}_query_labels.txt" \
        --query_group_id_file "$QUERY_DIR/${DATASET}_query_source_groups.txt" \
        --gt_file "$QUERY_DIR/${DATASET}_gt_labels_containment.bin" \
        --index_path_prefix "$OUTPUT_DIR/index_files/" \
        --result_path_prefix "$RESULT_DIR/" \
        --scenario containment \
        --num_entry_points "$NUM_ENTRY_POINTS" \
        --Lsearch $LSEARCH_VALUES > "$OTHER_DIR/${DATASET}_search_output.txt" 2>&1

    if [ $? -ne 0 ]; then
        echo "Error in iteration $i"
        exit 1
    fi
done

echo -e "All search iterations completed successfully for dataset $DATASET!"