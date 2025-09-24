#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "======================================================================="
echo "               Compile and Build UNG & ACORN Indexes"
echo "======================================================================="
echo

# --- Path Definitions ---
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
CONFIG_FILE="${SCRIPT_DIR}/hybrid_build_config.json"
BASE_OUTPUT_DIR="/data/fxy/FilterVector/FilterVectorResults/parall/parall_build"

# Build directories
BASE_BUILD_DIR="/data/fxy/FilterVector/build_para"
UNG_BUILD_DIR="${BASE_BUILD_DIR}/ung"
ACORN_BUILD_DIR="${BASE_BUILD_DIR}/acorn"


# --- Dependency Checks ---
if ! command -v jq &> /dev/null; then
    echo "[ERROR] Core dependency 'jq' is not installed. Please install it to continue." >&2
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "[ERROR] Configuration file '$CONFIG_FILE' not found." >&2
    exit 1
fi


# --- [Stage 1/3] Project Compilation ---
echo "--- [Stage 1/3] Checking and Compiling Projects ---"

# 1. Compile UNG
UNG_EXECUTABLE="${UNG_BUILD_DIR}/apps/build_UNG_index"
if [ ! -f "$UNG_EXECUTABLE" ]; then
    echo "[INFO] UNG executable not found. Starting compilation..."
    mkdir -p "$UNG_BUILD_DIR"
    cmake -S "${SCRIPT_DIR}/UNG/codes" -B "$UNG_BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
    make -C "$UNG_BUILD_DIR" -j
    echo "[SUCCESS] UNG compilation complete."
else
    echo "[INFO] UNG executable already exists. Skipping compilation."
fi

# 2. Compile ACORN
ACORN_EXECUTABLE="${ACORN_BUILD_DIR}/demos/test_acorn"
if [ ! -f "$ACORN_EXECUTABLE" ]; then
    echo "[INFO] ACORN executable not found. Starting compilation..."
    mkdir -p "$ACORN_BUILD_DIR"
    cmake -S "${SCRIPT_DIR}/ACORN" -B "$ACORN_BUILD_DIR" \
          -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF \
          -DBUILD_TESTING=ON -DBUILD_SHARED_LIBS=ON \
          -DCMAKE_BUILD_TYPE=Release
    make -C "$ACORN_BUILD_DIR" -j test_acorn
    echo "[SUCCESS] ACORN compilation complete."
else
    echo "[INFO] ACORN executable already exists. Skipping compilation."
fi
echo


# --- [Stage 2/3] Executing Experiments from Configuration ---
echo "--- [Stage 2/3] Parsing Configuration and Running Experiments ---"

# Use a while loop to process each experiment object provided by jq.
jq -c '.experiments[]' < "$CONFIG_FILE" | while read -r experiment; do
    
    echo
    echo "-----------------------------------------------------------------------"
    echo "                    Processing New Experiment"
    echo "-----------------------------------------------------------------------"

    # --- Parse Parameters for the Current Experiment ---
    DATASET=$(echo "$experiment" | jq -r '.common_params.dataset')
    QUERY_NUM=$(echo "$experiment" | jq -r '.common_params.query_num')
    BASE_DATA_DIR=$(echo "$experiment" | jq -r '.common_params.base_data_dir')
    UNG_MAX_DEGREE=$(echo "$experiment" | jq -r '.ung_params.max_degree')
    UNG_LBUILD=$(echo "$experiment" | jq -r '.ung_params.Lbuild')
    UNG_ALPHA=$(echo "$experiment" | jq -r '.ung_params.alpha')
    UNG_NUM_CROSS_EDGES=$(echo "$experiment" | jq -r '.ung_params.num_cross_edges')
    ACORN_N=$(echo "$experiment" | jq -r '.acorn_params.N')
    ACORN_M=$(echo "$experiment" | jq -r '.acorn_params.M')
    ACORN_M_BETA=$(echo "$experiment" | jq -r '.acorn_params.M_beta')
    ACORN_GAMMA=$(echo "$experiment" | jq -r '.acorn_params.gamma')

    # --- Create a unique output directory for this experiment's results ---
    PARAMS_SUFFIX="UNG_M${UNG_MAX_DEGREE}_LB${UNG_LBUILD}_A${UNG_ALPHA}_CE${UNG_NUM_CROSS_EDGES}_ACORN_M${ACORN_M}_G${ACORN_GAMMA}"
    UNIFIED_OUTPUT_DIR="${BASE_OUTPUT_DIR}/${DATASET}/query_${QUERY_NUM}/${PARAMS_SUFFIX}"

    echo "[INFO] Preparing environment for experiment with query_num=${QUERY_NUM}..."
    echo "[INFO] Output directory set to: ${UNIFIED_OUTPUT_DIR}"
    mkdir -p "$UNIFIED_OUTPUT_DIR"
    
    echo "[INFO] Cleaning up old log files..."
    rm -f "${UNIFIED_OUTPUT_DIR}/ung_build.log" "${UNIFIED_OUTPUT_DIR}/acorn_build.log"

    base_path="../FilterVectorData/${DATASET}"
    query_path="../FilterVectorData/${DATASET}/query_${QUERY_NUM}"
    base_label_path="../FilterVectorData/${DATASET}/base_${QUERY_NUM}"

    # --- [Stage 3/3] Parallel Index Building ---
    echo
    echo "--- [Stage 3/3] Starting Parallel Build Tasks for query_num=${QUERY_NUM} ---"
    start_time=$(date +%s)

    # 1. Start UNG build in the background
    (
        echo "[UNG] Build process started."
        "$UNG_EXECUTABLE" \
            --data_type float --dist_fn L2 --num_threads 32 \
            --max_degree "$UNG_MAX_DEGREE" --Lbuild "$UNG_LBUILD" --alpha "$UNG_ALPHA" --num_cross_edges "$UNG_NUM_CROSS_EDGES" \
            --base_bin_file "$BASE_DATA_DIR/${DATASET}/${DATASET}_base.bin" \
            --base_label_file "$BASE_DATA_DIR/${DATASET}/base_${QUERY_NUM}/${DATASET}_base_labels.txt" \
            --base_label_info_file "$BASE_DATA_DIR/${DATASET}/base_${QUERY_NUM}/${DATASET}_base_labels_info.log" \
            --base_label_tree_roots "$BASE_DATA_DIR/${DATASET}/base_${QUERY_NUM}/tree_roots.txt" \
            --index_path_prefix "${UNIFIED_OUTPUT_DIR}/ung_index_files/" \
            --result_path_prefix "${UNIFIED_OUTPUT_DIR}/results/" \
            --scenario general \
            --dataset "$DATASET" \
            --generate_query false \
            --ung_and_acorn false
        echo "[UNG] Build process finished."
    ) > "${UNIFIED_OUTPUT_DIR}/ung_build.log" 2>&1 &
    ung_pid=$!
    echo "[INFO] UNG build task started in background (PID: $ung_pid). Log: ${UNIFIED_OUTPUT_DIR}/ung_build.log"

    # 2. Start ACORN build in the background
    (
        echo "[ACORN] Build process started."
        "$ACORN_EXECUTABLE" build \
            "$ACORN_N" "$ACORN_GAMMA" "$DATASET" "$ACORN_M" "$ACORN_M_BETA" \
            "$base_path" "$base_label_path" "$query_path" \
            "dummy_csv_dir" "dummy_avg_csv_dir" "dummy_dis_path" \
            32 1 true "10" \
            "${UNIFIED_OUTPUT_DIR}/acorn_index_files/acorn.index" \
            "${UNIFIED_OUTPUT_DIR}/acorn_index_files/acorn1.index" \
            0
        echo "[ACORN] Build process finished."
    ) > "${UNIFIED_OUTPUT_DIR}/acorn_build.log" 2>&1 &
    acorn_pid=$!
    echo "[INFO] ACORN build task started in background (PID: $acorn_pid). Log: ${UNIFIED_OUTPUT_DIR}/acorn_build.log"

    # --- Wait for the current experiment's tasks to complete ---
    echo "[INFO] Waiting for build tasks for query_num=${QUERY_NUM} to complete..."
    wait $ung_pid
    ung_status=$?
    wait $acorn_pid
    acorn_status=$?

    end_time=$(date +%s)
    duration=$((end_time - start_time))

    # --- Report results for the current experiment ---
    echo
    echo "--- Experiment Summary (query_num: ${QUERY_NUM}) ---"
    if [ $ung_status -ne 0 ]; then
        echo "[WARNING] UNG build process exited with a non-zero status ($ung_status). Check logs for details."
    fi
    if [ $acorn_status -ne 0 ]; then
        echo "[WARNING] ACORN build process exited with a non-zero status ($acorn_status). Check logs for details."
    fi

    echo "[SUCCESS] Indexes generated in: $UNIFIED_OUTPUT_DIR"
    echo "[INFO] Total parallel build time: $duration seconds."
    echo "-----------------------------------------------------------------------"

done # End of the experiment loop

echo
echo "======================================================================="
echo "                          All experiments complete!"
echo "======================================================================="
echo
