#!/bin/bash

# LRC Microbenchmark Experiment Runner
# This script automates running the LRC coalescing experiments

set -e

BINARY="./l2_lrc_effect.o"
OUTPUT_DIR="./lrc_results"
NCU_METRICS="lrc__lts2lrc_sectors_op_read.sum,lrc__xbar2gpc_sectors_op_read.sum,lrc__average_xbar2gpc_sectors_op_read.ratio"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=== LRC Microbenchmark Experiment Suite ==="
echo "Binary: $BINARY"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Check if binary exists
if [ ! -f "$BINARY" ]; then
    echo "Error: Binary $BINARY not found. Please compile first:"
    echo "  nvcc -Xptxas -dlcm=cg -Xptxas -dscm=wt l2_lrc_effect.cu -o l2_lrc_effect.o"
    exit 1
fi

# Function to extract metric from ncu output
extract_metric() {
    local csv_file="$1"
    local metric_name="$2"
    
    if [ ! -f "$csv_file" ]; then
        echo "N/A"
        return
    fi
    
    # CSV format can vary (e.g. "Metric,Unit,Value" or "Metric,Value"), so use $NF to get the last column
    awk -F',' -v metric="$metric_name" '$1 ~ metric {gsub(/"/,"",$NF); print $NF}' "$csv_file" | head -1
}

# Function to run experiment and collect metrics
run_experiment() {
    local test_name="$1"
    local test_mode="$2"
    local num_warps="$3"
    local offset_bytes="$4"
    local delay_cycles="$5"
    local capacity_test="${6:-0}"
    local accesses_per_warp="${7:-1}"
    
    local output_file="$OUTPUT_DIR/${test_name}"
    local ncu_rep_file="${output_file}.ncu-rep"
    local csv_file="${output_file}.csv"
    local log_file="${output_file}.log"
    
    echo "Running: $test_name"
    echo "  Mode: $test_mode, Warps: $num_warps, Offset: $offset_bytes, Delay: $delay_cycles cycles, Capacity Test: $capacity_test, Accesses/Warp: $accesses_per_warp"
    
    # Run with NSight Compute
    # /accel-sim/ncu/ncu --metrics "$NCU_METRICS" \
    #     --export "$output_file" \
    #     --force-overwrite \
    #     "$BINARY" "$test_mode" "$num_warps" "$offset_bytes" "$delay_cycles" "$capacity_test" "$accesses_per_warp" \
    #     > "$log_file" 2>&1

    /accel-sim/ncu/ncu -f --set detailed -o "$output_file" --target-processes "all"\
        "$BINARY" "$test_mode" "$num_warps" "$offset_bytes" "$delay_cycles" "$capacity_test" "$accesses_per_warp" \
        > "$log_file" 2>&1
    /accel-sim/ncu/ncu --import "$ncu_rep_file" --csv --page raw > "$csv_file.tmp" 2>/dev/null

    # Transpose it using python
    python3 -c "import sys, csv; reader = csv.reader(sys.stdin); writer = csv.writer(sys.stdout); input_data = list(reader); writer.writerows(zip(*input_data))" < "${csv_file}.tmp" > "$csv_file"

    rm "${csv_file}.tmp"
    sleep 1
    # Check if CSV was created
    if [ ! -f "$csv_file" ]; then
        echo "Warning: CSV file not created. Checking for alternative names..."
        ls -la "$OUTPUT_DIR"/${test_name}* || true
    fi
    # Extract metrics
    local lts2lrc=$(extract_metric "$csv_file" "lrc__lts2lrc_sectors_op_read.sum" | tr -d '\r')
    local xbar2gpc=$(extract_metric "$csv_file" "lrc__xbar2gpc_sectors_op_read.sum" | tr -d '\r')
    local ratio=$(extract_metric "$csv_file" "lrc__average_xbar2gpc_sectors_op_read.ratio" | tr -d '\r')

    # Calculate coalescing metrics
    if [ -n "$lts2lrc" ] && [ -n "$xbar2gpc" ] && [ "$lts2lrc" != "0" ]; then
        local coalescing_ratio=$(echo "scale=4; $xbar2gpc / $lts2lrc" | bc)
        local coalescing_factor=$(echo "scale=4; $lts2lrc / $xbar2gpc" | bc)
    else
        local coalescing_ratio="N/A"
        local coalescing_factor="N/A"
    fi
    
    # Append to CSV
    echo "$test_name,$test_mode,$num_warps,$offset_bytes,$delay_cycles,$lts2lrc,$xbar2gpc,$ratio,$coalescing_ratio,$coalescing_factor" >> "$OUTPUT_DIR/results.csv"
    
    echo "  Results: lts2lrc=$lts2lrc, xbar2gpc=$xbar2gpc, ratio=$coalescing_ratio, factor=$coalescing_factor"
    echo ""
}

# Initialize results CSV
echo "test_name,test_mode,num_warps,offset_bytes,delay_cycles,lts2lrc_sectors,xbar2gpc_sectors,avg_ratio,coalescing_ratio,coalescing_factor" > "$OUTPUT_DIR/results.csv"

# Test 1: Coalescing Capacity (Question 1)
echo "=== Test 1: Coalescing Capacity ==="
echo "Varying number of warps accessing same address"
for warps in 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384; do
    # run_experiment "capacity_${warps}warps" 0 "$warps" 0 0
    run_experiment "capacity_1warps_${warps}accesses" 0 1 0 0 0 "$warps"
done

# # Test 2: Temporal Coalescing Window (Question 2)
# echo "=== Test 2: Temporal Coalescing Window ==="
# echo "Varying delay between accesses (32 warps, same address)"
# for delay in 0 5 10 20 50 100 200 500 1000; do
#     run_experiment "temporal_delay${delay}cycles" 3 32 0 "$delay"
# done

# # Test 3: Address Coalescing Conditions (Question 3)
# echo "=== Test 3: Address Coalescing Conditions ==="

# # 3a: Same address
# echo "3a: Same address (baseline)"
# run_experiment "same_address" 0 32 0 0

# # 3b: Same cacheline, different offsets
# echo "3b: Same cacheline, different byte offsets"
# for offset in 0 4 8 16 32 64 128 256; do
#     run_experiment "same_cacheline_offset${offset}" 1 32 "$offset" 0
# done

# # 3c: Different cachelines
# echo "3c: Different cachelines"
# run_experiment "diff_cacheline" 2 32 0 0

# # Test 4: Capacity test with multiple accesses per warp
# echo "=== Test 4: Multiple Accesses Per Warp ==="
# echo "Testing if multiple accesses from same warp can be coalesced"
# for accesses in 1 2 4 8; do
#     run_experiment "capacity_32warps_${accesses}accesses" 0 32 0 0 1 "$accesses"
# done

echo ""
echo "=== Experiments Complete ==="
echo "Results saved to: $OUTPUT_DIR/results.csv"
echo "Individual NCU reports in: $OUTPUT_DIR/"
echo ""
echo "To analyze results, use:"
echo "  cat $OUTPUT_DIR/results.csv | column -t -s,"
echo ""
echo "Or import into spreadsheet/analysis tool for plotting."

