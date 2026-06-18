#!/bin/bash
# ==============================================================================
# RGLS Hyperparameter Search Script
# 3-Phase Search: Coarse -> Fine -> Clip Residual
# ==============================================================================

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate wip

MODEL_PATH="/home/disk1/data/Dataset_Share/decapoda-research-llama-7B-hf"
LOG_DIR="logs_rgls_search"
mkdir -p ${LOG_DIR}

# Results files
PHASE1_RESULTS="rgls_phase1_results.csv"
PHASE2_RESULTS="rgls_phase2_results.csv"
PHASE3_RESULTS="rgls_phase3_results.csv"
FINAL_RESULTS="rgls_final_results.csv"

# Available GPUs
GPUS=(0 1)
NUM_GPUS=${#GPUS[@]}

# Fixed parameters
SPARSITY_RATIOS=(0.5 0.7)
SEED=0

# ==============================================================================
# Helper Functions
# ==============================================================================

# GPU tracking
declare -a GPU_PIDS=()
for ((i=0; i<NUM_GPUS; i++)); do
    GPU_PIDS[i]=0
done

get_available_gpu() {
    while true; do
        for ((i=0; i<NUM_GPUS; i++)); do
            local pid=${GPU_PIDS[i]}
            if [ ${pid} -eq 0 ] || ! kill -0 ${pid} 2>/dev/null; then
                wait ${pid} 2>/dev/null
                echo ${i}
                return 0
            fi
        done
        sleep 5
    done
}

run_single_experiment() {
    local gpu_id=$1
    local sparsity=$2
    local swap_ratio=$3
    local bandwidth=$4
    local alpha=$5
    local calib_batches=$6
    local clip_residual=$7
    local task_id=$8
    local total_tasks=$9
    local results_file=${10}
    
    local exp_name="sp${sparsity}_sr${swap_ratio}_bw${bandwidth}_a${alpha}_cb${calib_batches}_cr${clip_residual}"
    local log_file="${LOG_DIR}/${exp_name}.log"
    
    echo "[Task ${task_id}/${total_tasks}] GPU:${gpu_id} | ${exp_name}"
    
    CUDA_VISIBLE_DEVICES=${gpu_id} python main.py \
        --model ${MODEL_PATH} \
        --prune_method ria \
        --sparsity_ratio ${sparsity} \
        --adaptive \
        --refill_method swap \
        --swap_ratio ${swap_ratio} \
        --swap_candidate_bandwidth ${bandwidth} \
        --swap_alpha ${alpha} \
        --refill_calib_batches ${calib_batches} \
        --refill_clip_residual ${clip_residual} \
        --seed ${SEED} \
        > "${log_file}" 2>&1
    
    local exit_code=$?
    
    if [ ${exit_code} -eq 0 ]; then
        local ppl=$(grep "wikitext perplexity" "${log_file}" | awk '{print $NF}')
        if [ -n "$ppl" ]; then
            flock -x 200
            echo "${sparsity},${swap_ratio},${bandwidth},${alpha},${calib_batches},${clip_residual},${ppl}" >> ${results_file}
            flock -u 200
            echo "[Task ${task_id}/${total_tasks}] ✓ ${exp_name} PPL=${ppl}"
        fi
    else
        echo "[Task ${task_id}/${total_tasks}] ✗ Failed: ${exp_name}"
    fi
} 200>${LOG_DIR}/.lock

run_task_batch() {
    local results_file=$1
    shift
    local -a tasks=("$@")
    local total=${#tasks[@]}
    local idx=0
    
    while [ ${idx} -lt ${total} ]; do
        gpu_idx=$(get_available_gpu)
        gpu_id=${GPUS[${gpu_idx}]}
        
        # Parse task: sparsity:swap_ratio:bandwidth:alpha:calib_batches:clip_residual
        IFS=':' read -ra params <<< "${tasks[${idx}]}"
        
        run_single_experiment ${gpu_id} ${params[0]} ${params[1]} ${params[2]} ${params[3]} ${params[4]} ${params[5]} $((idx + 1)) ${total} ${results_file} &
        GPU_PIDS[${gpu_idx}]=$!
        
        idx=$((idx + 1))
        sleep 2
    done
    
    # Wait for all to complete
    for pid in "${GPU_PIDS[@]}"; do
        [ ${pid} -ne 0 ] && wait ${pid} 2>/dev/null
    done
    
    # Reset GPU PIDs
    for ((i=0; i<NUM_GPUS; i++)); do
        GPU_PIDS[i]=0
    done
}

# ==============================================================================
# PHASE 1: Coarse Search (12 experiments per sparsity = 24 total)
# ==============================================================================
echo ""
echo "========================================"
echo "PHASE 1: Coarse Search"
echo "========================================"

echo "sparsity,swap_ratio,bandwidth,alpha,calib_batches,clip_residual,ppl" > ${PHASE1_RESULTS}

# Build Phase 1 tasks
declare -a PHASE1_TASKS=()
for sparsity in "${SPARSITY_RATIOS[@]}"; do
    for swap_ratio in 0.002 0.005; do
        for bandwidth in 0.01 0.02; do
            for alpha in 0.3 0.5 0.7; do
                PHASE1_TASKS+=("${sparsity}:${swap_ratio}:${bandwidth}:${alpha}:8:0")
            done
        done
    done
done

echo "Phase 1 tasks: ${#PHASE1_TASKS[@]}"
echo "Starting Phase 1 at $(date)"

run_task_batch ${PHASE1_RESULTS} "${PHASE1_TASKS[@]}"

echo ""
echo "Phase 1 completed at $(date)"
echo ""
echo "=== Phase 1 Results ==="
column -t -s',' ${PHASE1_RESULTS}

# ==============================================================================
# PHASE 2: Fine Search (Top 3 per sparsity, 8 experiments each)
# ==============================================================================
echo ""
echo "========================================"
echo "PHASE 2: Fine Search"
echo "========================================"

echo "sparsity,swap_ratio,bandwidth,alpha,calib_batches,clip_residual,ppl" > ${PHASE2_RESULTS}

# Build Phase 2 tasks based on Phase 1 results
declare -a PHASE2_TASKS=()

for sparsity in "${SPARSITY_RATIOS[@]}"; do
    echo ""
    echo "Finding top 3 configs for sparsity=${sparsity}..."
    
    # Get top 3 for this sparsity (sorted by PPL ascending)
    top3=$(grep "^${sparsity}," ${PHASE1_RESULTS} | sort -t',' -k7 -n | head -3)
    
    echo "Top 3 for sparsity=${sparsity}:"
    echo "$top3"
    
    # For each top config, create fine-tuning tasks
    rank=0
    while IFS=',' read -r sp sr bw al cb cr ppl; do
        rank=$((rank + 1))
        echo "  Rank ${rank}: swap_ratio=${sr}, bandwidth=${bw}, alpha=${al}, PPL=${ppl}"
        
        # Calculate swap_ratio variants (4 points)
        sr_06=$(echo "${sr} * 0.6" | bc -l | xargs printf "%.3f")
        sr_08=$(echo "${sr} * 0.8" | bc -l | xargs printf "%.3f")
        sr_12=$(echo "${sr} * 1.2" | bc -l | xargs printf "%.3f")
        
        # Create tasks: 4 swap_ratio × 2 calib_batches = 8
        for new_sr in ${sr_06} ${sr_08} ${sr} ${sr_12}; do
            for new_cb in 8 16; do
                PHASE2_TASKS+=("${sparsity}:${new_sr}:${bw}:${al}:${new_cb}:0")
            done
        done
    done <<< "$top3"
done

echo ""
echo "Phase 2 tasks: ${#PHASE2_TASKS[@]}"
echo "Starting Phase 2 at $(date)"

run_task_batch ${PHASE2_RESULTS} "${PHASE2_TASKS[@]}"

echo ""
echo "Phase 2 completed at $(date)"
echo ""
echo "=== Phase 2 Results ==="
column -t -s',' ${PHASE2_RESULTS}

# ==============================================================================
# PHASE 3: Clip Residual Search (4 experiments per sparsity = 8 total)
# ==============================================================================
echo ""
echo "========================================"
echo "PHASE 3: Clip Residual Search"
echo "========================================"

echo "sparsity,swap_ratio,bandwidth,alpha,calib_batches,clip_residual,ppl" > ${PHASE3_RESULTS}

# Find global best for each sparsity from Phase 1 + Phase 2
declare -a PHASE3_TASKS=()

for sparsity in "${SPARSITY_RATIOS[@]}"; do
    echo ""
    echo "Finding best config for sparsity=${sparsity}..."
    
    # Combine Phase 1 and Phase 2 results, find best
    best_line=$(cat ${PHASE1_RESULTS} ${PHASE2_RESULTS} | grep "^${sparsity}," | sort -t',' -k7 -n | head -1)
    
    if [ -n "$best_line" ]; then
        IFS=',' read -r sp sr bw al cb cr ppl <<< "$best_line"
        echo "  Best: swap_ratio=${sr}, bandwidth=${bw}, alpha=${al}, calib_batches=${cb}, PPL=${ppl}"
        
        # Test clip_residual values
        for clip in 0 1.0 2.0 4.0; do
            PHASE3_TASKS+=("${sparsity}:${sr}:${bw}:${al}:${cb}:${clip}")
        done
    fi
done

echo ""
echo "Phase 3 tasks: ${#PHASE3_TASKS[@]}"
echo "Starting Phase 3 at $(date)"

run_task_batch ${PHASE3_RESULTS} "${PHASE3_TASKS[@]}"

echo ""
echo "Phase 3 completed at $(date)"
echo ""
echo "=== Phase 3 Results ==="
column -t -s',' ${PHASE3_RESULTS}

# ==============================================================================
# Final Summary
# ==============================================================================
echo ""
echo "========================================"
echo "FINAL SUMMARY"
echo "========================================"

# Combine all results
echo "sparsity,swap_ratio,bandwidth,alpha,calib_batches,clip_residual,ppl" > ${FINAL_RESULTS}
tail -n +2 ${PHASE1_RESULTS} >> ${FINAL_RESULTS}
tail -n +2 ${PHASE2_RESULTS} >> ${FINAL_RESULTS}
tail -n +2 ${PHASE3_RESULTS} >> ${FINAL_RESULTS}

echo ""
echo "=== Best Config per Sparsity ==="
for sparsity in "${SPARSITY_RATIOS[@]}"; do
    echo ""
    echo "Sparsity = ${sparsity}:"
    grep "^${sparsity}," ${FINAL_RESULTS} | sort -t',' -k7 -n | head -5 | \
        awk -F',' '{printf "  swap_ratio=%.3f, bw=%.3f, alpha=%.1f, cb=%d, clip=%.1f -> PPL=%.4f\n", $2, $3, $4, $5, $6, $7}'
done

echo ""
echo "========================================"
echo "All phases completed at $(date)"
echo "Results saved to:"
echo "  - ${PHASE1_RESULTS}"
echo "  - ${PHASE2_RESULTS}"
echo "  - ${PHASE3_RESULTS}"
echo "  - ${FINAL_RESULTS} (combined)"
echo "========================================"

