#!/usr/bin/env bash
# Control script for reproducing all six experiments in RLAIF-V.
# Each experiment follows the pipeline: diverse generation -> divide & conquer ->
# automatic checking -> pair construction -> DPO fine-tuning.
# Edit the variables below if paths are different.

set -e

# ==== 基础配置 ====
BASE_MODEL="liuhaotian/llava-v1.5-7b"   # 生成模型
QUES_DIR="./examples"                   # 问题文件目录
QUES_FILE="detail_test_30_input"       # 问题文件名（不含 .jsonl）
ANS_DIR="./results"                    # 实验结果保存目录
START=0
END=-1
NUM_GPUS=8

# 标注模型
LABELER_7B="liuhaotian/llava-v1.5-7b"
LABELER_13B="liuhaotian/llava-v1.5-13b"
LABELER_34B="openbmb/NeXT-34B"

# 实验参数
CANDIDATES=(2 4 8 16)
LABELERS=("$LABELER_7B" "$LABELER_13B" "$LABELER_34B")
SPLITS=("syntax" "heuristic" "hybrid")
LEARNING_RATES=(1e-5 5e-5 1e-4)
INTER_N=(4 4 8 8)
INTER_LABEL=("$LABELER_7B" "$LABELER_34B" "$LABELER_7B" "$LABELER_34B")

# ==== 辅助函数 ====
apply_split_strategy() {
    local name=$1
    local target="utils/diff_lib.py"
    local src="utils/split_strategies/${name}.py"
    if [[ ! -f $src ]]; then
        echo "Unknown split strategy: $name" >&2
        exit 1
    fi
    head -n 47 "$target" > "${target}.tmp"
    cat "$src" >> "${target}.tmp"
    tail -n +53 "$target" >> "${target}.tmp"
    mv "${target}.tmp" "$target"
}

run_generation() {
    local n=$1
    local save_dir=$2
    torchrun --nnodes=1 --nproc_per_node=${NUM_GPUS} \
        --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
        ./muffin/llava15_gen_data.py \
        --checkpoint "$BASE_MODEL" \
        --ds_name ${QUES_DIR}/${QUES_FILE}.jsonl \
        --answer_file ${save_dir}/diverse_gen_llava15_${QUES_FILE}.jsonl \
        --max_sample -1 \
        --start_pos $START --end_pos $END \
        --repeat $n \
        --max_tokens 512 \
        --num-workers 5 \
        --batch-size 8 \
        --temperature 0.7
}

run_divide() {
    local save_dir=$1
    bash ./script/data_gen/divide_and_conquer/llama3_8b_divide_and_conquer.sh \
        ${save_dir}/diverse_gen_llava15_${QUES_FILE} 0 -1 8 ${NUM_GPUS}
}

run_autocheck() {
    local labeler=$1
    local save_dir=$2
    local check_file=diverse_gen_llava15_${QUES_FILE}.s0-e-1.llama3-8b_divide.gq.qas.jsonl
    bash ./script/data_gen/omnilmm/omnilmm_autocheck.sh \
        "$labeler" "$save_dir" "$save_dir" "$check_file" 0 -1 ${NUM_GPUS}
}

run_pairs() {
    local save_dir=$1
    local gq_file=diverse_gen_llava15_${QUES_FILE}.s0-e-1.llama3-8b_divide.gq.jsonl
    local fb_file=autocheck_omni_0--1_diverse_gen_llava15_${QUES_FILE}.s0-e-1.llama3-8b_divide.gq.qas.jsonl
    bash ./script/data_gen/construct_pairs.sh \
        ${save_dir}/${fb_file} \
        ${save_dir}/${gq_file} \
        2
    python ./utils/get_pairs_filter_shorten.py \
        --path ${save_dir}/${fb_file%.jsonl}_pair_diff1_samp2.jsonl \
        --save_path ${save_dir}/${fb_file%.jsonl}_pair_diff1_samp2_balanceshort.jsonl
}

run_train() {
    local lr=$1
    local save_dir=$2
    deepspeed ./muffin/train/train_llava15.py \
        --deepspeed ./script/zero2.json \
        --model_name_or_path $BASE_MODEL \
        --data_dir ./RLAIF-V-Dataset_logps/ \
        --image_folder not_used \
        --vision_tower openai/clip-vit-large-patch14-336 \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --fully_tune True \
        --image_aspect_ratio pad \
        --bf16 True \
        --mm_projector_type mlp2x_gelu \
        --mm_vision_select_layer -2 \
        --output_dir ${save_dir}/checkpoints \
        --num_train_epochs 10 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 1 \
        --evaluation_strategy "no" \
        --save_strategy "steps" \
        --save_steps 167 \
        --save_total_limit 50 \
        --data_source_names '' \
        --data_source_weights 1 \
        --max_steps 2672 \
        --learning_rate $lr \
        --weight_decay 0.01 \
        --warmup_ratio 0.05 \
        --lr_scheduler_type "cosine" \
        --logging_steps 2 \
        --logging_dir ${save_dir}/log \
        --tf32 True \
        --model_max_length 2048 \
        --gradient_checkpointing True \
        --lazy_preprocess True \
        --task DPO \
        --report_to wandb \
        --run_name $(basename $save_dir) \
        --dataloader_num_workers 16 \
        --dpo_use_average False \
        --dpo_token_weighted False \
        --dpo_token_weight 1.0 \
        --dpo_beta 0.1
}

run_full() {
    local n=$1
    local labeler=$2
    local split=$3
    local lr=$4
    local tag=$5
    local exp_dir=${ANS_DIR}/${tag}
    mkdir -p "$exp_dir"

    apply_split_strategy "$split"
    run_generation "$n" "$exp_dir"
    run_divide "$exp_dir"
    run_autocheck "$labeler" "$exp_dir"
    run_pairs "$exp_dir"
    run_train "$lr" "$exp_dir"
}

experiment1() {
    for n in "${CANDIDATES[@]}"; do
        run_full "$n" "$LABELER_34B" "syntax" "5e-5" "exp1_n${n}"
    done
}

experiment2() {
    for lab in "${LABELERS[@]}"; do
        run_full 8 "$lab" "syntax" "5e-5" "exp2_$(basename $lab)"
    done
}

experiment3() {
    for sp in "${SPLITS[@]}"; do
        run_full 8 "$LABELER_34B" "$sp" "5e-5" "exp3_${sp}"
    done
}

experiment4() {
    for lr in "${LEARNING_RATES[@]}"; do
        run_full 8 "$LABELER_34B" "syntax" "$lr" "exp4_lr${lr}"
    done
}

experiment5() {
    for idx in ${!INTER_N[@]}; do
        n=${INTER_N[$idx]}
        lab=${INTER_LABEL[$idx]}
        run_full "$n" "$lab" "syntax" "5e-5" "exp5_n${n}_$(basename $lab)"
    done
}

experiment6() {
    run_full 8 "$LABELER_34B" "syntax" "5e-5" "exp6_final"
}

# 执行所有实验
experiment1
experiment2
experiment3
experiment4
experiment5
experiment6

