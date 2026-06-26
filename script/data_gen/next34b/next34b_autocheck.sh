#!/bin/bash
export PYTHONPATH=$PYTHONPATH:`realpath .`

model=$1
ans_dir=$2
ques_file=$3
start=${4:-0}
end=${5:--1}
gpus=${6:-1}

CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((gpus-1))) python llava_next_autocheck.py \
  --checkpoint $model \
  --ds_name ${ans_dir}/${ques_file} \
  --answer_file ${ans_dir}/autocheck_next34b_${start}-${end}_${ques_file} \
  --start_pos $start --end_pos $end --is_yesno
