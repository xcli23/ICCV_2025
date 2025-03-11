#!/bin/bash

# export TRANSFORMERS_OFFLINE=1
cuda=0
dataset=paintings #paintings, art, anime, DB
white_model=promtist #vicuna-13b vicuna-7b llama2-7b promtist sft
black_model=dreamlike # sd1.5 dreamlike sdXL
optimizer=spsa #mmt spsa 
h=0.05
lr=1e-3 #1e-5 2e-5 5e-5 1e-4 1e-1 1e-2
batch_size=2    #1 --> single, other --> batch
epochs=20
train_samples=32 #16 32 64 128 256
n_directions=5
lora_rank=4 #4 8
metric=total #"aesthetic, clip, pick, total"
gene_image=True #True False
soft_train=True #True False
soft_epoch=20
soft_lr=0.1
mu=0.1
intrinsic_dim=10
n_prompt_tokens=5
soft_train_batches=16
soft_n_directions=1
random_proj=uniform # normal uniform
debug=False # True False
for seed in 14;do #14 42 81
    # cmd="python -m debugpy --listen 5678 --wait-for-client test.py --seed $seed"
    # 
    base_dir="./result/${dataset}/${white_model}_${black_model}_opt_${optimizer}_proj${random_proj}/samples_${train_samples}_batch${batch_size}_lora_rank${lora_rank}/ep${epochs}_dir${n_directions}_lr${lr}_h${h}"
    
    # 
    if [ "$soft_train" = "True" ]; then
        soft_params_dir="/soft_ep${soft_epoch}_lr${soft_lr}_mu${mu}/dim${intrinsic_dim}_tokens${n_prompt_tokens}_soft_n_dir${soft_n_directions}"
    else
        soft_params_dir=""
    fi
    
    # 
    output_path="${base_dir}/seed${seed}/output.txt"
    output_dir=$(dirname "$output_path")
    if [ ! -d "$output_dir" ]; then
        mkdir -p "$output_dir"
    fi

    # 
    soft_output_path="${output_dir}${soft_params_dir}/seed${seed}/output.txt"
    soft_output_dir=$(dirname "$soft_output_path")
    if [ ! -d "$soft_output_dir" ]; then
        mkdir -p "$soft_output_dir"
    fi



    if [ "$debug" = "True" ]; then
        cmd="python -m debugpy --listen 5679 --wait-for-client main_soft.py"
    else
        cmd="python main_soft.py"
    fi
    cmd="$cmd --cuda $cuda\
    --dataset $dataset\
    --train_samples $train_samples\
    --white_model $white_model\
    --black_model $black_model\
    --optimizer $optimizer\
    --h $h\
    --lr $lr\
    --metric $metric\
    --batch_size $batch_size\
    --epochs $epochs\
    --n_directions $n_directions\
    --output_dir $output_dir\
    --soft_output_dir $soft_output_dir\
    --gene_image $gene_image\
    --lora_rank $lora_rank\
    --debug $debug\
    --soft_train $soft_train\
    --soft_epoch $soft_epoch\
    --soft_lr $soft_lr\
    --mu $mu\
    --n_prompt_tokens $n_prompt_tokens\
    --random_proj $random_proj\
    --intrinsic_dim $intrinsic_dim\
    --soft_n_directions $soft_n_directions\
    --seed $seed > $soft_output_path 2>&1"
    echo $cmd
    eval $cmd
    echo "$cmd" >> "$soft_output_path"
done
# -m debugpy --listen 5679 --wait-for-client 