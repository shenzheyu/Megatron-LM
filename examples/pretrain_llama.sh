#!/bin/bash
 
# Runs the "345M" parameter model
 
export CUDA_DEVICE_MAX_CONNECTIONS=1
 
GPUS_PER_NODE=4
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DATA_PATH="/workspace/dataset/BookCorpusDataset_text_document"
VOCAB_FILE="/workspace/dataset/gpt2-vocab.json"
MERGE_FILE="/workspace/dataset/gpt2-merges.txt"
 
DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
 

# LLaMa2 7b
LLAMA_ARGS="
    --no-async-tensor-model-parallel-allreduce \
    --num-layers 4 \
    --hidden-size 2048 \
    --num-attention-heads 32 \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --position-embedding-type rope \
    --swiglu \
    --ffn-hidden-size 11008\
    --disable-bias-linear \
    --normalization RMSNorm \
    --layernorm-epsilon 1e-6 \
    --micro-batch-size 4 \
    --global-batch-size 4 \
    --lr 0.00015 \
    --train-iters 100 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --no-gradient-accumulation-fusion \
    --fp16 \
    --tensor-model-parallel-size $WORLD_SIZE \
    --seed 3407
"


DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --split 949,50,1
"
 
OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 1
"
 
torchrun $DISTRIBUTED_ARGS pretrain_llama.py \
    $LLAMA_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl #\
    # --save $CHECKPOINT_PATH \
    # --load $CHECKPOINT_PATH
 
