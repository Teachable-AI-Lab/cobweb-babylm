#!/bin/bash

MODEL_PATH=$1
MODEL_BASENAME=$(basename $MODEL_PATH)

python3 -m lm_eval --model cobweb \
    --model_args load_model=True,load_file=$MODEL_PATH,window=10,bidirection=False, \
    --tasks wsc273,wikitext \
    --device cpu \
    --output_path results/blimp/${MODEL_BASENAME}/blimp_results.json \
    # --batch_size 128

    # --model_args pretrained=$MODEL_PATH \
    # --trust_remote_code \
    # --batch_size 128 \

                 # keep_stop_word=True,stop_word_as_anchor=True, \
                 # load_file=$MODEL_PATH,nodes_pred=1000, \
                 # used_cores=120,device=cpu \