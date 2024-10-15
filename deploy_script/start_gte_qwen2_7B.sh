
#!/bin/bash

source ~/.zshrc
conda activate sglang-infer
unset http_proxy
unset https_proxy

CUDA_VISIBLE_DEVICES=1 python -m sglang.launch_server \
--served-model-name Qwen/EMB \
--model-path /models/embed/gte-Qwen2-7B-instruct \
--is-embedding \
--host 0.0.0.0 \
--port 8112

