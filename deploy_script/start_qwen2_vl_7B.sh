
#!/bin/bash

source ~/.zshrc
conda activate vllm-infer
unset http_proxy
unset https_proxy

CUDA_VISIBLE_DEVICES=0 vllm serve /models/vlm/qwen/Qwen2-VL-7B-Instruct \
--served-model-name Qwen/Qwen2-VL-7B-Instruct  \
--host 0.0.0.0 \
--port 8022 \
--max-model-len 7920 \
--gpu-memory-utilization 0.95
