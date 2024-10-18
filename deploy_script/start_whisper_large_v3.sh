#!/bin/bash

source ~/.zshrc

unset http_proxy
unset https_proxy

CUDA_VISIBLE_DEVICES=3 /app/server \
	-pc \
	-pr \
	--language chinese \
	-m /models/alm/ggml/ggml-large-v3.bin \
	--host 0.0.0.0 \
	--port 8132


