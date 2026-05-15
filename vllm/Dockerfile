FROM vllm/vllm-openai:latest

ARG HF_TOKEN
ENV HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
ENV VLLM_ENABLE_CUDA_COMPATIBILITY=1

# Pre-download the model cache at build time
# The vllm image already has hf installed
RUN hf download unsloth/gemma-3-12b-it-FP8-Dynamic

EXPOSE 8000

CMD ["--model", "unsloth/gemma-3-12b-it-FP8-Dynamic", "--port", "8000", "--host", "0.0.0.0", "--trust-remote-code", "--max-model-len", "4096"]
