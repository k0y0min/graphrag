FROM vllm/vllm-openai:latest

ARG HF_TOKEN
ENV HUGGING_FACE_HUB_TOKEN=$HF_TOKEN

# Pre-download the model cache at build time
# The vllm image already has hf installed
RUN hf download unsloth/gemma-3-12b-it-FP8-Dynamic

EXPOSE 8000
