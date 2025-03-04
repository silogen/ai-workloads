./llama-gguf-split --merge \
    $(find . -name "*00001-of-*.gguf" | head -1) \
    ${MODEL_TAG}.merged.gguf
