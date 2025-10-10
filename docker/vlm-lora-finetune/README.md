# Clipora - Low Rank Adapter Fine Tuning of OpenCLIP

Original repo used as base: https://github.com/awilliamson10/clipora/tree/main/clipora

Added Dockerfile, AMD GPU support and some changes like ability to load model with finetuned LORA adapters.

Also added a quick-and-dirty API that can be used for demoing, see [README_api.md](./README_api.md).

## Build

```
docker build -t tag .
docker push tag
```

# ORIGINAL README:

## Main Features

- Fine tune OpenCLIP models via LoRA

### TODOS

- [x] Adapter saving
- [ ] Merging adapters with the original model
- [ ] Inferencing with the adapter
- [ ] Add better documentation
- [ ] Support for more logging

# References

The links below have most of the information/implmentation here in much cleaner ways, all credit to them, I just frankensteined them together to make this.

- https://github.com/cloneofsimo/lora
- https://arxiv.org/abs/2106.09685
- https://github.com/facebookresearch/ov-seg
- https://huggingface.co/blog/lora
- https://github.com/huggingface/peft
- https://github.com/KyanChen/MakeMultiHeadNaive
