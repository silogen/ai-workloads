# Tutorial 07: Fine-tuning Wan 2.2 for Video Generation

This tutorial shows how to fine-tune Wan 2.2 models for custom video generation using the DiffSynth framework. Wan 2.2 is a state-of-the-art text-to-video generation model available in 5B and 14B parameter versions. We'll demonstrate both LoRA (Low-Rank Adaptation) and full parameter fine-tuning approaches, using the Disney VideoGeneration Dataset as an example.

The tutorial covers the complete pipeline: downloading models and datasets to cluster MinIO storage, setting up the training environment, running fine-tuning jobs with multi-GPU support, and uploading trained checkpoints for future use.

We'll start with the 5B parameter model using LoRA fine-tuning, which provides a good balance between training efficiency and model quality. The approach can be scaled up to the 14B model or full parameter training for more demanding use cases.

## 1. Setup

Follow the setup in the [tutorial pre-requisites section](./tutorial-prereqs.md).

## 2. Download model and dataset

We'll use the workloads to download the Wan 2.2 model and the Disney VideoGeneration Dataset to cluster MinIO storage.

### Download Wan 2.2 5B Model

First, download the Wan 2.2 5B parameter text-to-video model:

```bash
helm template workloads/download-huggingface-model-to-bucket/helm \
  -f workloads/download-huggingface-model-to-bucket/helm/overrides/tutorial-07-wan2-2-ti2v-5b.yaml \
  --name-template download-wan2-2-ti2v-5b \
  | kubectl apply -f -
```

### Download Disney VideoGeneration Dataset

Next, download and preprocess the Disney VideoGeneration Dataset (Steamboat Willy):

```bash
helm template workloads/download-data-to-bucket/helm \
  -f workloads/download-data-to-bucket/helm/overrides/tutorial-07-disney-dataset.yaml \
  --name-template download-disney-dataset \
  | kubectl apply -f -
```

Monitor the downloads using [k9s or kubectl logs](./tutorial-prereqs.md#monitoring-progress-logs-and-gpu-utilization-with-k9s). The model download includes:
- Main diffusion model (3 safetensors files, ~18.7 GiB)
- VAE model (Wan2.2_VAE.pth, ~2.6 GiB)
- Text encoder (T5-XXL, ~11 GiB)
- Configuration and tokenizer files

## 3. Interactive exploration (optional)

For testing and experimentation, you can launch an interactive version that provides a ready environment without automatic training:

```bash
helm template workloads/media-finetune-wan/helm \
  --name-template wan-finetune-interactive \
  | kubectl apply -f -
```

Connect to the interactive pod:
```bash
kubectl exec -it wan-finetune-interactive -- /bin/bash
```

This gives you access to explore the DiffSynth framework, examine the downloaded models and data, and test training configurations manually.

## 4. LoRA fine-tuning on multiple GPUs

LoRA (Low-Rank Adaptation) provides efficient fine-tuning by updating only a small number of parameters. This approach is recommended for most use cases as it requires less compute resources while maintaining good performance.

### Launch 5B LoRA Fine-tuning

Run LoRA fine-tuning on the 5B model using our optimized configuration:

```bash
helm template workloads/media-finetune-wan/helm \
  -f workloads/media-finetune-wan/helm/overrides/tutorial-07-5b-lora.yaml \
  --name-template wan-finetune-5b-lora \
  | kubectl apply -f -
```

This configuration uses:
- **4 GPUs** with **32 CPU cores** for reasonable compilation performance
- **LoRA rank 32** for good parameter efficiency
- **DeepSpeed ZeRO Stage 2** for distributed training
- **BF16 precision** for memory efficiency
- **5 epochs** with gradient accumulation

### Monitor training progress

Follow the training progress using:

```bash
# Watch job status
kubectl get jobs -w

# View training logs (replace with your actual job name)
kubectl logs job/wan-finetune-5b-lora -f

# Check GPU utilization with k9s
k9s
```

Training phases you'll observe:
1. **Installation**: DiffSynth framework setup
2. **Resource Download**: Model and dataset download from MinIO
3. **Compilation**: PyTorch model compilation for AMD GPUs
4. **Training**: 5 epochs with progress bars
5. **Upload**: Checkpoint upload to MinIO

Total training time: ~90-120 minutes depending on cluster load.

## 5. Scale to larger configurations

For more advanced use cases, different configurations can be explored.

### 14B Model Fine-tuning

For higher quality results, use the 14B parameter model by using the relevant configuration from among the override files inside the workload.

### Full Parameter Fine-tuning

For maximum customization, use the relevant override file for `architecture: "full"` instead of LoRA. This requires:
- More GPU resources (4+ GPUs recommended)
- Increased memory allocation
- Longer training time but potentially better results

## 6. Working with checkpoints

Trained checkpoints are automatically uploaded to MinIO with organized paths:

```
default-bucket/models/Wan-AI/Wan2.2-TI2V-5B/Wan2.2-TI2V-5B_lora/20250925-141325/
├── epoch-0.safetensors
├── epoch-1.safetensors
├── epoch-2.safetensors
├── epoch-3.safetensors
├── epoch-4.safetensors
├── adapter_config.json
├── adapter_model.safetensors
└── training_args.bin
```

### Using trained models

The checkpoints can be used for:

1. **Further fine-tuning**: Resume training from any epoch
2. **Inference deployment**: Load for video generation
3. **Model evaluation**: Compare different configurations
4. **Checkpoint merging**: Combine LoRA weights with base model

### Download checkpoints locally

```bash
# Setup MinIO client (if not already configured)
mc alias set minio-cluster http://minio.cluster.local

# Download specific checkpoint
mc cp --recursive \
  minio-cluster/default-bucket/models/Wan-AI/Wan2.2-TI2V-5B/Wan2.2-TI2V-5B_lora/20250925-141325/ \
  ./local-checkpoints/
```

## 7. Hyperparameter tuning

Experiment with different LoRA ranks to find optimal configurations:

```bash
run_id=wan-lora-sweep
for rank in 8 16 32 64 128; do
  name="wan-5b-lora-r$rank-$run_id"
  helm template workloads/media-finetune-wan/helm \
    -f workloads/media-finetune-wan/helm/overrides/tutorial-07-5b-lora.yaml \
    --name-template $name \
    --set finetune_config.lora_rank=$rank \
    | kubectl apply -f -
done
```

This creates parallel jobs testing different LoRA ranks, allowing you to compare training efficiency and model quality.

## 8. Advanced configurations

### Custom datasets

To use your own video dataset:

1. **Prepare data**: Organize videos and captions in the required format
2. **Upload to MinIO**: Use the data upload workload
3. **Update configuration**: Modify `datasetId` and paths in your override file
4. **Adjust parameters**: Tune learning rate, batch size, and epochs based on your data size

### Multi-node training

For very large models or datasets, distribute across multiple nodes:

```yaml
resources:
  cpu: 64
  gpus: 8  # Use all 8 GPUs per node
  memory: 512Gi
```

### Memory optimization

For memory-constrained scenarios:

```yaml
finetune_config:
  # Use gradient checkpointing
  gradient_checkpointing: true
  # Reduce batch size
  train_batch_size: 8
  # Increase gradient accumulation
  gradient_accumulation_steps: 8
```

## 9. Troubleshooting

### Common issues and solutions

**Compilation taking too long**:
- Increase CPU allocation to 32+ cores
- Use at least 3-4 GPUs for better parallelization

**Out of memory errors**:
- Reduce batch size: `train_batch_size: 8`
- Enable gradient checkpointing
- Use smaller LoRA rank: `lora_rank: 16`

**Slow model download**:
- Check MinIO cluster connectivity
- Verify bucket credentials and permissions

**Training divergence**:
- Lower learning rate: `learning_rate: 1e-5`
- Increase warmup steps
- Use different noise schedules

### Monitoring resources

```bash
# Check GPU usage across cluster
kubectl top nodes

# View detailed pod resource usage
kubectl describe pod wan-finetune-5b-lora-xyz

# Monitor job events
kubectl describe job wan-finetune-5b-lora
```

## 10. Next steps

After successful fine-tuning:

1. **Deploy for inference**: Use trained checkpoints in inference workloads
2. **Quality evaluation**: Generate test videos and evaluate results
3. **Dataset expansion**: Add more diverse training data
4. **Architecture experiments**: Try different model variants and training strategies

The fine-tuned Wan 2.2 models can be integrated into video generation pipelines, content creation tools, or further research projects focusing on controllable video synthesis.

## Configuration files reference

The tutorial uses these override configurations:

- `workloads/download-huggingface-model-to-bucket/helm/overrides/tutorial-07-wan2-2-ti2v-5b.yaml`: Model download configuration
- `workloads/download-data-to-bucket/helm/overrides/tutorial-07-disney-dataset.yaml`: Dataset download configuration
- `workloads/media-finetune-wan/helm/overrides/tutorial-07-5b-lora.yaml`: Optimized 5B LoRA fine-tuning configuration

Each configuration includes detailed comments explaining the parameter choices and trade-offs for different use cases.
