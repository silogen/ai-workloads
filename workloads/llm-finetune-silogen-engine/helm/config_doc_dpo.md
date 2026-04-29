# Fine-tuning config structure and parameters for DPO

This document describes the structure of the DPO fine-tuning configuration, and the parameters and values that can be defined there.

See the fine-tuning config section [this config file](overrides/tiny-llama-dpo-full-param.yaml) for an example of a valid configuration.
See the various sub-configs for their options. Additional properties are not allowed.

**Top-level properties:**

| Property | Type | Required | Possible values | Default | Description |
| -------- | ---- | -------- | --------------- | ------- | ----------- |
| data_conf | `object` | ✅ | [ChatTrainValidConfig](#chattrainvalidconfig) |  | The data input config |
| training_args | `object` | ✅ | [SilogenDPOConfig](#silogendpoconfig) |  | TRL DPOTrainerArguments with some restrictions |
| batchsize_conf | `object` | ✅ | [BatchsizeConfig](#batchsizeconfig) |  | Batch size configuration |
| peft_conf | `object` | ✅ | [GenericPeftConfig](#genericpeftconfig) and/or [NoPeftConfig](#nopeftconfig) and/or [PretrainedPeftConfig](#pretrainedpeftconfig) |  | Adapter configuration |
| run_conf | `object` | ✅ | [RunConfig](#runconfig) |  | Model related configuration |
| method | `const` |  | `dpo` | `"dpo"` |  |
| overrides | `object` |  | [Overrides](#overrides) | `{"lr_multiplier": 1.0, "lr_batch_size_scaling": "none"}` | Override options to simplify the config interface |
| tracking | `object` or `null` |  | [FinetuningTrackingConfig](#finetuningtrackingconfig) | `null` | MLFlow tracking configuration |
| quant_conf | `object` |  | [BnBQuantizationConfig](#bnbquantizationconfig) and/or [NoQuantizationConfig](#noquantizationconfig) | `{"quantization_type": "no-quantization"}` | Quantization configuration |


---

# Definitions

## AutoSplitDataInput

Automatic validation split from the training data

#### Type: `object`

| Property | Type | Required | Possible values | Default | Description |
| -------- | ---- | -------- | --------------- | ------- | ----------- |
| type | `const` | ✅ | `AUTO_SPLIT` |  |  |
| data_type | `string` |  | string | `"ChatConversation"` | Generally, the data_type is automatically set based on the experiment config method. |
| ratio | `number` |  | number | `0.2` | Ratio of the training data to use for validation |
| seed | `integer` |  | integer | `1289525893` | Seed for the random number generator for splitting |

## BatchsizeConfig

Config for determining the total batch size

Total batch size is the effective batch size for the complete training run. It is equal to
number of processes * per-device batch size * accumulation.

The maximum batch size per device is the maximum batch size that can be accommodated on a single device.
This mostly limited by the memory capacity of the device.

#### Type: `object`

| Property | Type | Required | Possible values | Default | Description |
| -------- | ---- | -------- | --------------- | ------- | ----------- |
| total_train_batch_size | `integer` | ✅ | integer |  | The total batch size for the training run |
| max_per_device_train_batch_size | `integer` | ✅ | integer |  | The maximum training batch size per device |
| per_device_eval_batch_size | `integer` or `null` |  | integer | `null` | The maximum eval batch size per device, if not given, will use same as training batch size |

## BnBQuantizationConfig

Bits and Bytes configuration

The options are from the BitsAndBytes config,
see: https://huggingface.co/docs/transformers/en/main_classes/quantization#transformers.BitsAndBytesConfig

#### Type: `object`

| Property | Type | Required | Possible values | Default | Description |
| -------- | ---- | -------- | --------------- | ------- | ----------- |
| quantization_type | `const` |  | `bits-and-bytes` | `"bits-and-bytes"` |  |
| load_in_8bit | `boolean` |  | boolean | `false` |  |
| load_in_4bit | `boolean` |  | boolean | `false` |  |
| llm_int8_threshold | `number` |  | number | `6.0` |  |
| llm_int8_skip_modules | `array` or `null` |  | string | `null` |  |
| llm_int8_enable_fp32_cpu_offload | `boolean` |  | boolean | `false` |  |
| llm_int8_has_fp16_weight | `boolean` |  | boolean | `false` |  |
| bnb_4bit_compute_dtype | `string` or `null` |  | string | `null` |  |
| bnb_4bit_quant_type | `const` |  | `fp4` and/or `nf4` | `"fp4"` |  |
| bnb_4bit_use_double_quant | `boolean` |  | boolean | `false` |  |
| bnb_4bit_quant_storage | `string` or `null` |  | string | `null` |  |

## ChatTemplateName

Chat template to use.

#### Type: `string`

**Possible Values:** `mistral-with-system` or `chat-ml` or `poro` or `keep-original` or `simplified-llama31`

## ChatTrainValidConfig

Training time data configuration

Always defines some DataInput for training data and can include validation DataInput, though a trivial NoneDataInput
is also allowed for the validation side.

Additionally includes chat template and padding configurations, as those are part of the data input pipeline.

#### Type: `object`

| Property | Type | Required | Possible values | Default | Description |
| -------- | ---- | -------- | --------------- | ------- | ----------- |
| training_data | `object` | ✅ | [ConcatenationDataInput](#concatenationdatainput) and/or [WeightedMixDataInput](#weightedmixdatainput) |  |  |
| validation_data | `object` | ✅ | [AutoSplitDataInput](#autosplitdatainput) and/or [ConcatenationDataInput](#concatenationdatainput) and/or [NoneDataInput](#nonedatainput) |  |  |
| chat_template_name | `string` |  | [ChatTemplateName](#chattemplatename) | `"mistral-with-system"` |  |
| padding_side | `string` |  | string | `"right"` | Padding side, right is usually right. |
| missing_pad_token_strategy | `string` |  | [MissingPadTokenStrategy](#missingpadtokenstrategy) | `"bos-repurpose"` | See the MissingPadTokenStrategys for descriptions of the options |

## ConcatenationDataInput

A simple list of datasets

These are simply concatenated, the same as sampling all with equal weight.

The datasets themselves need to be in the finetuning supported JSONL formats.
For SFT this means lines:

    {"messages": [{"content": "string", "role": "string"}]}

For DPO this means lines of:
    {
       "prompt_messages": [{"content": "string", "role": "string"}],
       "chosen_messages": [{"content": "string", "role": "string"}],
        "rejected_messages": [{"content": "string", "role": "string"}]
    }

#### Type: `object`

| Property | Type | Required | Possible values | Default | Description |
| -------- | ---- | -------- | --------------- | ------- | ----------- |
| type | `const` | ✅ | `CONCATENATION` |  |  |
| datasets | `array` | ✅ | [DatasetDefinition](#datasetdefinition) |  |  |
| data_type | `string` |  | string | `"ChatConversation"` | Generally, the data_type is automatically set based on the experiment config method. |

## DatasetDefinition

Define how to load a dataset

#### Type: `object`

| Property | Type | Required | Possible values | Description |
| -------- | ---- | -------- | --------------- | ----------- |
| path | `string` | ✅ | string | Local path to a JSONL file in the finetuning data format |

## FinetuningTrackingConfig

Settings that define how run details are logged

#### Type: `object`

| Property | Type | Required | Possible values | Default | Description |
| -------- | ---- | -------- | --------------- | ------- | ----------- |
| mlflow_server_uri | `string` | ✅ | string |  | MLflow server URI. Can be local path. |
| experiment_name | `string` | ✅ | string |  | Experiment name that is used for MLFlow tracking. |
| hf_mlflow_log_artifacts | `string` |  | string | `"False"` | Whether to store model artifacts in MLFlow. |

## GenericPeftConfig

Config for any new initialized PEFT Adapter

See https://huggingface.co/docs/peft/tutorial/peft_model_config for the possible kwargs
and https://github.com/huggingface/peft/blob/v0.7.1/src/peft/utils/peft_types.py for the types.

Example:

    >>> loaded_data = {'peft_type':'LORA', 'task_type': 'CAUSAL_LM',
    ...         'peft_kwargs': {'r': 32, 'target_modules': ['v_proj']}}
    >>> generic_conf = GenericPeftConfig(**loaded_data)
    >>> generic_conf.get_peft_config()
    LoraConfig(task_type=<TaskType.CAUSAL_LM: 'CAUSAL_LM'>, peft_type=<PeftType.LORA: 'LORA'>, ...)

#### Type: `object`

| Property | Type | Required | Possible values | Default | Description |
| -------- | ---- | -------- | --------------- | ------- | ----------- |
| peft_type | `string` | ✅ | [PeftType](#pefttype) |  |  |
| task_type | `string` |  | [TaskType](#tasktype) | `"CAUSAL_LM"` |  |
| peft_kwargs | `object` |  | object |  |  |

## MissingPadTokenStrategy

Specifies the available missing pad token strategies.

We've shown in a small set of experiments that repurposing EOS can start to hurt performance
while the other options seem to work equally well.

Repurposing EOS is the default in many online sources, but it is actually a bad idea if we want to predict
EOS, as all the pad_token_ids get ignored in loss computation, and thus the model does not learn to predict
the end of the text. However, for models that have additional tokens for end of message, end of turn, etc.
this is not so dangerous.

Repurposing BOS is similar to repurposing EOS, but since we do not need to predict BOS, this may be more sensible.

Repurposing UNK can work with tokenizers that never produce UNKs in normal data (e.g. Mistral tokenizers should have
a byte fall-back so that everything can be tokenized).

UNK_CONVERT_TO_EOS uses a hack where the unk_token_id is initially used for padding, but in the collation phase the
input-side UNKs (padding) gets set to EOS, so that the input-side padding looks like EOS. On the output-side, the
UNKs (padding) still gets ignored. NOTE: This will leave the tokenizer's pad_token_id set to the unk_token_id; so
any subsequent use of the model where padding is involved should somehow explicitly set the pad_token_id again.

#### Type: `string`

**Possible Values:** `eos-repurpose` or `bos-repurpose` or `unk-repurpose` or `unk-convert-to-eos`

## ModelArguments

These are passed to AutoModelForCausalLM.from_pretrained

See parameter docstrings and help at:
https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained
See below in "Parameters for big model inference" too, it affects training too. Also note that this link takes you
to the transformers main branch version - be sure to compare with the installed version of transformers (that keeps
changing over time, and it is difficult to keep this docstring up to date, so we wanted to link to the latest here).

Some important parameters to consider are:

- device_map :
    A map that specifies where each submodule should go. It doesn’t need to be refined to each parameter/buffer
    name, once a given module name is inside, every submodule of it will be sent to the same device. If we only pass
    the device (e.g., "cpu", "cuda:1", "mps", or a GPU ordinal rank like 1) on which the model will be allocated,
    the device map will map the entire model to this device. Passing device_map = 0 means put the whole model on GPU
    0.
- attn_implementation :
    The attention implementation to use in the model (if relevant). Can be any of "eager" (manual implementation of
    the attention), "sdpa" (using F.scaled_dot_product_attention), or "flash_attention_2" (using
    Dao-AILab/flash-attention). By default, if available, SDPA will be used for torch>=2.1.1. The default is
    otherwise the manual "eager" implementation.

NOTE:
    This does not include quantization_config. Quantization config is specified separately.

#### Type: `object`

| Property | Type | Required | Possible values | Default | Description |
| -------- | ---- | -------- | --------------- | ------- | ----------- |
| silogen_extra_args | `object` |  | object |  | Don't specify directly - this gathers additional args passed to the model |
| dtype | `const` or `string` |  | `auto` and/or string | `"auto"` |  |
| pretrained_model_name_or_path | `string` or `null` |  | Format: [`path`](https://json-schema.org/understanding-json-schema/reference/string#built-in-formats) and/or string | `null` | Can be either:<br />- A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.<br />- A path to a *directory* containing model weights saved using `~PreTrainedModel.save_pretrained`.<br />- A path or url to a *tensorflow index checkpoint file*.<br />- A path or url to a model folder containing a *flax checkpoint file* in *.msgpack* format.<br />- `None` if you are both providing the configuration and state dictionary. |
| config | `string` or `null` |  | Format: [`path`](https://json-schema.org/understanding-json-schema/reference/string#built-in-formats) and/or string | `null` | Configuration for the model to use instead of an automatically loaded configuration.<br />Can be either an instance of a class derived from `PretrainedConfig`, or a string/path valid as input to `PretrainedConfig.from_pretrained`. |
| cache_dir | `string` or `null` |  | Format: [`path`](https://json-schema.org/understanding-json-schema/reference/string#built-in-formats) and/or string | `null` | Path to a directory in which a downloaded pretrained model configuration should be cached. |
| from_tf | `boolean` |  | boolean | `false` | Load the model weights from a TensorFlow checkpoint save file. |
| from_flax | `boolean` |  | boolean | `false` | Load the model weights from a Flax checkpoint save file. |
| ignore_mismatched_sizes | `boolean` |  | boolean | `false` | Whether or not to raise an error if some of the weights from the checkpoint do not have the same size as the weights of the model. |
| force_download | `boolean` |  | boolean | `false` | Whether or not to force the (re-)download of the model weights and configuration files. |
| proxies | `object` or `null` |  | object | `null` | A dictionary of proxy servers to use by protocol or endpoint. |
| output_loading_info | `boolean` |  | boolean | `false` | Whether or not to also return a dictionary containing missing keys, unexpected keys and error messages. |
| local_files_only | `boolean` |  | boolean | `false` | Whether or not to only look at local files (i.e., do not try to download the model). |
| token | `boolean` or `string` or `null` |  | boolean and/or string | `null` | The token to use as HTTP bearer authorization for remote files. |
| revision | `string` |  | string | `"main"` | The specific model version to use. It can be a branch name, a tag name, or a commit id. |
| attn_implementation | `string` or `null` |  | string | `null` | The attention implementation to use in the model. Can be any of 'eager', 'sdpa', 'flash_attention_2', or 'flash_attention_3'.<br />Accepts HF kernel references in the form: <namespace>/<repo_name>[@<revision>][:<kernel_name>] |
| device_map | `integer` or `object` or `string` or `null` |  | integer and/or object and/or string | `null` | A map that specifies where each submodule should go. |
| max_memory | `object` or `null` |  | object | `null` | A dictionary device identifier to maximum memory if using `device_map`. |
| tp_plan | `string` or `null` |  | string | `null` | A torch tensor parallel plan. Currently only accepts 'auto'. |
| tp_size | `string` or `null` |  | string | `null` | A torch tensor parallel degree. If not provided would default to world size. |
| offload_folder | `string` or `null` |  | Format: [`path`](https://json-schema.org/understanding-json-schema/reference/string#built-in-formats) and/or string | `null` | If the `device_map` contains any value 'disk', the folder where we will offload weights. |
| offload_buffers | `boolean` |  | boolean | `false` | Whether or not to offload the buffers with the model parameters. |
| subfolder | `string` |  | string | `""` | In case the relevant files are located inside a subfolder of the model repo on huggingface.co. |
| variant | `string` or `null` |  | string | `null` | If specified load weights from `variant` filename, e.g. pytorch_model.<variant>.bin. |
| use_safetensors | `boolean` or `null` |  | boolean | `null` | Whether or not to use `safetensors` checkpoints. |
| weights_only | `boolean` |  | boolean | `true` | Indicates whether unpickler should be restricted to loading only tensors and primitive types. |
| key_mapping | `object` or `null` |  | object | `null` | A potential mapping of the weight names if using a model on the Hub which is compatible to a Transformers architecture, but was not converted accordingly. |

## NoPeftConfig

A trivial config specifying that no peft is used

#### Type: `object`

| Property | Type | Required | Possible values | Description |
| -------- | ---- | -------- | --------------- | ----------- |
| peft_type | `const` | ✅ | `NO_PEFT` |  |

## NoQuantizationConfig

A marker not to use quantization

#### Type: `object`

| Property | Type | Required | Possible values | Default | Description |
| -------- | ---- | -------- | --------------- | ------- | ----------- |
| quantization_type | `const` |  | `no-quantization` | `"no-quantization"` |  |

## NoneDataInput

A special type for not using data e.g. in validation

#### Type: `object`

| Property | Type | Required | Possible values | Default | Description |
| -------- | ---- | -------- | --------------- | ------- | ----------- |
| type | `const` | ✅ | `NONE` |  |  |
| data_type | `string` |  | string | `"ChatConversation"` | Generally, the data_type is automatically set based on the experiment config method. |

## Overrides

Override options

These implement dynamic scaling for the learning rate.

#### Type: `object`

| Property | Type | Required | Possible values | Default | Description |
| -------- | ---- | -------- | --------------- | ------- | ----------- |
| lr_multiplier | `number` |  | number | `1.0` | Multiplier applied to the learning rate in the training_args |
| lr_batch_size_scaling | `string` |  | `none` `sqrt` `linear` | `"none"` | Scales the learning rate in the training_args by a factor derived from the total training batch size.             'none': No scaling.             'sqrt': Multiplies learning rate by square root of batch size (a classic scaling rule).             'linear': Multiplies learning rate by the batch size (a more modern scaling rule). |

## PeftType

Enum class for the different types of adapters in PEFT.

Supported PEFT types:
- PROMPT_TUNING
- MULTITASK_PROMPT_TUNING
- P_TUNING
- PREFIX_TUNING
- LORA
- ADALORA
- BOFT
- ADAPTION_PROMPT
- IA3
- LOHA
- LOKR
- OFT
- XLORA
- POLY
- LN_TUNING
- VERA
- FOURIERFT
- HRA
- BONE
- RANDLORA
- C3A

#### Type: `string`

**Possible Values:** `PROMPT_TUNING` or `MULTITASK_PROMPT_TUNING` or `P_TUNING` or `PREFIX_TUNING` or `LORA` or `ADALORA` or `BOFT` or `ADAPTION_PROMPT` or `IA3` or `LOHA` or `LOKR` or `OFT` or `POLY` or `LN_TUNING` or `VERA` or `FOURIERFT` or `XLORA` or `HRA` or `VBLORA` or `CPT` or `BONE` or `RANDLORA` or `TRAINABLE_TOKENS` or `C3A`

## PretrainedPeftConfig

PEFT adapter uses the config and initialisation from a pretrained adapter

#### Type: `object`

| Property | Type | Required | Possible values | Description |
| -------- | ---- | -------- | --------------- | ----------- |
| peft_type | `const` | ✅ | `PRETRAINED_PEFT` |  |
| name_or_path | `string` | ✅ | string | HF ID or path to the pretrained peft. |

## RunConfig

Experiment running configuration

#### Type: `object`

| Property | Type | Required | Possible values | Default | Description |
| -------- | ---- | -------- | --------------- | ------- | ----------- |
| model | `string` |  | string | `"/local_resources/basemodel"` | Local path to model to be fine-tuned. Normally this should be /local_resources/basemodel |
| model_args | `object` |  | [ModelArguments](#modelarguments) | `{"dtype": "auto", "pretrained_model_name_or_path": null, "config": null, "cache_dir": null, "from_tf": false, "from_flax": false, "ignore_mismatched_sizes": false, "force_download": false, "proxies": null, "output_loading_info": false, "local_files_only": false, "token": null, "revision": "main", "attn_implementation": null, "device_map": "auto", "max_memory": null, "tp_plan": null, "tp_size": null, "offload_folder": null, "offload_buffers": false, "subfolder": "", "variant": null, "use_safetensors": null, "weights_only": true, "key_mapping": null}` |  |
| tokenizer | `string` or `null` |  | string | `null` | Model HuggingFace ID, or path, or None to use the one associated with the model |
| use_fast_tokenizer | `boolean` |  | boolean | `true` | Use the Fast version of the tokenizer. The 'slow' version may be compatible with more features. |
| resume_from_checkpoint | `boolean` or `string` |  | boolean and/or string | `false` | Normally should be set to 'auto' to continue if a checkpoint exists.        Can set to True to always try to continue, False to never try, or a path to load from a specific path. |
| final_checkpoint_name | `string` |  | string | `"checkpoint-final"` | Name of final checkpoint. Should be left as default |
| determinism | `string` |  | `no` `half` `full` | `"no"` | Set the level of determinism in implementations. Deterministic implementations are not always available,            and when they are, they are usually slower than their non-deterministic counterparts. Recommended for            debugging only.            'no': No determinism.            'half': Prefer deterministic implementations.            'full': Only fully deterministic implementations, error out on operations that only have non-deterministic                    implementations. |

## SilogenDPOConfig

HuggingFace TRL DPOConfig as Config with additional SiloGen conventions

The list of training arguments is best available online (the version might not be up-to-date here):
https://huggingface.co/docs/transformers/v4.57.3/en/main_classes/trainer#transformers.TrainingArguments

Additionally, the DPOConfig has arguments specific to DPO training, which can be found here:
https://huggingface.co/docs/trl/v0.13.0/en/dpo_trainer#trl.DPOConfig

The object does a lot of things besides specifying the training configuration options (e.g. it
has computed properties like true training batch size etc.)

## TaskType

Enum class for the different types of tasks supported by PEFT.

Overview of the supported task types:
- SEQ_CLS: Text classification.
- SEQ_2_SEQ_LM: Sequence-to-sequence language modeling.
- CAUSAL_LM: Causal language modeling.
- TOKEN_CLS: Token classification.
- QUESTION_ANS: Question answering.
- FEATURE_EXTRACTION: Feature extraction. Provides the hidden states which can be used as embeddings or features
  for downstream tasks.

#### Type: `string`

**Possible Values:** `SEQ_CLS` or `SEQ_2_SEQ_LM` or `CAUSAL_LM` or `TOKEN_CLS` or `QUESTION_ANS` or `FEATURE_EXTRACTION`

## WeightedDatasetDefinition

Define a dataset, with a weight for sampling

#### Type: `object`

| Property | Type | Required | Possible values | Default | Description |
| -------- | ---- | -------- | --------------- | ------- | ----------- |
| path | `string` | ✅ | string |  | Local path to a JSONL file in the finetuning data format |
| sampling_weight | `number` |  | number | `1.0` |  |

## WeightedMixDataInput

A list of datasets where each is sampled by a certain weight

These datasets are interleaved based on the sampling weights. The resulting dataset is fully precomputed, upto
the point where every single sample in every dataset gets picked. This means that with small sampling weights,
it can take a lot of draws to see every sample from a dataset and so the resulting dataset can be very large.

The datasets themselves need to be in the finetuning supported JSONL formats.
For SFT this means lines:

    {"messages": [{"content": "string", "role": "string"}]}

For DPO this means lines of:
    {
       "prompt_messages": [{"content": "string", "role": "string"}],
       "chosen_messages": [{"content": "string", "role": "string"}],
        "rejected_messages": [{"content": "string", "role": "string"}]
    }

#### Type: `object`

| Property | Type | Required | Possible values | Default | Description |
| -------- | ---- | -------- | --------------- | ------- | ----------- |
| type | `const` | ✅ | `PRECOMPUTE_WEIGHTED_MIX` |  |  |
| datasets | `array` | ✅ | [WeightedDatasetDefinition](#weighteddatasetdefinition) |  |  |
| data_type | `string` |  | string | `"ChatConversation"` | Generally, the data_type is automatically set based on the experiment config method. |
| seed | `integer` |  | integer | `19851243` | Seed for the random number generator for interleaving draws |
