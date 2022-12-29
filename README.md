<h1 align="center"> <p>🤗 PET</p></h1>
<h3 align="center">
    <p>State-of-the-art Parameter-Efficient Tuning (PET) methods</p>
</h3>

Parameter-Efficient Tuning (PET) methods enable efficient adaptation of pre-trained language models (PLMs) to various downstream applications without fine-tuning all the model's parameters. Fine-tuning large-scale PLMs is often prohibitively costly. In this regard, PET methods only fine-tune a small number of (extra) model parameters, thereby greatly decreasing the computational and storage costs. Recent State-of-the-Art PET techniques achieve performance comparable to that of full fine-tuning. 

Seamlessly integrated with 🤗 Accelerate for large scale models leveraging PyTorch FSDP. 

Supported methods:

1. LoRA: [LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS](https://arxiv.org/pdf/2106.09685.pdf)
2. Prefix Tuning: [P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks](https://arxiv.org/pdf/2110.07602.pdf)
3. P-Tuning: [GPT Understands, Too](https://arxiv.org/pdf/2103.10385.pdf)
4. Prompt Tuning: [The Power of Scale for Parameter-Efficient Prompt Tuning](https://arxiv.org/pdf/2104.08691.pdf) 

## Getting started

```python
from transformers import AutoModelForSeq2SeqLM
from pet import get_pet_config, get_pet_model
model_name_or_path = "bigscience/mt0-large"
tokenizer_name_or_path = "bigscience/mt0-large"

config = {
    "pet_type":"LORA",
    "task_type":"SEQ_2_SEQ_LM",
    "r": 8,
    "lora_alpha": 32,
    "lora_dropout": 0.1
}
pet_config = get_pet_config(config)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
model = get_pet_model(model, pet_config)
model.print_trainable_parameters()
# output: trainable params: 2359296 || all params: 1231940608 || trainable%: 0.19151053100118282
```

## Use Cases

### Get comparable performance to full finetuning by adapting LLMs to downstream tasks using consumer hardware

GPU memory required for adapting LLMs on the few-shot dataset `ought/raft/twitter_complaints`. Here, settings considered
are full finetuning, PET-LoRA using plain PyTorch and  PET-LoRA using DeepSpeed with CPU Offloading. 

Hardware: Single A100 80GB GPU with CPU RAM above 64GB

|   Model         | Full Finetuning | PET-LoRA PyTorch  | PET-LoRA DeepSpeed with CPU Offloading |
| --------- | ---- | ---- | ---- |
| bigscience/T0_3B (3B params) | 47.14GB GPU / 2.96GB CPU  | 14.4GB GPU / 2.96GB CPU | 9.8GB GPU / 17.8GB CPU |
| bigscience/mt0-xxl (12B params) | OOM GPU | 56GB GPU / 3GB CPU | 22GB GPU / 52GB CPU |

Performance of PET-LoRA tuned `bigscience/T0_3B` on `ought/raft/twitter_complaints` leaderboard. 
A point to note is that we didn't try to sequeeze performance by playing around with input instruction templates, LoRA hyperparams and other training related hyperparams. Also, we didn't use the larger 13B mt0-xxl model.
So, we are already seeing comparable performance to SoTA with parameter effcient tuning. Also, the final checkpoint size is just `19MB` in comparison to `11GB` size of the backbone `bigscience/T0_3B` model.

|   Submission Name        | Accuracy |
| --------- | ---- |
| Human baseline (crowdsourced) |	0.897 |
| Flan-T5 | 0.892 |
| lora-t0-3b | 0.863 |

**Therefore, we can see that performance comparable to SoTA is achievable by PET methods with consumer hardware such as 16GB and 24GB GPUs.**

### Parameter Efficient Tuning of Diffusion Models [ToDo]

### Parameter Efficient Tuning of LLMs for RLHF components such as Ranker and Policy [ToDo]

### Save compute and storage even for medium and small models

Save storage by avoiding full finetuning of models on each of the downstream tasks/datasets,
With PET methods, users only need to store tiny checkpoints in the order of `MBs` all the while retaining 
performance comparable to full finetuning.

An example of using LoRA for the task of adaping `LayoutLMForTokenClassification` on `FUNSD` dataset is given in `~examples/PET_LoRA_LayoutLMForTokenClassification_on_FUNSD.py`. We can observe that with only `0.62 %` of parameters being trainable, we achieve performance (F1 0.777) comparable to full finetuning (F1 0.786) (without any hyerparam tuning runs for extracting more performance), and the checkpoint of this is only `2.8MB`.

Now, if there are `N` such datasets, just have these PET models one for each dataset and save a lot of storage without having to worry about the problem of catastrophic forgetting or overfitting of backbone/base model.


## PET + 🤗 Accelerate

PET models work with 🤗 Accelerate out of the box. Use 🤗 Accelerate for Distributed training on various hardware such as GPUs, Apple Silicon devices etc during training.
Use 🤗 Accelerate for inferencing on consumer hardware with small resources.

### Example of PET model distributed training using 🤗 Accelerate

### Example of PET model inference using 🤗 Accelerate


## Models support matrix

### Causal Language Modeling
|   Model         | LoRA | Prefix Tuning  | P-Tuning | Prompt Tuning  |
| --------- | ---- | ---- | ---- | ----  |
| GPT-2          | ✅  | ✅  | ✅  | ✅  |
| Bloom          | ✅  | ✅  | ✅  | ✅  |
| OPT            | ✅  | ✅  | ✅  | ✅  |
| GPT-Neo        | ✅  | ✅  | ✅  | ✅  |
| GPT-J          | ✅  | ✅  | ✅  | ✅  |

### Conditional Generation
|   Model         | LoRA | Prefix Tuning  | P-Tuning | Prompt Tuning  | 
| --------- | ---- | ---- | ---- | ---- |
| T5        | ✅   | ✅   | ✅   | ✅   |
| BART      | ✅   | ✅   | ✅   | ✅   |

### Sequence Classification
|   Model         | LoRA | Prefix Tuning  | P-Tuning | Prompt Tuning  | 
| --------- | ---- | ---- | ---- | ----  |
| BERT           | ✅  | ✅  | ✅  | ✅  |  
| RoBERTa        | ✅  | ✅  | ✅  | ✅  |
| GPT-2          | ✅  | ✅  | ✅  | ✅  | 
| Bloom          | ✅  | ✅  | ✅  | ✅  |   
| OPT            | ✅  | ✅  | ✅  | ✅  |
| GPT-Neo        | ✅  | ✅  | ✅  | ✅  |
| GPT-J          | ✅  | ✅  | ✅  | ✅  |
| Deberta        | ✅  |     | ✅  | ✅  |     
| Deberta-v2     | ✅  |     | ✅  | ✅  |    

### Token Classification
|   Model         | LoRA | Prefix Tuning  | P-Tuning | Prompt Tuning  | 
| --------- | ---- | ---- | ---- | ----  |
| BERT           | ✅  | ✅  |   |   |  
| RoBERTa        | ✅  | ✅  |   |   |
| GPT-2          | ✅  | ✅  |   |   | 
| Bloom          | ✅  | ✅  |   |   |   
| OPT            | ✅  | ✅  |   |   |
| GPT-Neo        | ✅  | ✅  |   |   |
| GPT-J          | ✅  | ✅  |   |   |
| Deberta        | ✅  |     |   |   | 
| Deberta-v2     | ✅  |     |   |   |


## Caveats:

1. Currently DeepSpeed requires PR [ZeRO3 handling frozen weights](https://github.com/microsoft/DeepSpeed/pull/2653) to fix [[REQUEST] efficiently deal with frozen weights during training](https://github.com/microsoft/DeepSpeed/issues/2615) issue on DeepSpeed repository. Example is provided in `~examples/pet_lora_seq2seq_accelerate_ds_zero3_offload.py`. 
  a. First run `accelerate config --config_file ds_zero3_cpu.yaml` and answer the questionaire. 
  Below are the contents of the config file.
  ```
  compute_environment: LOCAL_MACHINE
  deepspeed_config:
    gradient_accumulation_steps: 1
    gradient_clipping: 1.0
    offload_optimizer_device: cpu
    offload_param_device: cpu
    zero3_init_flag: true
    zero3_save_16bit_model: true
    zero_stage: 3
  distributed_type: DEEPSPEED
  downcast_bf16: 'no'
  dynamo_backend: 'NO'
  fsdp_config: {}
  machine_rank: 0
  main_training_function: main
  megatron_lm_config: {}
  mixed_precision: 'no'
  num_machines: 1
  num_processes: 1
  rdzv_backend: static
  same_network: true
  use_cpu: false
  ```
  b. run the below command to launch example script
  ```
  accelerate launch --config_file ds_zero3_cpu.yaml examples/pet_lora_seq2seq_accelerate_ds_zero3_offload.py
  ```

2. Below is an example of using PyTorch FSDP for training. However, it doesn't lead to 
any GPU memory savings. Please refer issue [[FSDP] FSDP with CPU offload consumes 1.65X more GPU memory when training models with most of the params frozen](https://github.com/pytorch/pytorch/issues/91165). 

  ```python
  from pet.utils.other import fsdp_auto_wrap_policy

  ...

  if os.environ.get("ACCELERATE_USE_FSDP", None) is not None:
      accelerator.state.fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(model)

  model = accelerator.prepare(model)
  ```

  Example of parameter efficient tuning with `mt0-xxl` base model using 🤗 Accelerate is provided in `~examples/pet_lora_seq2seq_accelerate_fsdp.py`. 
  a. First run `accelerate config --config_file fsdp_config.yaml` and answer the questionaire. 
  Below are the contents of the config file.
  ```
  command_file: null
  commands: null
  compute_environment: LOCAL_MACHINE
  deepspeed_config: {}
  distributed_type: FSDP
  downcast_bf16: 'no'
  dynamo_backend: 'NO'
  fsdp_config:
    fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
    fsdp_backward_prefetch_policy: BACKWARD_PRE
    fsdp_offload_params: true
    fsdp_sharding_strategy: 1
    fsdp_state_dict_type: FULL_STATE_DICT
    fsdp_transformer_layer_cls_to_wrap: T5Block
  gpu_ids: null
  machine_rank: 0
  main_process_ip: null
  main_process_port: null
  main_training_function: main
  megatron_lm_config: {}
  mixed_precision: 'no'
  num_machines: 1
  num_processes: 2
  rdzv_backend: static
  same_network: true
  tpu_name: null
  tpu_zone: null
  use_cpu: false
  ```
  b. run the below command to launch example script
  ```
  accelerate launch --config_file fsdp_config.yaml examples/pet_lora_seq2seq_accelerate_fsdp.py
  ```

3. When using `P_TUNING` or `PROMPT_TUNING` with `SEQ_2_SEQ` task, remember to remove the `num_virtual_token` virtual prompt predictions from the left side of the model outputs during evaluations. 

4. `P_TUNING` or `PROMPT_TUNING` doesn't support `generate` functionality of transformers bcause `generate` strictly requires `input_ids`/`decoder_input_ids` but 
`P_TUNING`/`PROMPT_TUNING` appends soft prompt embeddings to `input_embeds` to create
new `input_embeds` to be given to the model. Therefore, `generate` doesn't support this yet.

## Backlog:
1. Explore and possibly integrate `(IA)^3` and `UniPELT`
2. Add tests
3. Add more use cases and examples

## Citing 🤗 PET

If you use 🤗 PET in your publication, please cite it by using the following BibTeX entry.

```bibtex
@Misc{pet,
  title =        {PET: State-of-the-art Parameter-Efficient Tuning (PET) methods},
  author =       {Sourab Mangrulkar},
  howpublished = {\url{https://github.com/huggingface/pet}},
  year =         {2022}
}
```