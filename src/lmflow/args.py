#!/usr/bin/env python
# coding=utf-8
"""This script defines dataclasses: ModelArguments and DatasetArguments,
that contain the arguments for the model and dataset used in training.

It imports several modules, including dataclasses, field from typing, Optional from typing,
require_version from transformers.utils.versions, MODEL_FOR_CAUSAL_LM_MAPPING,
and TrainingArguments from transformers.

MODEL_CONFIG_CLASSES is assigned a list of the model config classes from
MODEL_FOR_CAUSAL_LM_MAPPING. MODEL_TYPES is assigned a tuple of the model types
extracted from the MODEL_CONFIG_CLASSES.
"""

from dataclasses import dataclass, field
from typing import Optional, List

from transformers.utils.versions import require_version

from transformers import (
    MODEL_FOR_CAUSAL_LM_MAPPING,
    TrainingArguments,
)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Define a class ModelArguments using the dataclass decorator. 
    The class contains several optional parameters that can be used to configure a model. 
    
    model_name_or_path : str
        a string representing the path or name of a pretrained
        model checkpoint for weights initialization. If None, a model will be trained from scratch.

    model_type :  str
        a string representing the type of model to use if training from
        scratch. If not provided, a pretrained model will be used.
    
    config_overrides :  str
        a string representing the default config settings to override
        when training a model from scratch.
    
    config_name : str
        a string representing the name or path of the pretrained config to
        use, if different from the model_name_or_path.
    
    tokenizer_name :  str
        a string representing the name or path of the pretrained tokenizer
        to use, if different from the model_name_or_path.

    cache_dir :  str
        a string representing the path to the directory where pretrained models
        downloaded from huggingface.co will be stored.

    use_fast_tokenizer : bool
        a boolean indicating whether to use a fast tokenizer (backed by the
        tokenizers library) or not.

    model_revision :  str
        a string representing the specific model version to use (can be a
        branch name, tag name, or commit id).

    use_auth_token : bool
        a boolean indicating whether to use the token generated when running
        huggingface-cli login (necessary to use this script with private models).

    torch_dtype :  str
        a string representing the dtype to load the model under. If auto is
        passed, the dtype will be automatically derived from the model's weights.

    use_ram_optimized_load : bool
        a boolean indicating whether to use disk mapping when memory is not
        enough.
    use_int8 : bool
        a boolean indicating whether to load int8 quantization for inference.
    load_in_4bit : bool
        whether to load the model in 4bit
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    model_n_layers: : int = field(
        default=4,
        metadata={"help": "The number of layers.",},
    )
    lora_model_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The incremental model diff introduced by LoRA finetuning."
                " Along with the original non-finetuned model forms the whole"
                " finetuned model."
            )
        }
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    arch_type: Optional[str] = field(
        default="decoder_only",
        metadata={"help": "The architecture type of the model. Currently supported decoder_only or encoder_decoder"}
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    arch_type: Optional[str] = field(
        default="decoder_only",
        metadata={
            "help": (
                "Model architecture type, e.g. \"decoder_only\","
                " \"encoder_decoder\""
            ),
            "choices": ["decoder_only", "encoder_decoder", "text_regression", "vision_encoder_decoder"],
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust remote code when loading model."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    use_lora: bool = field(
        default=False,
        metadata={"help": "Whether to lora."},
    )
    use_qlora: bool = field(
        default=False,
        metadata={"help": "Whether to use qlora."},
    )
    bits: int = field(
        default=4,
        metadata={"help": "The number of bits for quantization.",
                  "choices": [4, 8], },
    )
    quant_type: str = field(
        default='nf4',
        metadata={"help": "The quantization type for quantization.",
                  "choices": ["nf4", "fp4"], },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Whether to use double quantization."},
    )
    lora_r: int = field(
        default=8,
        metadata={"help": "the rank of the lora parameters. The smaller lora_r is , the fewer parameters lora has."},
    )
    lora_alpha: int = field(
        default=32,
        metadata={
            "help": "Merging ratio between the fine-tuned model and the original. This is controlled by a parameter called alpha in the paper."},
    )
    lora_target_modules: List[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name",
                                }
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "The dropout rate in lora.linear."},
    )
    save_aggregated_lora: bool = field(
        default=False,
        metadata={"help": "Whether to save aggregated lora."},
    )
    use_ram_optimized_load: bool = field(
        default=True,
        metadata={"help": "Whether use disk mapping when memory is not enough."}
    )
    use_flash_attention: bool = field(
        default=False,
        metadata={
            "help": (
                "whether use flash attention layer to reduce GPU memory with"
                " higher time cost."
            )
        }
    )
    truncate_to_model_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "whether truncate the dataset to model max length."
            )
        }
    )
    do_rope_scaling: bool = field(
        default=False,
        metadata={
            "help": (
                "whether do ROPE scaling for llama model."
                "Linear_scaling credits to the Reddit user /u/kaiokendev."
                "https://arxiv.org/abs/2306.15595"
                "NTK_scaling credits to the Reddit users /u/bloc97 and /u/emozilla."
                "https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/"
            )
        }
    )
    rope_pi_ratio: int = field(
        default=1,
        metadata={
            "help": (
                "the ratio of pi in RoPE scaling."
            )
        }
    )
    rope_ntk_ratio: int = field(
        default=1,
        metadata={
            "help": (
                "the ratio of NTK in RoPE scaling."
            )
        }
    )
    use_int8: bool = field(
        default=False,
        metadata={"help": "whether to load int8 quantization for inference"}
    )
    load_in_4bit: Optional[bool] = field(
        default=True,
        metadata={
            "help": "whether to load the model in 4bit"
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class VisModelArguments(ModelArguments):
    low_resource: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Use 8 bit and float16 when loading llm"
        }
    )
    custom_model: bool = field(
        default=False,
        metadata={"help": "flag for the model from huggingface or not"}
    )
    pretrained_language_projection_path: str = field(
        default=None,
        metadata={"help": "path for model pretrained_language_projection_path"}
    )
    custom_vision_model: bool = field(
        default=False,
        metadata={"help": "flag for the model from huggingface or not"}
    )
    image_encoder_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The name or path of the image encoder to use."
            )
        },
    )
    qformer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "llm model in multi-modality model"
            )
        },
    )
    llm_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "llm model in multi-modality model"
            )
        },
    )
    use_prompt_cache: bool = field(
        default=False,
        metadata={"help": "Whether to use prompt cache."},
    )
    prompt_cache_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to prompt cache."},
    )
    llava_loading: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to load module by module from pretrained model."},
    )
    with_qformer: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use qformer."},
    )
    vision_select_layer: Optional[int] = field(
        default=-2,
        metadata={"help": "Which layer to select in vision model."},
    )
    llava_pretrain_model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to llava pretrained model."},
    )
    save_pretrain_model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained model."},
    )


@dataclass
class DatasetArguments:
    """
    Define a class DatasetArguments using the dataclass decorator.
    The class contains several optional parameters that can be used to configure a dataset for a language model.


    dataset_path : str
        a string representing the path of the dataset to use.

    dataset_name : str
        a string representing the name of the dataset to use. The default value is "customized".

    is_custom_dataset : bool
        a boolean indicating whether to use custom data. The default value is False.

    customized_cache_dir : str
        a string representing the path to the directory where customized dataset caches will be stored.

    dataset_config_name : str
        a string representing the configuration name of the dataset to use (via the datasets library).

    train_file : str
        a string representing the path to the input training data file (a text file).

    validation_file : str
        a string representing the path to the input evaluation data file to evaluate the perplexity on (a text file).

    max_train_samples : int
        an integer indicating the maximum number of training examples to use for debugging or quicker training.
        If set, the training dataset will be truncated to this number.

    max_eval_samples: int
        an integer indicating the maximum number of evaluation examples to use for debugging or quicker training.
        If set, the evaluation dataset will be truncated to this number.

    streaming : bool
        a boolean indicating whether to enable streaming mode.

    block_size: int
        an integer indicating the optional input sequence length after tokenization. The training dataset will be
        truncated in blocks of this size for training.

    train_on_prompt: bool
        a boolean indicating whether to train on prompt for conversation datasets such as ShareGPT.

    disable_conversation_bos_token: bool
        [DEPRECATE SOON] a boolean indicating whether to disable the bos token for conversation datasets.
        
    disable_conversation_eos_token: bool
        [DEPRECATE SOON] a boolean indicating whether to disable the eos token for conversation datasets.

    conversation_template: str
        a string representing the template for conversation datasets.

    The class also includes some additional parameters that can be used to configure the dataset further, such as `overwrite_cache`,
    `validation_split_percentage`, `preprocessing_num_workers`, `disable_group_texts`, `demo_example_in_prompt`, `explanation_in_prompt`,
    `keep_linebreaks`, and `prompt_structure`.

    The field function is used to set default values and provide help messages for each parameter. The Optional type hint is
    used to indicate that a parameter is optional. The metadata argument is used to provide additional information about
    each parameter, such as a help message.
    """

    dataset_path: Optional[str] = field(
        default=None, metadata={"help": "The path of the dataset to use."}
    )
    dataset_name: Optional[str] = field(
        default="customized", metadata={"help": "Should be \"customized\""}
    )
    is_custom_dataset: Optional[bool] = field(
        default=False, metadata={"help": "whether to use custom data"}
    )
    customized_cache_dir: Optional[str] = field(
        default=".cache/llm-ft/datasets",
        metadata={"help": "Where do you want to store the customized dataset caches"},
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=1e10,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    group_texts_batch_size: int = field(
        default=1000,
        metadata={
            "help": (
                "Number of samples that will be grouped together to go though"
                " `group_texts` operation. See `--disable_group_texts` for"
                " detailed explanation of this operation."
            )
        }
    )
    disable_group_texts: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether we disable group of original samples together to"
                " generate sample sequences of length `block_size`"
                " By Default, it is True, which means the long samples"
                " are truncated to `block_size` tokens"
                " and short samples are padded to `block_size` tokens."
                " If set to False, we group every 1000 tokenized"
                " sequences together, divide them into"
                " [{total_num_tokens} / {block_size}] sequences,"
                " each with `block_size` tokens"
                " (the remaining tokens are ommited."
                " This group text behavior is useful"
                " for continual pretrain or pretrain."
            )
        },
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "Evaluation File Path"},
    )
    train_on_prompt: bool = field(
        default=False,
        metadata={"help": "Whether to train on prompt for conversation datasets such as ShareGPT."}
    )
    disable_conversation_bos_token: bool = field(
        default=False,
        metadata={"help": "Whether to disable the bos token for conversation datasets."}
    )
    disable_conversation_eos_token: bool = field(
        default=False,
        metadata={"help": "Whether to disable the eos token for conversation datasets."}
    )
    conversation_template: Optional[str] = field(
        default='empty',
        metadata={"help": "The template for conversation datasets."}
    )

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")

        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


@dataclass
class MultiModalDatasetArguments(DatasetArguments):
    image_folder: Optional[str] = field(
        default=None, metadata={"help": "The folder of the image file."}
    )
    image_aspect_ratio: Optional[str] = field(
        default="pad", metadata={"help": "The ratio type"}
    )
    is_multimodal: Optional[bool] = field(
        default=True, metadata={"help": "Flag for the modality type."}
    )
    use_image_start_end: Optional[bool] = field(
        default=True, metadata={"help": "Flag for the modality type."}
    )
    sep_style: Optional[str] = field(
        default="plain", metadata={"help": "Sep style in multi_modality dataset."}
    )


@dataclass
class FinetunerArguments(TrainingArguments):
    """
    Adapt transformers.TrainingArguments
    """
    owlore_metric_path: Optional[str] = field(
        default=None, metadata={"help": "The path of the eval dataset to use."}
    )
    eval_dataset_path: Optional[str] = field(
        default=None, metadata={"help": "The path of the eval dataset to use."}
    )
    remove_unused_columns: Optional[bool] = field(
        default=False,
        metadata={
            "help": "whether to remove the unused columns in collate fn"}
    )
    finetune_part: Optional[str] = field(
        default="language_projection",
        metadata={
            "help": "the module to finetune."
        }
    )
    save_language_projection: Optional[str] = field(
        default=False,
        metadata={
            "help": "whether to save language projection layer in multi-modal models."
        }
    )
    use_lisa: bool = field(
        default=False,
        metadata={
            "help": "whether to use LISA training strategy."
        }
    )
    lisa_activated_layers: int = field(
        default=2,
        metadata={
            "help": "the number of activated layers in LISA."
        }
    )
    lisa_interval_steps: int = field(
        default=20,
        metadata={
            "help": "the number of steps in each freezing interval of LISA, i.e. the selected unfreezed layers are randomly switched every {lisa_interval_steps} steps."
        }
    )
    lisa_layers_attribute: str = field(
        default="model.model.layers",
        metadata={
            "help": "where the layer attribute stores, e.g. model.model.layers"
        }
    )
    lisa_prob_mode: str = field(
        default="uniform",
        metadata={
            "help": "probabilities of different layers"
        }
    )
    galore: bool = field(
        default=False,
        metadata={"help": "Whether to enable Galore."},
    )


@dataclass
class EvaluatorArguments:
    """
    Define a class EvaluatorArguments using the dataclass decorator. The class contains several optional
    parameters that can be used to configure a evaluator.

    local_rank : str
        For distributed training: local_rank

    random_shuffle : bool

    use_wandb : bool

    random_seed : int, default = 1

    output_dir : str, default = './output_dir',

    mixed_precision : str, choice from ["bf16","fp16"].
        mixed precision mode, whether to use bf16 or fp16

    deepspeed :
        Enable deepspeed and pass the path to deepspeed json config file (e.g. ds_config.json) or an already
        loaded json file as a dict

    temperature : float
        An argument of model.generate in huggingface to control the diversity of generation.

    repetition_penalty : float
        An argument of model.generate in huggingface to penalize repetitions.
    """
    local_rank: int = field(
        default=-1,
        metadata={"help": "For distributed training: local_rank"
                  }
    )

    random_shuffle: Optional[bool] = field(
        default=False,
        metadata={"help": ""
                  }
    )

    use_wandb: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "When this flag is True, wandb will be enabled"
            )
        },
    )
    random_seed: Optional[int] = field(
        default=1,
        metadata={
            "help": (
                "used to set random seed"
            )
        },
    )
    output_dir: Optional[str] = field(
        default="./output_dir",
        metadata={"help": "Output path for the inferenced results"},
    )
    mixed_precision: Optional[str] = field(
        default="bf16",
        metadata={
            "help": (
                "mixed precision mode, whether to use bf16 or fp16"
            ),
            "choices": ["bf16", "fp16"],
        },
    )
    deepspeed: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Enable deepspeed and pass the path to deepspeed json config file (e.g. ds_config.json) or an already"
                " loaded json file as a dict"
            )
        },
    )
    answer_type: Optional[str] = field(
        default="text",
        metadata={
            "help": (
                'Question type for answer extraction from the decoder output.'
                ' Supported types: \n'
                '   1) "multiple_choice", e.g. A, B, C, D, ...\n'
                '   2) "binary_choice", e.g. yes, no, maybe\n'
                '   3) "math", e.g. 1.0, -3.52\n'
                '   4) "text", e.g. "I think that it is okay"\n'
                '   5) Special treatment for several datasets\n'
                '     - "gsm8k"\n'
                '     - "svamp"\n'
                '     - "asdiv"\n'
                '     - "addsub"\n'
                '     - "singleeq"\n'
                '     - "multiarith"\n'
                '     - "aqua"\n'
                '     - "csqa"\n'
                '     - "strategyqa"\n'
                '     - "pubmedqa"\n'
                '     - "medmcqa"\n'
                '     - "usmle"\n'
            )
        },
    )
    prompt_structure: Optional[str] = field(
        default="{input}",
        metadata={
            "help": (
                'Prompt structure to facilitate prompt engineering during'
                ' inference. The model will receive'
                ' `prompt_structure.format(input=input)` as its input.'
            )
        },
    )
    evaluate_block_size: Optional[int] = field(
        default=512,
        metadata={
            "help": (
                "the model will have at least block_size tokens for context when calculating the conditional likelihood of any one token"
                " (provided there are block_size preceding tokens available to condition on)"
            )
        },
    )
    metric: Optional[str] = field(
        default="accuracy",
        metadata={
            "help": "the metric the model will be evaluated on",
            "choices": ["ppl", "perplexity", "acc", "accuracy", "nll", "neg_log_likelihood"],
        },
    )
    inference_batch_size_per_device: Optional[int] = field(
        default=1,
        metadata={
            "help": (
                "every device will infer {inference_batch_size_per_device}"
                " samples in parallel. The inferred results will be concatenaed"
                " with inputs and attach a reward."
            ),
        },
    )
    use_accelerator_for_evaluator: bool = field(
        default=False, metadata={"help": "Whether to use Huggingface Accelerator instead of Deepspeed"},
    )

    temperature: float = field(
        default=0,
        metadata={"help": "Temperature during inference."},
    )

    repetition_penalty: float = field(
        default=1,
        metadata={"help": "Repetition_penalty during inference."},
    )

    max_new_tokens: int = field(
        default=100,
        metadata={"help": "Maximum length during inference."},
    )


@dataclass
class InferencerArguments:
    """
    Define a class InferencerArguments using the dataclass decorator. The class contains several optional
    parameters that can be used to configure a inferencer.

    local_rank : str
        For distributed training: local_rank

    random_seed : int, default = 1

    deepspeed :
        Enable deepspeed and pass the path to deepspeed json config file (e.g. ds_config.json) or an already
        loaded json file as a dict
    mixed_precision : str, choice from ["bf16","fp16"].
        mixed precision mode, whether to use bf16 or fp16

    temperature : float
        An argument of model.generate in huggingface to control the diversity of generation.

    repetition_penalty : float
        An argument of model.generate in huggingface to penalize repetitions.
    """
    device: str = field(
        default="gpu",
        metadata={
            "help": "device of chatbot",
            "choices": ["gpu", "cpu"],
        },
    )
    local_rank: int = field(
        default=-1,
        metadata={"help": "For distributed training: local_rank"
                  },
    )

    temperature: float = field(
        default=0.0,
        metadata={"help": "Temperature during inference."},
    )

    repetition_penalty: float = field(
        default=1,
        metadata={"help": "Repetition_penalty during inference."},
    )

    max_new_tokens: int = field(
        default=100,
        metadata={"help": "Maximum length during inference."},
    )

    random_seed: Optional[int] = field(
        default=1,
        metadata={
            "help": (
                "used to set random seed"
            )
        },
    )
    deepspeed: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Enable deepspeed and pass the path to deepspeed json config file (e.g. ds_config.json) or an already"
                " loaded json file as a dict"
            )
        },
    )
    mixed_precision: Optional[str] = field(
        default="bf16",
        metadata={
            "help": (
                "mixed precision mode, whether to use bf16 or fp16"
            ),
            "choices": ["bf16", "fp16"],
        },
    )
    do_sample: Optional[bool] = field(
        default=False,
        metadata={
            "help": "whether turn on true random sampling during inference."
        },
    )
    use_accelerator: bool = field(
        default=False, metadata={"help": "Whether to use Huggingface Accelerator instead of Deepspeed"},
    )


@dataclass
class RaftAlignerArguments(TrainingArguments):
    """
    Define a class RaftAlignerArguments to configure raft aligner.
    """
    output_reward_path: Optional[str] = field(
        default="tmp/raft_aligner/",
        metadata={
            "help": "The path of output rewards."
        }
    )
    output_min_length: Optional[int] = field(
        default=64,
        metadata={
            "help": (
                "minimum length of the output token sequence generated from"
                " model given an input."
            ),
        },
    )
    output_max_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "maximum length of the output token sequence generated from"
                " model given an output."
            ),
        },
    )
    num_raft_iteration: Optional[int] = field(
        default=20,
        metadata={
            "help": "number of iterations of the raft aligner."
        },
    )
    raft_batch_size: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "only select {raft_batch_size} samples each time for STF training."
            )
        },
    )
    top_reward_percentage: Optional[float] = field(
        default=0.2,
        metadata={
            "help": (
                "only top {top_reward_percentage} samples in the raft batch,"
                " (in terms of rewards), will be used for SFT the model."
            ),
        },
    )
    inference_batch_size_per_device: Optional[int] = field(
        default=1,
        metadata={
            "help": (
                "every device will infer {inference_batch_size_per_device}"
                " samples in parallel. The inferred results will be concatenaed"
                " with inputs and attach a reward."
            ),
        },
    )
    collection_strategy: Optional[str] = field(
        default="top",
        metadata={
            "help": (
                "{collection_strategy} is either top or local"
                " top means that we rank the samples globally regardless of the prompts"
                " local means that we only rank the samples with the same prompt"
            ),
        },
    )


@dataclass
class BenchmarkingArguments:
    dataset_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "benchmark dataset name provided by lmflow"
        },
    )
    lm_evaluation_metric: Optional[str] = field(
        default="accuracy",
        metadata={
            "help": "the metric the model will be evaluated on",
            "choices": ["acc", "acc_norm", "bleu", "chrf", "em", "f1", "ppl", \
                        "ter", "r@1", "r@2", "mrr", "mc1", "mc2", "word_perplexity", \
                        "byte_perplexity", "bits_per_byte"],
        },
    )


@dataclass
class DPOAlignerArguments:
    """
    The arguments for the DPO training script.
    """
    local_rank: int = field(
        default=-1,
        metadata={"help": "For distributed training: local_rank"
                  },
    )
    # data parameters
    beta: Optional[float] = field(
        default=0.1,
        metadata={
            "help": "the beta parameter for DPO loss"
        }
    )
    # # training parameters
    learning_rate: Optional[float] = field(
        default=5e-4,
        metadata={
            "help": "optimizer learning rate"
        }
    )
    lr_scheduler_type: Optional[str] = field(
        default="cosine",
        metadata={
            "help": "the lr scheduler type"
        }
    )
    warmup_steps: Optional[int] = field(
        default=100, metadata={
            "help": "the number of warmup steps"
        }
    )
    weight_decay: Optional[float] = field(
        default=0.05, metadata={
            "help": "the weight decay"
        }
    )
    optimizer_type: Optional[str] = field(
        default="paged_adamw_32bit",
        metadata={
            "help": "the optimizer type"
        }
    )

    per_device_train_batch_size: Optional[int] = field(
        default=4,
        metadata={
            "help": "train batch size per device"
        }
    )
    per_device_eval_batch_size: Optional[int] = field(
        default=1, metadata={
            "help": "eval batch size per device"
        }
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=4,
        metadata={
            "help": "the number of gradient accumulation steps"
        },
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True,
        metadata={
            "help": "whether to use gradient checkpointing"
        },
    )

    gradient_checkpointing_use_reentrant: Optional[bool] = field(
        default=False,
        metadata={
            "help": "whether to use reentrant for gradient checkpointing"
        },
    )
    max_prompt_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "the maximum prompt length"
        },
    )
    max_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "the maximum sequence length"
        },
    )
    max_steps: Optional[int] = field(
        default=1000,
        metadata={
            "help": "max number of training steps"
        },
    )
    logging_steps: Optional[int] = field(
        default=10,
        metadata={
            "help": "the logging frequency"
        },
    )
    save_steps: Optional[int] = field(
        default=100,
        metadata={
            "help": "the saving frequency"
        },
    )
    eval_steps: Optional[int] = field(
        default=100,
        metadata={
            "help": "the evaluation frequency"
        },
    )
    output_dir: Optional[str] = field(
        default="./results",
        metadata={
            "help": "the output directory"
        },
    )
    log_freq: Optional[int] = field(
        default=1,
        metadata={
            "help": "the logging frequency"
        },
    )
    sanity_check: Optional[bool] = field(
        default=False,
        metadata={
            "help": "only train on 1000 samples"
        }
    )
    report_to: Optional[str] = field(
        default="wandb",
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
                    '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
                    'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )
    seed: Optional[int] = field(
        default=0, metadata={"help": "Random seed that will be set at the beginning of training."}
    )
    run_name: Optional[str] = field(
        default="dpo", metadata={"help": "The name of the run."}
    )


PIPELINE_ARGUMENT_MAPPING = {
    "finetuner": FinetunerArguments,
    "evaluator": EvaluatorArguments,
    "inferencer": InferencerArguments,
    "raft_aligner": RaftAlignerArguments,
    "dpo_aligner": DPOAlignerArguments,
}


class AutoArguments:
    """
    Automatically choose arguments from FinetunerArguments or EvaluatorArguments.
    """

    def get_pipeline_args_class(pipeline_name: str):
        return PIPELINE_ARGUMENT_MAPPING[pipeline_name]
