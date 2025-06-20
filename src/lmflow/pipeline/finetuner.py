#!/usr/bin/env python
# coding=utf-8
"""The Finetuner class simplifies the process of running finetuning process on a language model for a TunableModel instance with given dataset.
"""
import ipdb
import copy
import logging
import os
import sys
import json
import torch
import datasets
import numpy as np
import transformers
import evaluate
from itertools import chain
from transformers import (
    Trainer,
    default_data_collator,
    set_seed,
)
from copy import deepcopy
from transformers.utils import send_example_telemetry
from transformers.trainer_utils import get_last_checkpoint
from transformers.trainer_callback import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from lmflow.datasets.dataset import Dataset
from lmflow.pipeline.base_tuner import BaseTuner
from lmflow.pipeline.utils.peft_trainer import PeftTrainer, PeftSavingCallback


logger = logging.getLogger(__name__)


class Finetuner(BaseTuner):
    """
    Initializes the `Finetuner` class with given arguments.

    Parameters
    ------------
    model_args : ModelArguments object.
        Contains the arguments required to load the model.

    data_args : DatasetArguments object.
        Contains the arguments required to load the dataset.

    finetuner_args : FinetunerArguments object.
        Contains the arguments required to perform finetuning.

    args : Optional.
        Positional arguments.

    kwargs : Optional.
        Keyword arguments.

    """
    def __init__(self, model_args, data_args, finetuner_args, *args, **kwargs):

        self.model_args = model_args
        self.data_args = data_args
        self.finetuner_args = finetuner_args

        # Sending telemetry. Tracking the example usage helps us better
        # allocate resources to maintain them. The information sent is the one
        # passed as arguments along with your Python/PyTorch versions.
        send_example_telemetry("run_clm", model_args, data_args)

        # Setup logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            handlers=[logging.StreamHandler(sys.stdout)],
        )

        log_level = finetuner_args.get_process_log_level()
        logger.setLevel(log_level)
        datasets.utils.logging.set_verbosity(log_level)
        transformers.utils.logging.set_verbosity(log_level)
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()

        # Log on each process the small summary:
        logger.warning(
            f"Process rank: {finetuner_args.local_rank},"
            f" device: {finetuner_args.device},"
            f" n_gpu: {finetuner_args.n_gpu},"
            f"distributed training: {bool(finetuner_args.local_rank != -1)},"
            f" 16-bits training: {finetuner_args.fp16}"
        )
        logger.info(f"Training/evaluation parameters {finetuner_args}")

        # Detecting last checkpoint.
        last_checkpoint = None
        if os.path.isdir(finetuner_args.output_dir) and finetuner_args.do_train and not finetuner_args.overwrite_output_dir:
            last_checkpoint = get_last_checkpoint(finetuner_args.output_dir)
            if last_checkpoint is None and len(os.listdir(finetuner_args.output_dir)) > 0:
                raise ValueError(
                    f"Output directory ({finetuner_args.output_dir}) already"
                    " exists and is not empty. "
                    "Use --overwrite_output_dir to overcome."
                )
            elif last_checkpoint is not None and finetuner_args.resume_from_checkpoint is None:
                logger.info(
                    f"Checkpoint detected, resuming training at"
                    f" {last_checkpoint}. To avoid this behavior, change"
                    " the `--output_dir` or add `--overwrite_output_dir` to"
                    " train from scratch."
                )
        self.last_checkpoint = last_checkpoint

        # Set seed before initializing model.
        set_seed(finetuner_args.seed)


    def group_text(self, tokenized_datasets, model_max_length):
        """
        Groups texts together to form blocks of maximum length `model_max_length` and returns the processed data as
        a dictionary.
        """
        data_args = self.data_args
        finetuner_args = self.finetuner_args

        if data_args.block_size is None:
            block_size = model_max_length
            if block_size > 1024:
                logger.warning(
	    			"The chosen tokenizer supports a `model_max_length` that is"
	    			" longer than the default `block_size` value"
	    			" of 1024. If you would like to use a longer `block_size`"
	    			" up to `tokenizer.model_max_length` you can override this "
	    			" default with `--block_size xxx`."
                )
                block_size = 1024
        else:
            if data_args.block_size > model_max_length:
                if self.model_args.truncate_to_model_max_length:
                    logger.warning(
                        f"The block_size passed ({data_args.block_size}) is larger"
                        f" than the maximum length for the model"
                        f"({model_max_length})."
                        f" Using block_size={model_max_length}."
                        f"If you would like to use a longer 'block_size' that is"
                        f" longer than the maximum length supported by the model,"
                        f" you can override this behavior with"
                        f"default with `--truncate_to_model_max_length False`."
                    )
                    block_size = model_max_length
                else:
                    logger.warning(
                        f"The block_size passed ({data_args.block_size}) is larger"
                        f"than the maximum length for the model"
                        f"({model_max_length})."
                        f"Using block_size={data_args.block_size}.")
                    block_size = data_args.block_size
            else:
                block_size = data_args.block_size
        # Main data processing function that will concatenate all texts from
        # our dataset and generate chunks of block_size.
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model
            # supported it instead of this drop, you can customize this part to
            # your needs.
            total_length = (total_length // block_size) * block_size
            # Split by chunks of max_len.
            result = {
                k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            return result

        # Note that with `batched=True`, this map processes 1,000 texts
        # together, so group_texts throws away a remainder for each of those
        # groups of 1,000 texts. You can adjust that batch_size here but a
        # higher value might be slower to preprocess.
        #
        # To speed up this part, we use multiprocessing. See the documentation
        # of the map method for more information:
        # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map
        with finetuner_args.main_process_first(desc="grouping texts together"):
            group_batch_size = data_args.group_texts_batch_size
            if data_args.disable_group_texts:
                group_batch_size = 1
            if not data_args.streaming:
                lm_datasets = tokenized_datasets.map(
                    group_texts,
                    batched=True,
                    batch_size=group_batch_size,
                    num_proc=data_args.preprocessing_num_workers,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc=f"Grouping texts in chunks of {block_size}",
                )
            else:
                lm_datasets = tokenized_datasets.map(
                    group_texts,
                    batched=True,
                    batch_size=group_batch_size,
                )

        return lm_datasets


    def tune(self,
             model,
             dataset,
             transform_dataset_in_place=True,
             data_collator=None):
        """
        Perform tuning for a model

        Parameters
        ------------
        model : TunableModel object.
            TunableModel to perform tuning.

        dataset:
            dataset to train model.

        """
        model_args = self.model_args
        data_args = self.data_args
        finetuner_args = self.finetuner_args
        if not transform_dataset_in_place:
            dataset = copy.deepcopy(dataset)

        # Tokenization and text grouping must be done in the main process
        if dataset.backend == "custom_multi_modal":
            dataset.backend_dataset.register_tokenizer(
                model.tokenizer, model.image_processor)
            lm_dataset = dataset
        else:
            with finetuner_args.main_process_first(desc="dataset map tokenization"):
                tokenized_dataset = model.tokenize(dataset)
                if data_args.disable_group_texts:
                    lm_dataset = tokenized_dataset
                else:
                    lm_dataset = self.group_text(
                        tokenized_dataset,
                        model_max_length=model.get_max_length(),
                    )

        train_dataset = lm_dataset.get_backend_dataset()
        logger.info(f"Number of train samples: {len(train_dataset)}")

        if finetuner_args.do_eval:
            eval_dataset_args = deepcopy(data_args)
            eval_dataset_args.dataset_path = finetuner_args.eval_dataset_path
            eval_dataset = Dataset(eval_dataset_args)
            with finetuner_args.main_process_first(desc="dataset map tokenization"):
                tokenized_dataset = model.tokenize(eval_dataset)
                if data_args.disable_group_texts:
                    lm_dataset = tokenized_dataset
                else:
                    lm_dataset = self.group_text(
                        tokenized_dataset,
                        model_max_length=model.get_max_length(),
                    )
            eval_dataset = lm_dataset.get_backend_dataset()
            logger.info(f"Number of eval samples: {len(train_dataset)}")


            def preprocess_logits_for_metrics(logits, labels):
                if isinstance(logits, tuple):
                    # Depending on the model and config, logits may contain extra tensors,
                    # like past_key_values, but logits always come first
                    logits = logits[0]
                return logits.argmax(dim=-1)

            metric = evaluate.load("accuracy")

            def compute_metrics(eval_preds):
                preds, labels = eval_preds
                # preds have the same shape as the labels, after the argmax(-1) has been calculated
                # by preprocess_logits_for_metrics but we need to shift the labels
                labels = labels[:, 1:].reshape(-1)
                preds = preds[:, :-1].reshape(-1)
                return metric.compute(predictions=preds, references=labels)

        if finetuner_args.do_train:
            if data_args.max_train_samples is not None:
                max_train_samples = min(len(train_dataset), data_args.max_train_samples)
                train_dataset = train_dataset.select(range(max_train_samples))

        # Initialize our Trainer
        training_args = finetuner_args

        if model_args.use_lora:
            FinetuningTrainer = PeftTrainer
            trainer_callbacks = [PeftSavingCallback]
        else:
            FinetuningTrainer = Trainer
            trainer_callbacks = []
        if data_collator is None:
            data_collator = default_data_collator

        if training_args.use_lisa:
            class DynamicLayerActivationCallback(TrainerCallback):
                def __init__(self, n_layers, interval_steps, model, prob_mode):
                    super().__init__()
                    self.n_layers = n_layers
                    self.interval_steps = interval_steps
                    self.prob_mode = prob_mode
                    self.model = model

                    # Determine the way to access layers based on the model type
                    class_to_layers_map = {
                        'LlamaForCausalLM': 'model.model.layers',
                        'Qwen2ForCausalLM': 'model.model.layers',
                        'MistralForCausalLM': 'model.model.layers',
                        'MixtralForCausalLM': 'model.model.layers',
                        'GemmaForCausalLM': 'model.model.layers',
                        'GPT2LMHeadModel': 'model.transformer.h',
                    }
                    model_class_name = self.model.__class__.__name__
                    if model_class_name in class_to_layers_map:
                        self.layers_attribute = class_to_layers_map[model_class_name]
                    else:
                        self.layers_attribute = training_args.lisa_layers_attribute
                    # import ipdb
                    # ipdb.set_trace()
		
                    self.total_layers = model_args.model_n_layers #self.total_layers = len(eval('self.' + self.layers_attribute))  # Dynamically execute to get the number of layers
			
                    self.active_layers_indices = []

                def freeze_all_layers(self):
                    layers = eval('self.' + self.layers_attribute)  # Dynamically execute to get layers
                    for layer in layers:
                        for param in layer.parameters():
                            param.requires_grad = False

                def on_step_begin(self, args, state, control, **kwargs):
                    # Check if it's time to switch active layers, including at step 0
                    if state.global_step % self.interval_steps == 0:
                        self.switch_active_layers()

                def switch_active_layers(self):
                    # First, disable gradients for all layers
                    self.freeze_all_layers()

                    # Randomly select n_layers to activate
                    layers = eval('self.' + self.layers_attribute)  # Re-fetch layer references
                    if self.prob_mode == 'decrease':
                        probabilities = np.arange(self.total_layers, 0, -1)
                        layer_probabilities = probabilities / np.sum(probabilities)
                    if self.prob_mode == 'increase':
                        probabilities = np.arange(1, self.total_layers + 1)
                        layer_probabilities = probabilities / np.sum(probabilities)
                    if self.prob_mode == 'uniform':
                        layer_probabilities = None
                    if self.prob_mode == 'owl' or self.prob_mode == 'owl_reverse' or self.prob_mode == 'norm':
                        # load metric
                        importance_scores = []
                        if self.prob_mode == 'norm':
                            with open('metric_cache/weight_norm.txt', 'r') as f:
                                for line in f.readlines():
                                    importance_scores.append(float(line.strip()))
                        else:
                            with open(finetuner_args.owlore_metric_path, 'r') as f:
                                for line in f.readlines():
                                    importance_scores.append(float(line.strip()))

                        # extract the importance metrics for each layer and compute their average values
                        layer_scores = {}
                        layer_id = 0
                        for value in importance_scores:
                            if layer_id not in layer_scores:
                                layer_scores[layer_id] = []
                            layer_scores[layer_id].append(value)
                            layer_id += 1

                        layer_avg_scores = {layer_id: np.mean(scores) for layer_id, scores in layer_scores.items()}

                        # norm the scores
                        total_score = sum(layer_avg_scores.values())
                        layer_probs = {layer_id: score / total_score for layer_id, score in layer_avg_scores.items()}

                        # create a list of probabilities for each layer
                        num_layers = max(layer_probs.keys()) + 1
                        layer_probabilities = np.zeros(num_layers)
                        for layer_id, prob in layer_probs.items():
                            layer_probabilities[layer_id] = prob
                        
                        
                        if self.prob_mode == 'owl_reverse':
                            max_prob = np.max(layer_probabilities)
                            reversed_probs = max_prob - layer_probabilities
                            reversed_probs /= np.sum(reversed_probs)
                            layer_probabilities = reversed_probs

                        print("Layer selection probabilities:")
                        print(layer_probabilities)
                    self.active_layers_indices = np.random.choice(range(self.total_layers), self.n_layers, replace=False, p=layer_probabilities)
                    print(f"Activating layers at indices: {self.active_layers_indices} for the next steps.", flush=True)

                    # Enable gradients only for the selected layers
                    for idx in self.active_layers_indices:
                        for _name, param in layers[idx].named_parameters():
                            if model_args.use_lora:
                                if 'lora' in _name:
                                    param.requires_grad = True
                            else:
                                param.requires_grad = True
            # Instantiate the callback
            dynamic_layer_activation_callback = DynamicLayerActivationCallback(
                n_layers=training_args.lisa_activated_layers,                     # Number of layers to activate
                interval_steps=training_args.lisa_interval_steps,               # Step interval to update active layers
                model=model.get_backend_model(),
                prob_mode=training_args.lisa_prob_mode,  # Probability mode for layer selection
            )

            trainer_callbacks.append(dynamic_layer_activation_callback)

        #############################
        # optimizers implementation #
        #############################
        optimizer = None
        if training_args.galore:
            from galore_torch import GaLoreAdamW, GaLoreAdamW8bit, GaLoreAdafactor
            
            galore_params = []
            galore_params_ranked = {}
            target_modules_list = ["attn", "mlp"]
            for module_name, module in model.get_backend_model().named_modules():
                if not isinstance(module, torch.nn.Linear):
                    continue

                if not any(target_key in module_name for target_key in target_modules_list):
                    continue

                print('enable GaLore for weights in module: ', module_name)
                layer_num = ''.join(filter(str.isdigit, module_name)) # get layer number
                rank = 256 # TODO: fixed rank, we also can use different rank for different layers
                if rank is not None:
                    galore_params_ranked[module_name] = {"rank": rank, "param": module.weight}
                    print(f'Layer number: {layer_num}; Ranking number: {rank}')
                galore_params.append(module.weight)

            id_galore_params = [id(p) for p in galore_params]
            # make parameters without "rank" to another group
            regular_params = [p for p in model.get_backend_model().parameters() if id(p) not in id_galore_params]
            # then call galore_adamw
            param_groups = [{'params': regular_params}]
            for module_name, weight_data in galore_params_ranked.items():
                param_group = {
                    'params': [weight_data["param"]],
                    'rank': weight_data["rank"],
                    'update_proj_gap': 100, 
                    'scale': 1.0,
                    'proj_type': 'std',
                    'module_names': [module_name]
                }
                param_groups.append(param_group)
            # galore_optimizer = GaLoreAdamW8bit(param_groups, lr=training_args.learning_rate)
            galore_optimizer = GaLoreAdamW(param_groups, lr=training_args.learning_rate)
            optimizer = galore_optimizer

        trainer = FinetuningTrainer(
            model=model.get_backend_model(),
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            tokenizer=model.get_tokenizer(),
            # Data collator will default to DataCollatorWithPadding, so we change it.
            data_collator=data_collator,
            compute_metrics=compute_metrics if training_args.do_eval else None,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics if training_args.do_eval else None,
            callbacks=trainer_callbacks,
            optimizers=(optimizer, None)
        )
        
        # Training
        if training_args.do_train:
            checkpoint = None
            last_checkpoint = self.last_checkpoint
            if training_args.resume_from_checkpoint is not None:
                checkpoint = training_args.resume_from_checkpoint
            elif last_checkpoint is not None:
                checkpoint = last_checkpoint
            train_result = trainer.train(resume_from_checkpoint=checkpoint)

            if not model_args.use_lora:
                trainer.save_model()  # Saves the tokenizer too for easy upload
            else:
                if model_args.save_aggregated_lora:
                    model.merge_lora_weights()
                model.save(finetuner_args.output_dir,model_args.save_aggregated_lora)
            # save language_projection for multi-modal model;
            if self.finetuner_args.save_language_projection:
                language_projection_state = trainer.model.language_projection.state_dict()
                torch.save(
                    osp.join(
                        self.finetuner_args.output_dir,
                        "language_projection.pth"),
                    language_projection_state)
            metrics = train_result.metrics

            max_train_samples = (
                data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
            )
            metrics["train_samples"] = min(max_train_samples, len(train_dataset))

            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()

        kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-generation"}
        if data_args.dataset_name is not None:
            kwargs["dataset_tags"] = data_args.dataset_name
            if data_args.dataset_config_name is not None:
                kwargs["dataset_args"] = data_args.dataset_config_name
                kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
            else:
                kwargs["dataset"] = data_args.dataset_name

        if training_args.push_to_hub:
            trainer.push_to_hub(**kwargs)
        else:
            trainer.create_model_card(**kwargs)

        return model
