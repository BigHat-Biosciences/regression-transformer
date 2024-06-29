from dataclasses import dataclass, field
from typing import Optional
import argparse
import os

from transformers import MODEL_WITH_LM_HEAD_MAPPING
from transformers.training_args import TrainingArguments

MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


@dataclass
class CustomTrainingArguments(TrainingArguments):
    """
    NOTE: Expanding TrainingArguments class from transformers with custom arguments.

    eval_accumulation_steps (:obj:`int`, `optional`):
            Number of predictions steps to accumulate the output tensors for, before moving the results to the CPU. If
            left unset, the whole predictions are accumulated on GPU/TPU before being moved to the CPU (faster but
            requires more memory).
    """

    # Overwrite all bool type arguments to have type str2bool
    use_wandb: str2bool = field(default=False, metadata={"help": "Use wandb for logging"})
    do_train: str2bool = field(default=False, metadata={"help": "Whether to run training."})
    do_eval: str2bool = field(default=False, metadata={"help": "Whether to run eval on the dev set."})
    do_predict: str2bool = field(default=False, metadata={"help": "Whether to run predictions on the test set."})
    use_cpu: str2bool = field(default=False, metadata={"help": "Use CPU for training and evaluation"})
    fp16: str2bool = field(default=False, metadata={"help": "Whether to use 16-bit (mixed) precision training."})
    evaluate_during_training: str2bool = field(default=False, metadata={"help": "Run evaluation during training at each logging step."})
    overwrite_output_dir: str2bool = field(default=False, metadata={"help": "Overwrite the content of the output directory"})

    output_dir: str = field(
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
        default=os.environ["SM_MODEL_DIR"],
    )

    training_name: str = field(
        default="rt_training", metadata={"help": "Name used to identify the training."}
    )
    num_train_epochs: int = field(default=10, metadata={"help": "Number of epochs."})
    batch_size: int = field(default=16, metadata={"help": "Size of the batch."})
    log_interval: int = field(
        default=100, metadata={"help": "Number of steps between log intervals."}
    )
    gradient_interval: int = field(
        default=1, metadata={"help": "Gradient accumulation steps"}
    )
    eval_steps: int = field(
        default=1000,
        metadata={"help": "The time interval at which validation is performed."},
    )

    max_span_length: int = field(
        default=5, metadata={"help": "Max length of a span of masked tokens for PLM."}
    )
    plm_probability: float = field(
        default=1 / 6,
        metadata={
            "help": "Ratio of length of a span of masked tokens to surrounding context length for PLM."
        },
    )
    alternate_steps: int = field(
        default=50,
        metadata={
            "help": "Per default, training alternates between property prediction and "
            "conditional generation. This argument specifies the alternation frequency."
            "If you set it to 0, no alternation occurs and we fall back to vanilla "
            "permutation language modeling (PLM). Default: 50."
        },
    )
    cc_loss: str2bool = field(
        default=False,
        metadata={
            "help": "Whether the cycle-consistency loss is computed during the conditional "
            "generation task. Defaults to False."
        },
    )
    cc_loss_weight: float = field(
        default=1.0,
        metadata={
            "help": "Weight of the cycle-consistency loss. Only applies if `cc_loss` is True. "
            "Defaults to 1.0."
        },
    )
    cg_collator: str = field(
        default="vanilla_cg",
        metadata={
            "help": "The collator class. Following options are implemented: "
            "'vanilla_cg': Collator class that does not mask the properties but anything else as a regular DataCollatorForPermutationLanguageModeling. Can optionally replace the properties with sampled values. "
            "NOTE: This collator can deal with multiple properties. "
            "'multientity_cg': A training collator the conditional-generation task that can handle multiple entities. "
            "Default: vanilla_cg."
        },
    )
    entity_to_mask: int = field(
        default=-1,
        metadata={
            "help": "Only applies if `cg_collator='multientity_cg'`. The entity that is being masked during training. 0 corresponds to first entity and so on. -1 corresponds to "
            "a random sampling scheme where the entity-to-be-masked is determined "
            "at runtime in the collator. NOTE: If 'mask_entity_separator' is true, "
            "this argument will not have any effect. Defaults to -1."
        },
    )
    entity_separator_token: str = field(
        default=".",
        metadata={
            "help": "Only applies if `cg_collator='multientity_cg'`.The token that is used to separate "
            "entities in the input. Defaults to '.' (applicable to SMILES & SELFIES)"
        },
    )
    mask_entity_separator: str2bool = field(
        default=False,
        metadata={
            "help": "Only applies if `cg_collator='multientity_cg'`. Whether or not the entity separator token can be masked. If True, *all** textual tokens can be masked and we "
            "the collator behaves like the `vanilla_cg ` even though it is a `multientity_cg`. If False, the exact behavior "
            "depends on the entity_to_mask argument. Defaults to False."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    data_dir: Optional[str] = field(
        default=os.environ["SM_CHANNEL_TRAINING"],
        metadata={"help": "The input data dir. Should contain the .txt files for the task."},
    )
    train_data_path: str = field(
        default="train.csv",
        metadata={
            "help": "Path to a `.csv` file with the input training data. The file has to "
            "contain a `text` column (with the string input, e.g, SMILES, AAS, natural "
            "text) and an arbitrary number of numerical columns."
        },
    )
    train_metadata_path: str = field(
        default=None,
        metadata={"help": "Path to a `.csv` file with metadata columns for the training data."},
    )
    test_data_path: str = field(
        default="test.csv",
        metadata={
            "help": "Path to a `.csv` file with the input testing data. The file has to "
            "contain a `text` column (with the string input, e.g, SMILES, AAS, natural "
            "text) and an arbitrary number of numerical columns."
        },
    )
    test_metadata_path: str = field(
        default=None,
        metadata={"help": "Path to a `.csv` file with metadata columns for the testing data."},
    )
    block_size: int = field(
        default=-1,
        metadata={"help": "Optional input sequence length after tokenization."},
    )
    augment: Optional[int] = field(
        default=0,
        metadata={
            "help": "Factor by which the training data is augmented. The data modality "
            "(SMILES, SELFIES, AAS, natural text) is inferred from the tokenizer. "
            "NOTE: For natural text, no augmentation is supported. Defaults to 0, "
            "meaning no augmentation. (gt4sd)"
        },
    )
    line_by_line: Optional[str2bool] = field(
        default=False,
        metadata={
            "help": "Whether lines of text in the dataset are to be handled as distinct samples."
        },
    )
    overwrite_cache: str2bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    save_datasets: str2bool = field(
        default=False,
        metadata={
            "help": "Whether to save the datasets to disk. Datasets will be saved as `.txt` file to "
            "the same location where `train_data_path` and `test_data_path` live. Defaults to False. (gt4sd)"
        },
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    model_path: str = field(
        default=None,
        metadata={"help": "Path where the model artifacts are stored. (gt4sd)"},
    )
    checkpoint_name: str = field(
        default=str(),
        metadata={
            "help": "Name for the checkpoint that should be copied to inference model. "
            "Has to be a subfolder of `model_path`. Defaults to empty string meaning that "
            "files are taken from `model_path` (i.e., after training finished). (gt4sd)"
        },
    )
    model_type: Optional[str] = field(
        default="xlnet",
        metadata={
            "help": "If training from scratch, pass a model type from the list: "
            f"{', '.join(MODEL_TYPES)}. If `model_path` is also provided, `model_path` "
            "takes preference."
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path. But `model_path` takes preference."
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path. If not provided, will be "
            "inferred from `model_path`. If `model_path` is not provided either you "
            "have to pass a tokenizer."
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from s3"
        },
    )


@dataclass
class EvalArguments:
    """
    Argumnts for model evaluation.

    eval_accumulation_steps (:obj:`int`, `optional`):
            Number of predictions steps to accumulate the output tensors for, before moving the results to the CPU. If
            left unset, the whole predictions are accumulated on GPU/TPU before being moved to the CPU (faster but
            requires more memory).
    """

    eval_file: str = field(metadata={"help": "Path to the data used for evaluation"})
    param_path: str = field(
        metadata={"help": "Path to the .json file with evaluation parameter"}
    )
