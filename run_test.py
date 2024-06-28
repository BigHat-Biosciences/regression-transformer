import os
from terminator.args import CustomTrainingArguments, ModelArguments
from terminator.collators import TRAIN_COLLATORS
from terminator.datasets import get_dataset
from terminator.tokenization import ExpressionBertTokenizer
from terminator.trainer import CustomTrainer, get_trainer_dict
from terminator.utils import get_latest_checkpoint

if __name__ == "__main__":
    print("Hello World!")
    print(os.environ)
