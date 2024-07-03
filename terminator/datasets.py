from transformers import LineByLineTextDataset, PreTrainedTokenizer, TextDataset
from typing import List, Dict
import torch


def get_dataset(
    filepath: str,
    tokenizer: PreTrainedTokenizer,
    block_size: int,
    line_by_line: bool = True,
):
    if line_by_line:
        return LineByLineTextDataset(
            tokenizer=tokenizer, file_path=filepath, block_size=block_size
        )
    else:
        return TextDataset(
            tokenizer=tokenizer,
            file_path=filepath,
            block_size=block_size,
        )


class TextDatasetFromList(TextDataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, lines: List[str], block_size: int):
        batch_encoding = tokenizer(lines, add_special_tokens=True, truncation=True, max_length=block_size)
        self.examples = batch_encoding["input_ids"]
        self.examples = [{"input_ids": torch.tensor(e, dtype=torch.long)} for e in self.examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        return self.examples[i]
