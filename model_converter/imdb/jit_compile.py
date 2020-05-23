import os
import sys
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import torch
import numpy as np
from transformers import ElectraTokenizer, ElectraConfig

from model import ElectraForSequenceClassification


parser = argparse.ArgumentParser()
# NOTE This should be same as the setting of the android!!!
parser.add_argument("--max_seq_len", default=40, type=int, help="Maximum sequence length")
args = parser.parse_args()

# 1. Convert model
tokenizer = ElectraTokenizer.from_pretrained("monologg/electra-small-finetuned-imdb")

model = ElectraForSequenceClassification.from_pretrained("monologg/electra-small-finetuned-imdb", torchscript=True)
model.eval()

input_ids = torch.tensor([[0] * args.max_seq_len], dtype=torch.long)
print(input_ids.size())
traced_model = torch.jit.trace(
    model,
    input_ids
)
torch.jit.save(traced_model, "app/src/main/assets/imdb_small.pt")

# 2. Testing...
# Tokenize input text
text = "This movie is awesome lol!"
encode_inputs = tokenizer.encode_plus(
    text,
    return_tensors="pt",
    max_length=args.max_seq_len,
    pad_to_max_length=True
)
print(encode_inputs)
print(encode_inputs["input_ids"].size())

# Load model
loaded_model = torch.jit.load("app/src/main/assets/imdb_small.pt")
loaded_model.eval()
with torch.no_grad():
    outputs = loaded_model(encode_inputs["input_ids"])
print(outputs)
with torch.no_grad():
    outputs = model(encode_inputs["input_ids"])
print(outputs)
