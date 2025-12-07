# fine_tune.py
# Example fine-tuning script for GPT-2 (not required to run for the assignment)

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.utils.data import DataLoader

# Load GPT-2
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")D

# GPT-2 has no pad token → set it to eos_token
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")

# Dummy dataset example (SQuAD-style formatting)
def format_example(question, answer):
    return f"Question: {question}\nAnswer: That is a great question. {answer} Let me know if you have any other questions."

# Example formatted pair
examples = [
    format_example("Why is the sky blue?", "The sky is blue due to Rayleigh scattering.")
]

# Tokenize
inputs = tokenizer(examples, return_tensors="pt", padding=True, truncation=True)
dataset = [(inputs["input_ids"], inputs["attention_mask"])]

dataloader = DataLoader(dataset, batch_size=1)

optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

# Training loop (conceptual, not required to run)
for epoch in range(1):
    for input_ids, attention_mask in dataloader:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

print("Fine-tuning step completed (conceptually).")
