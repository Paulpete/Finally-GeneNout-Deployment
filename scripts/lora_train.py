# Minimal LoRA/PEFT trainer prototype.
# Requires: transformers, accelerate, datasets, peft
# Usage:
# python3 scripts/lora_train.py --train artifacts/cand-xxx_train.jsonl --output artifacts/cand-xxx_lora --epochs 1
import argparse
import os
import json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType

parser = argparse.ArgumentParser()
parser.add_argument('--train', type=str, required=True)
parser.add_argument('--output', type=str, required=True)
parser.add_argument('--model', type=str, default='gpt2')  # demo; use a small causal model
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--batch', type=int, default=8)
args = parser.parse_args()

ds = load_dataset('json', data_files=args.train)['train']

tokenizer = AutoTokenizer.from_pretrained(args.model)
if tokenizer.pad_token_id is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model = AutoModelForCausalLM.from_pretrained(args.model)

# LoRA config
config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"] if hasattr(model, 'get_input_embeddings') else None,
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, config)

def preprocess(ex):
    prompt = ex.get('prompt', '')
    completion = ex.get('completion', '')
    full = prompt + tokenizer.eos_token + completion + tokenizer.eos_token
    return tokenizer(full, truncation=True, max_length=512)

 ds = ds.map(preprocess, remove_columns=ds.column_names)
 ds.set_format(type='numpy', columns=['input_ids', 'attention_mask'])

training_args = TrainingArguments(
    output_dir=args.output,
    per_device_train_batch_size=args.batch,
    num_train_epochs=args.epochs,
    logging_steps=10,
    save_strategy="epoch",
    fp16=False,
    learning_rate=1e-4
)

trainer = Trainer(model=model, args=training_args, train_dataset=ds)
trainer.train()
model.save_pretrained(args.output)
print('Saved LoRA checkpoint to', args.output)
