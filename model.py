from transformers import BertTokenizer, BertConfig, BertForSequenceClassification, GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import torch
import numpy as np
from torch import nn
from sklearn.metrics import accuracy_score, f1_score
import time
import psutil
import os
from datasets import Dataset
from sklearn.utils import resample
import pandas as pd

# Custom BERT Configuration and Model Definition
bert_config = BertConfig(
    vocab_size=30522,
    hidden_size=256,  # Reduced from 768
    num_hidden_layers=6,  # Reduced from 12
    num_attention_heads=8,  # Reduced from 12
    intermediate_size=1024,  # Reduced from 3072
    max_position_embeddings=512,
    type_vocab_size=2,
    num_labels=2,
)

# Define custom BERT model for sequence classification
bert_model = BertForSequenceClassification(bert_config)

# Load Tokenizers
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', tokenization_spaces=True)
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

# Load and preprocess the WikiText dataset
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')

# Function to prepare sentence pairs from texts
def prepare_sentence_pairs(texts, tokenizer, max_length=128):
    sentence_pairs = []
    for text in texts:
        sentences = text.split(". ")
        for i in range(len(sentences) - 1):
            sentence_a = sentences[i]
            sentence_b = sentences[i + 1]
            if sentence_a and sentence_b:
                # Tokenize sentences separately
                tokens_a = tokenizer.tokenize(sentence_a)
                tokens_b = tokenizer.tokenize(sentence_b)
                
                # Truncate
                total_length = len(tokens_a) + len(tokens_b)
                if total_length > max_length - 3:  # Account for [CLS], [SEP], [SEP]
                    if len(tokens_a) > len(tokens_b):
                        tokens_a = tokens_a[:(max_length - 3 - len(tokens_b))]
                    else:
                        tokens_b = tokens_b[:(max_length - 3 - len(tokens_a))]
                
                # Combine tokens
                tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']
                segment_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)
                
                # Convert to ids and pad
                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                attention_mask = [1] * len(input_ids)
                padding_length = max_length - len(input_ids)
                input_ids = input_ids + ([0] * padding_length)
                attention_mask = attention_mask + ([0] * padding_length)
                segment_ids = segment_ids + ([0] * padding_length)
                
                # Remove assertions and ensure all sequences are of max_length
                input_ids = input_ids[:max_length]
                attention_mask = attention_mask[:max_length]
                segment_ids = segment_ids[:max_length]
                
                encoded = {
                    'input_ids': torch.tensor(input_ids),
                    'attention_mask': torch.tensor(attention_mask),
                    'token_type_ids': torch.tensor(segment_ids),
                    'labels': torch.tensor([0])
                }
                sentence_pairs.append(encoded)
            
            # Add negative example (non-consecutive sentences)
            if i + 2 < len(sentences):
                sentence_b = sentences[i + 2]
                if sentence_a and sentence_b:
                    # Tokenize sentences separately
                    tokens_a = tokenizer.tokenize(sentence_a)
                    tokens_b = tokenizer.tokenize(sentence_b)
                    
                    # Truncate
                    total_length = len(tokens_a) + len(tokens_b)
                    if total_length > max_length - 3:  # Account for [CLS], [SEP], [SEP]
                        if len(tokens_a) > len(tokens_b):
                            tokens_a = tokens_a[:(max_length - 3 - len(tokens_b))]
                        else:
                            tokens_b = tokens_b[:(max_length - 3 - len(tokens_a))]
                    
                    # Combine tokens
                    tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']
                    segment_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)
                    
                    # Convert to ids and pad
                    input_ids = tokenizer.convert_tokens_to_ids(tokens)
                    attention_mask = [1] * len(input_ids)
                    padding_length = max_length - len(input_ids)
                    input_ids = input_ids + ([0] * padding_length)
                    attention_mask = attention_mask + ([0] * padding_length)
                    segment_ids = segment_ids + ([0] * padding_length)
                    
                    # Remove assertions and ensure all sequences are of max_length
                    input_ids = input_ids[:max_length]
                    attention_mask = attention_mask[:max_length]
                    segment_ids = segment_ids[:max_length]
                    
                    encoded = {
                        'input_ids': torch.tensor(input_ids),
                        'attention_mask': torch.tensor(attention_mask),
                        'token_type_ids': torch.tensor(segment_ids),
                        'labels': torch.tensor([1])
                    }
                    sentence_pairs.append(encoded)

    return Dataset.from_list(sentence_pairs)

# Prepare datasets for training and validation
train_data = dataset['train']['text'][:1000]  # Reduced from 5000
val_data = dataset['validation']['text'][:200]  # Reduced from 1000
train_dataset = prepare_sentence_pairs(train_data, bert_tokenizer)
val_dataset = prepare_sentence_pairs(val_data, bert_tokenizer)

# Define a custom data collator
from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=bert_tokenizer)

# Define a custom evaluation function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    return {"accuracy": accuracy, "f1": f1}

# Modify Training Arguments
training_args = TrainingArguments(
    output_dir='./custom_bert_results',
    num_train_epochs=3,  # Reduced from 10
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_dir='./custom_logs',
    logging_steps=100,
    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    warmup_steps=1000,
    weight_decay=0.01,
    learning_rate=5e-5,
)

# Define Trainer for Custom Model
trainer = Trainer(
    model=bert_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train and evaluate the model
if __name__ == "__main__":
    start_time = time.time()
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024
    
    train_result = trainer.train()
    
    training_time = time.time() - start_time
    
    final_memory = process.memory_info().rss / 1024 / 1024
    memory_used = final_memory - initial_memory
    
    eval_results = trainer.evaluate()
    
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Memory usage: {memory_used:.2f} MB")
    print(f"Final evaluation results: {eval_results}")
    print(f"Training loss: {train_result.training_loss}")
    
    bert_model.save_pretrained('./custom_bert_results/bert')
    bert_tokenizer.save_pretrained('./custom_bert_results/bert')

# GPT-2 fine-tuning
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
gpt2_model.resize_token_embeddings(len(gpt2_tokenizer))

def prepare_gpt2_dataset(texts, max_length=256):
    gpt2_dataset = []
    for text in texts:
        encoded = gpt2_tokenizer(text, truncation=True, max_length=max_length, padding='max_length', return_tensors='pt')
        encoded['labels'] = encoded['input_ids'].clone()
        gpt2_dataset.append({key: value.squeeze() for key, value in encoded.items()})
    return Dataset.from_list(gpt2_dataset)

gpt2_train_dataset = prepare_gpt2_dataset(train_data)
gpt2_val_dataset = prepare_gpt2_dataset(val_data)

gpt2_training_args = TrainingArguments(
    output_dir='./custom_gpt2_results',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    logging_dir='./gpt2_logs',
    logging_steps=50,
    evaluation_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    load_best_model_at_end=True,
    warmup_steps=500,
    weight_decay=0.01,
    learning_rate=5e-5,
)

gpt2_trainer = Trainer(
    model=gpt2_model,
    args=gpt2_training_args,
    train_dataset=gpt2_train_dataset,
    eval_dataset=gpt2_val_dataset,
    data_collator=lambda data: {'input_ids': torch.stack([f['input_ids'] for f in data]),
                                'attention_mask': torch.stack([f['attention_mask'] for f in data]),
                                'labels': torch.stack([f['labels'] for f in data])},
)

def balance_dataset(dataset):
    df = pd.DataFrame(dataset)
    df_majority = df[df.labels==0]
    df_minority = df[df.labels==1]
    
    # Upsample minority class
    df_minority_upsampled = resample(df_minority, 
                                     replace=True,     # sample with replacement
                                     n_samples=len(df_majority),    # to match majority class
                                     random_state=123) # reproducible results
    
    # Combine majority class with upsampled minority class
    df_upsampled = pd.concat([df_majority, df_minority_upsampled])
    
    return Dataset.from_pandas(df_upsampled)

# After preparing your dataset
train_dataset = balance_dataset(train_dataset)