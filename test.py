from transformers import BertTokenizer, BertForSequenceClassification, GPT2LMHeadModel, GPT2Tokenizer
import torch
import time
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import psutil
import os
import json

# Load the trained BERT tokenizer and model for sequence classification
try:
    bert_tokenizer = BertTokenizer.from_pretrained('./custom_bert_results/bert')
    bert_model = BertForSequenceClassification.from_pretrained('./custom_bert_results/bert')
    print("Loaded custom BERT model and tokenizer.")
except OSError:
    print("Custom BERT model not found. Loading pre-trained BERT model.")
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Load the fine-tuned GPT-2 tokenizer and model
try:
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained('./custom_gpt2_results/gpt2')
    gpt2_model = GPT2LMHeadModel.from_pretrained('./custom_gpt2_results/gpt2')
    print("Loaded custom GPT-2 model and tokenizer.")
except OSError:
    print("Custom GPT-2 model not found. Loading pre-trained GPT-2 model.")
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')

def predict_nsp(model, tokenizer, sentence_a, sentence_b):
    inputs = tokenizer(sentence_a, sentence_b, return_tensors='pt', truncation=True, padding='max_length', max_length=128)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=1)[0]
    
    prediction = torch.argmax(probs).item()
    return prediction, probs[prediction].item()

def generate_next_sentence(model, tokenizer, input_text, max_length=50):
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
    
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    
    start_time = time.time()
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        pad_token_id=pad_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7
    )
    inference_time = time.time() - start_time
    
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    next_sentence = generated_text[len(input_text):].strip()
    
    return next_sentence, inference_time

def evaluate_bert(model, tokenizer, sentence_pairs):
    model.eval()
    predictions = []
    labels = []
    inference_times = []
    
    for sentence_a, sentence_b, label in tqdm(sentence_pairs, desc="Evaluating BERT"):
        start_time = time.time()
        prediction, _ = predict_nsp(model, tokenizer, sentence_a, sentence_b)
        inference_time = time.time() - start_time
        
        predictions.append(prediction)
        labels.append(label)
        inference_times.append(inference_time)
    
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    precision = precision_score(labels, predictions, average='weighted')
    recall = recall_score(labels, predictions, average='weighted')
    auc_roc = roc_auc_score(labels, predictions, average='weighted')
    avg_inference_time = np.mean(inference_times)
    throughput = 1 / avg_inference_time
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'auc_roc': auc_roc,
        'avg_inference_time': avg_inference_time,
        'throughput': throughput
    }

def evaluate_gpt2_perplexity(model, tokenizer, texts):
    model.eval()
    total_loss = 0
    total_tokens = 0
    inference_times = []

    with torch.no_grad():
        for text in tqdm(texts, desc="Evaluating GPT-2 Perplexity"):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            
            start_time = time.time()
            outputs = model(**inputs, labels=inputs["input_ids"])
            inference_time = time.time() - start_time
            
            total_loss += outputs.loss.item() * inputs["input_ids"].size(1)
            total_tokens += inputs["input_ids"].size(1)
            inference_times.append(inference_time)

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    avg_inference_time = np.mean(inference_times)
    throughput = 1 / avg_inference_time
    
    return perplexity, avg_inference_time, throughput

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # in MB

def calculate_and_save_metrics(bert_results, gpt2_results, bert_model, gpt2_model):
    # Get current memory usage
    current_memory_usage = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)  # in MB

    # Calculate error rate
    error_rate = 1 - bert_results['accuracy']

    metrics = {
        "performance_metrics": {
            "accuracy": bert_results['accuracy'],
            "error_rate": error_rate,
            "precision": bert_results['precision'],
            "recall": bert_results['recall'],
            "f1_score": bert_results['f1_score'],
            "auc_roc": bert_results['auc_roc']
        },
        "inference_metrics": {
            "avg_inference_time": bert_results['avg_inference_time'],
            "throughput": bert_results['throughput']
        },
        "resource_utilization": {
            "bert_model_size": sum(p.numel() for p in bert_model.parameters()) * 4 / (1024 * 1024),  # Size in MB
            "gpt2_model_size": sum(p.numel() for p in gpt2_model.parameters()) * 4 / (1024 * 1024),  # Size in MB
            "memory_usage": current_memory_usage
        }
    }

    # Add GPT-2 specific metrics
    metrics["gpt2_metrics"] = {
        "perplexity": gpt2_results['perplexity'],
        "avg_inference_time": gpt2_results['avg_inference_time'],
        "throughput": gpt2_results['throughput']
    }

    # Save metrics to JSON file
    with open('current_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)

    print("\nMetrics have been updated and saved to current_metrics.json")
    print(f"Accuracy: {metrics['performance_metrics']['accuracy']:.4f}")
    print(f"Error Rate: {metrics['performance_metrics']['error_rate']:.4f}")

if __name__ == "__main__":
    # Prepare test data
    sentence_pairs = [
        ("He is a talented musician.", "He can play multiple instruments.", 1),
        ("The sun is shining.", "It's a beautiful day.", 1),
        ("I love programming.", "It's a challenging field.", 1),
        ("She is a doctor.", "She helps people get better.", 1),
        ("The earth is round.", "Gravity keeps us on the ground.", 0),
        ("I enjoy reading.", "Books are a great source of knowledge.", 1),
        ("The sky is blue.", "The ocean is blue.", 0),
        ("She is a teacher.", "She teaches students.", 1),
        ("The car is fast.", "The car can go very fast.", 1),
        ("I love pizza.", "Pizza is a delicious food.", 1),
    ]   

    test_texts = [
        "The quick brown fox jumps over the lazy dog. This sentence is often used for testing.",
        "Artificial intelligence is rapidly advancing. Many industries are being transformed by AI.",
        "Climate change is a global concern. Scientists are working on solutions to mitigate its effects.",
        "The Internet has revolutionized communication. It has connected people across the world.",
        "Space exploration continues to fascinate humanity. New discoveries are made regularly."
    ]

    print("Evaluating BERT model:")
    initial_memory = get_memory_usage()
    bert_results = evaluate_bert(bert_model, bert_tokenizer, sentence_pairs)
    final_memory = get_memory_usage()
    bert_memory_usage = final_memory - initial_memory

    print(f"BERT Accuracy: {bert_results['accuracy']:.4f}")
    print(f"BERT F1 Score: {bert_results['f1_score']:.4f}")
    print(f"BERT Precision: {bert_results['precision']:.4f}")
    print(f"BERT Recall: {bert_results['recall']:.4f}")
    print(f"BERT Average Inference Time: {bert_results['avg_inference_time']:.5f} seconds")
    print(f"BERT Memory Usage: {bert_memory_usage:.2f} MB")

    print("\nEvaluating GPT-2 model:")
    initial_memory = get_memory_usage()
    gpt2_perplexity, gpt2_avg_inference_time, gpt2_throughput = evaluate_gpt2_perplexity(gpt2_model, gpt2_tokenizer, test_texts)
    final_memory = get_memory_usage()
    gpt2_memory_usage = final_memory - initial_memory

    print(f"GPT-2 Perplexity: {gpt2_perplexity:.4f}")
    print(f"GPT-2 Average Inference Time: {gpt2_avg_inference_time:.5f} seconds")
    print(f"GPT-2 Memory Usage: {gpt2_memory_usage:.2f} MB")

    print("\nGenerating text for each test case:")
    for i, (sentence_a, sentence_b, _) in enumerate(sentence_pairs, 1):
        print(f"\nTest Case {i}:")
        print(f"Input: {sentence_a}")
        generated_text, generation_time = generate_next_sentence(gpt2_model, gpt2_tokenizer, sentence_a, max_length=100)
        print(f"Generated text: {generated_text}")
        print(f"Generation time: {generation_time:.5f} seconds")

    print("\nNote: For Training Time, Training Loss, and Validation Loss, please refer to the logs from your training script.")

    # After all evaluations, call the new function
    gpt2_results = {
        'perplexity': gpt2_perplexity,
        'avg_inference_time': gpt2_avg_inference_time,
        'throughput': gpt2_throughput
    }

    calculate_and_save_metrics(bert_results, gpt2_results, bert_model, gpt2_model)