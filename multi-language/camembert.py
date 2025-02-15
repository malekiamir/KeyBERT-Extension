import os
import numpy as np
import nltk
import pandas as pd
from keybert import KeyBERT
from transformers import AutoModel, AutoTokenizer
import torch
from nltk.corpus import stopwords
from rouge_score import rouge_scorer

# Download French stopwords if not already available
nltk.download('stopwords')
french_stopwords = set(stopwords.words('french'))

# Paths
DATASET_PATH = "WKC/docsutf8"
KEYS_PATH = "WKC/keys"
OUTPUT_FILE = "keywords_comparison.csv"

# Load CamemBERT model and tokenizer
camembert_model = AutoModel.from_pretrained("camembert-base")
camembert_tokenizer = AutoTokenizer.from_pretrained("camembert-base")


# Define a wrapper to get embeddings
class CamembertEmbedding:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, documents):
        """Generate document embeddings using CamemBERT."""
        if isinstance(documents, str):
            documents = [documents]  # Ensure input is a list

        inputs = self.tokenizer(documents, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Use mean of all token embeddings instead of [CLS]
        embeddings = outputs.last_hidden_state.mean(dim=1)  # Shape: (batch_size, hidden_dim)
        return embeddings.cpu().numpy()  # Convert to NumPy format for KeyBERT


# Initialize KeyBERT with CamemBERT
camembert_embedding = CamembertEmbedding(camembert_model, camembert_tokenizer)
kw_model = KeyBERT(camembert_embedding)
scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

# Store results
results = []
rouge_precision_sum = 0
rouge_recall_sum = 0
rouge_f1_sum = 0
num_files = 0


# Function to load ground truth keywords
def load_ground_truth(filename):
    key_file = os.path.join(KEYS_PATH, filename)
    if os.path.exists(key_file):
        with open(key_file, "r", encoding="utf-8") as file:
            return [line.strip() for line in file.readlines()]
    return []


# Process all .txt files in the dataset
def extract_keywords_and_compare():
    global rouge_precision_sum, rouge_recall_sum, rouge_f1_sum, num_files
    for filename in os.listdir(DATASET_PATH):
        if filename.endswith(".txt"):
            file_path = os.path.join(DATASET_PATH, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()

                # Extract keywords with custom French stopwords
                extracted_keywords = kw_model.extract_keywords(
                    text, keyphrase_ngram_range=(1, 3), stop_words=list(french_stopwords), top_n=5
                )
                extracted_keywords = [kw[0] for kw in extracted_keywords]

                # Load ground truth
                ground_truth = load_ground_truth(filename.replace(".txt", ".key")
                                                 .replace("docsutf8", "keys"))

                # Compute ROUGE-L scores
                extracted_text = " ".join(extracted_keywords)
                ground_truth_text = " ".join(ground_truth)
                rouge_l = scorer.score(ground_truth_text, extracted_text)["rougeL"]

                # Update sum for averages
                rouge_precision_sum += rouge_l.precision
                rouge_recall_sum += rouge_l.recall
                rouge_f1_sum += rouge_l.fmeasure
                num_files += 1

                # Store results
                results.append({
                    "filename": filename,
                    "extracted_keywords": ", ".join(extracted_keywords),
                    "ground_truth": ", ".join(ground_truth),
                    "rougeL_precision": rouge_l.precision,
                    "rougeL_recall": rouge_l.recall,
                    "rougeL_f1": rouge_l.fmeasure
                })


# Run extraction and comparison
extract_keywords_and_compare()

# Compute average ROUGE scores
if num_files > 0:
    avg_rouge_precision = rouge_precision_sum / num_files
    avg_rouge_recall = rouge_recall_sum / num_files
    avg_rouge_f1 = rouge_f1_sum / num_files
    print(f"Average ROUGE-L Precision: {avg_rouge_precision:.4f}")
    print(f"Average ROUGE-L Recall: {avg_rouge_recall:.4f}")
    print(f"Average ROUGE-L F1 Score: {avg_rouge_f1:.4f}")

# Save results to CSV
pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)
print(f"Keyword comparison results saved to {OUTPUT_FILE}")
