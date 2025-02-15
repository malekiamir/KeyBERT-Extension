import os
import nltk
import pandas as pd
import torch
from keybert import KeyBERT
from transformers import XLMRobertaTokenizer
from sentence_transformers import SentenceTransformer, InputExample, losses, models
from torch.utils.data import DataLoader
from nltk.corpus import stopwords
from rouge_score import rouge_scorer

# Download French stopwords
nltk.download('stopwords')
french_stopwords = set(stopwords.words('french'))

# Paths
DATASET_PATH = "WKC/docsutf8"
KEYS_PATH = "WKC/keys"
FINE_TUNED_MODEL_PATH = "fine_tuned_xlm_roberta"
OUTPUT_FILE = "keywords_comparison.csv"

# Load tokenizer and define Sentence-Transformers model
model_name = "xlm-roberta-base"
tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
word_embedding_model = models.Transformer(model_name)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# Prepare training data
train_examples = []
for filename in os.listdir(DATASET_PATH):
    if filename.endswith(".txt"):
        file_path = os.path.join(DATASET_PATH, filename)
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()

        ground_truth_file = os.path.join(KEYS_PATH, filename.replace(".txt", ".key"))
        if os.path.exists(ground_truth_file):
            with open(ground_truth_file, "r", encoding="utf-8") as key_file:
                keywords = [line.strip() for line in key_file.readlines()]

            # Create positive pairs (text, keyword)
            for keyword in keywords:
                train_examples.append(InputExample(texts=[text, keyword], label=1.0))

# Create DataLoader
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=1)
train_loss = losses.MultipleNegativesRankingLoss(model)

# Fine-tune the model
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=2, warmup_steps=50)

# Save the fine-tuned model
model.save(FINE_TUNED_MODEL_PATH)
print(f"Fine-tuned model saved to {FINE_TUNED_MODEL_PATH}")

# Load fine-tuned model for keyword extraction
fine_tuned_model = SentenceTransformer(FINE_TUNED_MODEL_PATH)
kw_model = KeyBERT(fine_tuned_model)
scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

# Store results
results = []
rouge_precision_sum = 0
rouge_recall_sum = 0
rouge_f1_sum = 0
num_files = 0


# Function to read ground truth keywords
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

                # Extract keywords
                extracted_keywords = kw_model.extract_keywords(
                    text, keyphrase_ngram_range=(1, 3), stop_words=list(french_stopwords), top_n=5
                )
                extracted_keywords = [kw[0] for kw in extracted_keywords]

                # Load ground truth
                ground_truth = load_ground_truth(filename.replace(".txt", ".key"))

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