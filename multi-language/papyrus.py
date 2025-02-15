import os
import pandas as pd
import nltk
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords
from rouge_score import rouge_scorer

# Download French stopwords if not already available
nltk.download('stopwords')
french_stopwords = set(stopwords.words('french'))

# Paths
DATASET_FILE = "papyrus-f.tsv"  # Update with the correct file path if needed
OUTPUT_FILE = "keywords_comparison_papyrus-f.csv"

# Load the dataset
df = pd.read_csv(DATASET_FILE, sep="\t")

# Load XLM-R model for better French encoding
model = SentenceTransformer("sentence-transformers/all-distilroberta-v1")
kw_model = KeyBERT(model)
scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

# Store results
results = []
rouge_precision_sum = 0
rouge_recall_sum = 0
rouge_f1_sum = 0
num_rows = 0

# Process the dataset
for index, row in df.iterrows():
    text = row["sentences"]
    ground_truth = row["label"].split(" , ")  # Papyrus-f labels are comma-separated

    # Extract keywords
    extracted_keywords = kw_model.extract_keywords(
        text, keyphrase_ngram_range=(1, 3), stop_words=list(french_stopwords), top_n=5
    )
    extracted_keywords = [kw[0] for kw in extracted_keywords]

    # Compute ROUGE-L scores
    extracted_text = " ".join(extracted_keywords)
    ground_truth_text = " ".join(ground_truth)
    rouge_l = scorer.score(ground_truth_text, extracted_text)["rougeL"]

    # Update sum for averages
    rouge_precision_sum += rouge_l.precision
    rouge_recall_sum += rouge_l.recall
    rouge_f1_sum += rouge_l.fmeasure
    num_rows += 1

    # Store results
    results.append({
        "index": row["index"],
        "extracted_keywords": ", ".join(extracted_keywords),
        "ground_truth": ", ".join(ground_truth),
        "rougeL_precision": rouge_l.precision,
        "rougeL_recall": rouge_l.recall,
        "rougeL_f1": rouge_l.fmeasure
    })

# Compute average ROUGE scores
if num_rows > 0:
    avg_rouge_precision = rouge_precision_sum / num_rows
    avg_rouge_recall = rouge_recall_sum / num_rows
    avg_rouge_f1 = rouge_f1_sum / num_rows
    print(f"Average ROUGE-L Precision: {avg_rouge_precision:.4f}")
    print(f"Average ROUGE-L Recall: {avg_rouge_recall:.4f}")
    print(f"Average ROUGE-L F1 Score: {avg_rouge_f1:.4f}")

# Save results to CSV
pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)
print(f"Keyword comparison results saved to {OUTPUT_FILE}")
