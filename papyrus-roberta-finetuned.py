import pandas as pd
import random
import torch
import nltk
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, InputExample, losses, models, datasets
from torch.utils.data import DataLoader
from nltk.corpus import stopwords
from keybert import KeyBERT

# Download French stopwords if not already available
nltk.download('stopwords')
french_stopwords = set(stopwords.words('french'))

# Paths
DATASET_FILE = "papyrus-f2.tsv"  # Ensure the correct path
FINE_TUNED_MODEL_PATH = "fine_tuned_xlm_roberta"
OUTPUT_FILE = "keywords_comparison_papyrus-f_finetuned.csv"

# Load dataset
df = pd.read_csv(DATASET_FILE, sep="\t")

# Load pre-trained sentence transformer
model = SentenceTransformer("sentence-transformers/all-distilroberta-v1")

# Prepare training data for fine-tuning
train_examples = []
for _, row in df.iterrows():
    text = row["sentences"]
    positive_keywords = row["label"].split(" , ")

    # Get a random subset of negative examples instead of storing all unique labels
    negative_keywords = random.sample(
        df.loc[df.index != _, "label"].str.split(" , ").explode().dropna().unique().tolist(),
        min(len(positive_keywords), 5)
    )

    for pos in positive_keywords:
        train_examples.append(InputExample(texts=[text, pos], label=1.0))
    for neg in negative_keywords:
        train_examples.append(InputExample(texts=[text, neg], label=0.0))

# Create DataLoader
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=1)

# Define loss function
train_loss = losses.MultipleNegativesRankingLoss(model)

# Fine-tune the model
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=1,
    warmup_steps=100
)

# Save the fine-tuned model
model.save(FINE_TUNED_MODEL_PATH)
print(f"Fine-tuned model saved to {FINE_TUNED_MODEL_PATH}")

# Use fine-tuned model in KeyBERT
fine_tuned_model = SentenceTransformer(FINE_TUNED_MODEL_PATH)
kw_model = KeyBERT(fine_tuned_model)
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
