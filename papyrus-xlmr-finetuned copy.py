import os
import nltk
import pandas as pd
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers import losses
from torch.utils.data import DataLoader
from rouge_score import rouge_scorer
from nltk.corpus import stopwords

# Download French stopwords if not already available
nltk.download('stopwords')
french_stopwords = set(stopwords.words('french'))

# Paths
DATASET_PATH = "papyrus-f.tsv"
OUTPUT_FILE = "keywords_comparison_papyrus.csv"

# Load the pre-trained XLM-R model
model = SentenceTransformer("xlm-roberta-base")
kw_model = KeyBERT(model)  # Use KeyBERT with the XLM-R model
scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

# Store results
results = []
rouge_precision_sum = 0
rouge_recall_sum = 0
rouge_f1_sum = 0
num_files = 0

# Load dataset
def load_dataset():
    df = pd.read_csv(DATASET_PATH, sep='\t')  # Read TSV file
    return df[['sentences', 'label']]

# Prepare training data
def prepare_training_data(df):
    train_data = []
    for _, row in df.iterrows():
        text = row['sentences']
        ground_truth_keywords = row['label'].split(', ')  # Assuming comma-separated keywords

        # Extract keywords with KeyBERT
        extracted_keywords = kw_model.extract_keywords(
            text, keyphrase_ngram_range=(1, 2), stop_words=list(french_stopwords), top_n=5
        )
        extracted_keywords = [kw[0] for kw in extracted_keywords]

        # Generate training data: pairs of text and keywords
        for keyword in extracted_keywords:
            if keyword in ground_truth_keywords:
                train_data.append(InputExample(texts=[text, keyword], label=1.0))  # Similar pair
            else:
                train_data.append(InputExample(texts=[text, keyword], label=0.0))  # Non-similar pair
    return train_data

# Fine-tune the model
def fine_tune_model():
    df = load_dataset()
    train_data = prepare_training_data(df)
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=16)
    train_loss = losses.CosineSimilarityLoss(model)

    # Fine-tune the model
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=1,
        warmup_steps=100,
        output_path='./fine_tuned_papyrus_model'
    )

# Evaluate the model
def extract_keywords_and_compare():
    global rouge_precision_sum, rouge_recall_sum, rouge_f1_sum, num_files
    model = SentenceTransformer('./fine_tuned_papyrus_model')
    df = load_dataset()

    for _, row in df.iterrows():
        text = row['sentences']
        ground_truth_keywords = row['label'].split(', ')

        # Extract keywords with fine-tuned KeyBERT
        extracted_keywords = kw_model.extract_keywords(
            text, keyphrase_ngram_range=(1, 2), stop_words=list(french_stopwords), top_n=5
        )
        extracted_keywords = [kw[0] for kw in extracted_keywords]

        # Compute ROUGE-L scores
        extracted_text = " ".join(extracted_keywords)
        ground_truth_text = " ".join(ground_truth_keywords)
        rouge_l = scorer.score(ground_truth_text, extracted_text)["rougeL"]

        # Update sum for averages
        rouge_precision_sum += rouge_l.precision
        rouge_recall_sum += rouge_l.recall
        rouge_f1_sum += rouge_l.fmeasure
        num_files += 1

        # Store results
        results.append({
            "text": text,
            "extracted_keywords": ", ".join(extracted_keywords),
            "ground_truth": ", ".join(ground_truth_keywords),
            "rougeL_precision": rouge_l.precision,
            "rougeL_recall": rouge_l.recall,
            "rougeL_f1": rouge_l.fmeasure
        })

# Main function
def main():
    fine_tune_model()
    extract_keywords_and_compare()

    # Compute average ROUGE scores
    if num_files > 0:
        avg_rouge_precision = rouge_precision_sum / num_files
        avg_rouge_recall = rouge_recall_sum / num_files
        avg_rouge_f1 = rouge_f1_sum / num_files
        print(f"Average ROUGE-L Precision: {avg_rouge_precision:.4f}")
        print(f"Average ROUGE-L Recall: {avg_rouge_recall:.4f}")
        print(f"Average ROUGE-L F1 Score: {avg_rouge_f1:.4f}")

    # Save results
    pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)
    print(f"Keyword comparison results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
