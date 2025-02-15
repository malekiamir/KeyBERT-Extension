import os
import nltk
import pandas as pd
import stanza
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords
from rouge_score import rouge_scorer
import ssl
from collections import Counter

ssl._create_default_https_context = ssl._create_unverified_context

nltk.download('stopwords')
french_stopwords = set(stopwords.words('french'))

# Initialize Stanza NLP Pipeline for French (Now with Named Entity Recognition)
stanza.download("fr")
nlp = stanza.Pipeline("fr", processors="tokenize,mwt,pos,lemma,ner")  # Added NER

# Paths
DATASET_PATH = "WKC/docsutf8"
KEYS_PATH = "WKC/keys"
OUTPUT_FILE = "keywords_comparison.csv"

# Load multilingual model
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
kw_model = KeyBERT(model)
scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


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


# Function to lemmatize text using Stanza and extract named entities
def preprocess_text(text):
    doc = nlp(text)

    # Extract lemmatized words (except stopwords)
    lemmatized_tokens = [word.lemma for sent in doc.sentences for word in sent.words]
    cleaned_text = " ".join([token for token in lemmatized_tokens if token.lower() not in french_stopwords])

    # Extract named entities and count their frequency
    named_entity_list = [ent.text for ent in doc.ents]
    entity_counts = Counter(named_entity_list)

    # Keep the **top 3 most frequent** named entities
    most_frequent_entities = {entity for entity, count in entity_counts.most_common(3)}

    return cleaned_text, most_frequent_entities  # Return both


# Process all .txt files in the dataset
def extract_keywords_and_compare():
    global rouge_precision_sum, rouge_recall_sum, rouge_f1_sum, num_files
    for filename in os.listdir(DATASET_PATH):
        if filename.endswith(".txt"):
            file_path = os.path.join(DATASET_PATH, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()

                # Apply lemmatization and extract named entities
                lemmatized_text, named_entities = preprocess_text(text)

                # Extract keywords with KeyBERT (without removing stopwords yet)
                extracted_keywords = kw_model.extract_keywords(
                    lemmatized_text, keyphrase_ngram_range=(1, 3), top_n=5
                )
                extracted_keywords = [kw[0] for kw in extracted_keywords]
                for entity in named_entities:
                    if entity.lower() not in extracted_keywords:
                        extracted_keywords.append(entity)  # Add missing entities

                ground_truth = load_ground_truth(filename.replace(".txt", ".key").replace("docsutf8", "keys"))

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
