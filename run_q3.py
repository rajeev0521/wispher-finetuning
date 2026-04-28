"""Run Q3 spelling verification on the full dataset."""
import sys
import os
import json

# Setup path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.spelling_checker import HindiSpellingChecker

# Initialize with frequency dictionary from the input word list
word_list_path = os.path.join(PROJECT_ROOT, 'data', 'input', 'unique_words.csv')
output_path = os.path.join(PROJECT_ROOT, 'results', 'q3_spelling_results.csv')

print("Initializing HindiSpellingChecker v2...")
checker = HindiSpellingChecker(
    word_list_path=word_list_path,
    freq_top_n=5000
)

# Run on full dataset
print(f"Processing {word_list_path}...")
stats = checker.check_csv(
    csv_path=word_list_path,
    output_csv_path=output_path,
    word_column='word'
)

print('\n' + '=' * 60)
print('Q3 RESULTS SUMMARY')
print('=' * 60)
print(json.dumps(stats, indent=2, ensure_ascii=False))

# Also generate new low-confidence review sample
import pandas as pd
import random

df = pd.read_csv(output_path)
low_conf = df[df['confidence'] == 'low']
print(f"\nLow confidence words: {len(low_conf)}")

if len(low_conf) > 0:
    sample_size = min(50, len(low_conf))
    random.seed(42)
    sample = low_conf.sample(n=sample_size, random_state=42)
    review_path = os.path.join(PROJECT_ROOT, 'results', 'q3_low_confidence_review.csv')
    sample.to_csv(review_path, index=False, encoding='utf-8-sig')
    print(f"Low confidence review sample saved to {review_path}")
