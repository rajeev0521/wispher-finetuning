"""
Lattice Builder for ASR Evaluation
====================================
Constructs lattices from multiple ASR model outputs and computes
lattice-based WER that is fairer to models when reference has errors.
"""

import re
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Set
from collections import Counter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# TEXT NORMALIZATION FOR ALIGNMENT
# ============================================================================

def normalize_for_alignment(text: str) -> str:
    """
    Normalize text for alignment comparison.
    Removes punctuation, normalizes whitespace, lowercases English.
    """
    if not isinstance(text, str):
        return ""
    # Remove common punctuation
    text = re.sub(r'[।,?!;:\.\-–—\"\'""''…\(\)\[\]\{\}]', ' ', text)
    # Lowercase English
    text = text.lower()
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize(text: str) -> List[str]:
    """Split text into word tokens."""
    return normalize_for_alignment(text).split()


# ============================================================================
# PAIRWISE ALIGNMENT (Needleman-Wunsch)
# ============================================================================

def edit_distance_normalized(w1: str, w2: str) -> float:
    """
    Compute normalized edit distance between two words.
    Returns value in [0, 1]. 0 = identical, 1 = completely different.
    """
    if w1 == w2:
        return 0.0
    
    m, n = len(w1), len(w2)
    if m == 0 or n == 0:
        return 1.0
    
    # Standard edit distance DP
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if w1[i-1] == w2[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,      # deletion
                dp[i][j-1] + 1,      # insertion
                dp[i-1][j-1] + cost  # substitution
            )
    
    return dp[m][n] / max(m, n)


def needleman_wunsch(seq1: List[str], seq2: List[str], 
                      gap_penalty: float = 1.0) -> List[Tuple[Optional[str], Optional[str]]]:
    """
    Pairwise sequence alignment using Needleman-Wunsch algorithm.
    
    Args:
        seq1: First word sequence
        seq2: Second word sequence  
        gap_penalty: Cost of insertion/deletion
    
    Returns:
        List of aligned pairs (word1, word2), None indicates a gap
    """
    m, n = len(seq1), len(seq2)
    
    # Score matrix
    score = np.zeros((m + 1, n + 1))
    for i in range(1, m + 1):
        score[i][0] = i * gap_penalty
    for j in range(1, n + 1):
        score[0][j] = j * gap_penalty
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            match_cost = edit_distance_normalized(seq1[i-1], seq2[j-1])
            score[i][j] = min(
                score[i-1][j-1] + match_cost,  # match/substitute
                score[i-1][j] + gap_penalty,      # delete from seq1
                score[i][j-1] + gap_penalty,      # insert from seq2
            )
    
    # Traceback
    alignment = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            match_cost = edit_distance_normalized(seq1[i-1], seq2[j-1])
            if score[i][j] == score[i-1][j-1] + match_cost:
                alignment.append((seq1[i-1], seq2[j-1]))
                i -= 1
                j -= 1
                continue
        if i > 0 and score[i][j] == score[i-1][j] + gap_penalty:
            alignment.append((seq1[i-1], None))
            i -= 1
        else:
            alignment.append((None, seq2[j-1]))
            j -= 1
    
    alignment.reverse()
    return alignment


# ============================================================================
# MULTIPLE SEQUENCE ALIGNMENT (Progressive)
# ============================================================================

def align_multiple_sequences(sequences: List[List[str]]) -> List[List[Optional[str]]]:
    """
    Progressive multiple sequence alignment.
    Aligns all sequences to the first (human reference) iteratively.
    
    Args:
        sequences: List of word sequences [human_ref, model_1, model_2, ...]
    
    Returns:
        Matrix where each row is an aligned sequence, padded with None for gaps
    """
    if len(sequences) == 0:
        return []
    if len(sequences) == 1:
        return [list(sequences[0])]
    
    # Start with the first sequence (human reference)
    master_alignment = [[w] for w in sequences[0]]
    
    for seq_idx in range(1, len(sequences)):
        seq = sequences[seq_idx]
        
        # Extract the current master consensus
        master_words = []
        for position in master_alignment:
            # Use the most common non-None word at this position
            words = [w for w in position if w is not None]
            if words:
                master_words.append(Counter(words).most_common(1)[0][0])
            else:
                master_words.append(None)
        
        master_clean = [w for w in master_words if w is not None]
        
        # Align new sequence to master
        alignment = needleman_wunsch(master_clean, seq)
        
        # Merge into master alignment
        new_master = []
        master_idx = 0
        
        for w_master, w_new in alignment:
            if w_master is not None:
                # Find corresponding position in master_alignment
                while master_idx < len(master_alignment):
                    master_pos_words = [w for w in master_alignment[master_idx] if w is not None]
                    if master_pos_words and master_pos_words[0] == w_master:
                        new_master.append(master_alignment[master_idx] + [w_new])
                        master_idx += 1
                        break
                    else:
                        # Gap position in master — add None for new sequence
                        new_master.append(master_alignment[master_idx] + [None])
                        master_idx += 1
            else:
                # Insertion in new sequence — create new position
                padding = [None] * (seq_idx)
                new_master.append(padding + [w_new])
        
        # Add remaining master positions
        while master_idx < len(master_alignment):
            new_master.append(master_alignment[master_idx] + [None])
            master_idx += 1
        
        master_alignment = new_master
    
    return master_alignment


# ============================================================================
# LATTICE CONSTRUCTION
# ============================================================================

def build_lattice(
    human_ref: str,
    model_outputs: List[str],
    trust_threshold: int = 3,
) -> List[Set[str]]:
    """
    Construct a lattice from human reference and multiple model outputs.
    
    Each position in the lattice is a set of valid word alternatives.
    
    Args:
        human_ref: Human reference transcription
        model_outputs: List of model output transcriptions
        trust_threshold: Minimum models that must agree to potentially override reference
    
    Returns:
        List of sets, each set containing valid word alternatives for that position
    """
    # Tokenize all sequences
    ref_tokens = tokenize(human_ref)
    model_tokens_list = [tokenize(m) for m in model_outputs]
    
    # All sequences: [reference, model_1, model_2, ...]
    all_sequences = [ref_tokens] + model_tokens_list
    
    # Multiple sequence alignment
    aligned = align_multiple_sequences(all_sequences)
    
    # Build lattice bins
    lattice = []
    for position in aligned:
        bin_words = set()
        
        for word in position:
            if word is not None and word.strip():
                bin_words.add(word.strip())
        
        if bin_words:
            lattice.append(bin_words)
    
    return lattice


def lattice_to_string(lattice: List[Set[str]]) -> str:
    """Convert lattice to readable string representation."""
    bins = []
    for bin_set in lattice:
        if len(bin_set) == 1:
            bins.append(list(bin_set)[0])
        else:
            alternatives = '|'.join(sorted(bin_set))
            bins.append(f"[{alternatives}]")
    return ' '.join(bins)


# ============================================================================
# LATTICE-BASED WER COMPUTATION
# ============================================================================

def compute_standard_wer(reference: str, hypothesis: str) -> Dict:
    """
    Compute standard WER between reference and hypothesis.
    """
    ref_words = tokenize(reference)
    hyp_words = tokenize(hypothesis)
    
    # DP alignment
    m, n = len(ref_words), len(hyp_words)
    dp = np.zeros((m + 1, n + 1), dtype=int)
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(
                    dp[i-1][j-1] + 1,  # substitution
                    dp[i-1][j] + 1,     # deletion
                    dp[i][j-1] + 1,     # insertion
                )
    
    errors = dp[m][n]
    wer = errors / m if m > 0 else 0
    
    return {
        'wer': wer,
        'errors': errors,
        'ref_length': m,
        'hyp_length': n,
    }


def compute_lattice_wer(lattice: List[Set[str]], hypothesis: str) -> Dict:
    """
    Compute WER using lattice as reference.
    A hypothesis word is correct if it matches ANY alternative in the lattice bin.
    
    Uses DP alignment where match cost is 0 if hyp word ∈ lattice bin.
    """
    hyp_words = tokenize(hypothesis)
    m = len(lattice)
    n = len(hyp_words)
    
    # DP table
    dp = np.zeros((m + 1, n + 1), dtype=float)
    
    for i in range(m + 1):
        dp[i][0] = i  # deletion cost
    for j in range(n + 1):
        dp[0][j] = j  # insertion cost
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # Check if hypothesis word matches any lattice alternative
            if hyp_words[j-1] in lattice[i-1]:
                match_cost = 0
            else:
                match_cost = 1
            
            dp[i][j] = min(
                dp[i-1][j-1] + match_cost,  # match/substitute
                dp[i-1][j] + 1,               # deletion (lattice word missing in hyp)
                dp[i][j-1] + 1,               # insertion (extra word in hyp)
            )
    
    errors = dp[m][n]
    wer = errors / m if m > 0 else 0
    
    return {
        'wer': wer,
        'errors': int(errors),
        'lattice_length': m,
        'hyp_length': n,
    }


# ============================================================================
# MAIN EVALUATION FUNCTION
# ============================================================================

def evaluate_with_lattice(
    csv_path: str,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Run lattice-based evaluation on the Question 4 CSV data.
    
    Expected CSV columns:
        segment_url_link, Human, Model H, Model i, Model k, Model l, Model m, Model n
    
    Returns:
        DataFrame with standard WER and lattice WER for each model
    """
    df = pd.read_csv(csv_path)
    
    # Identify model columns (exclude 'segment_url_link' and 'Human')
    model_columns = [col for col in df.columns if col.strip() not in ['segment_url_link', 'Human', '']]
    model_columns = [col.strip() for col in model_columns if col.strip()]
    
    logger.info(f"Found {len(model_columns)} model columns: {model_columns}")
    logger.info(f"Processing {len(df)} segments...")
    
    results = []
    
    for idx, row in df.iterrows():
        human_ref = str(row['Human'])
        
        # Get all model outputs
        model_outputs = []
        for col in model_columns:
            output = str(row[col]) if pd.notna(row[col]) else ""
            model_outputs.append(output)
        
        # Build lattice
        lattice = build_lattice(human_ref, model_outputs)
        
        # Compute WER for each model
        for col_idx, col in enumerate(model_columns):
            hyp = model_outputs[col_idx]
            
            # Standard WER (vs human reference only)
            std_wer = compute_standard_wer(human_ref, hyp)
            
            # Lattice WER (vs lattice with alternatives)
            lat_wer = compute_lattice_wer(lattice, hyp)
            
            results.append({
                'segment_idx': idx,
                'model': col,
                'standard_wer': std_wer['wer'],
                'lattice_wer': lat_wer['wer'],
                'wer_reduction': std_wer['wer'] - lat_wer['wer'],
                'ref_length': std_wer['ref_length'],
            })
    
    results_df = pd.DataFrame(results)
    
    # Aggregate by model
    summary = results_df.groupby('model').agg({
        'standard_wer': 'mean',
        'lattice_wer': 'mean',
        'wer_reduction': 'mean',
    }).round(4)
    
    summary['improvement_%'] = ((summary['standard_wer'] - summary['lattice_wer']) / 
                                 summary['standard_wer'] * 100).round(2)
    
    logger.info("\n=== WER Comparison: Standard vs Lattice ===")
    logger.info(f"\n{summary.to_string()}")
    
    if output_path:
        results_df.to_csv(output_path, index=False)
        logger.info(f"\nDetailed results saved to {output_path}")
    
    return results_df, summary


# ============================================================================
# DEMO
# ============================================================================

if __name__ == "__main__":
    # Example from the task description
    human = "उसने चौदह किताबें खरीदीं"
    model_outputs = [
        "उसने 14 किताबें खरीदीं",
        "उसने चौदह किताबे खरीदी",
        "उसने चौदह पुस्तकें खरीदीं",
    ]
    
    lattice = build_lattice(human, model_outputs)
    print("Lattice:", lattice_to_string(lattice))
    
    for i, model in enumerate(model_outputs):
        std = compute_standard_wer(human, model)
        lat = compute_lattice_wer(lattice, model)
        print(f"\nModel {i+1}: '{model}'")
        print(f"  Standard WER: {std['wer']:.3f}")
        print(f"  Lattice WER:  {lat['wer']:.3f}")
