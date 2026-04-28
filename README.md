<p align="center">
  <h1 align="center">🎙️ Hindi ASR: Whisper Fine-Tuning, Post-Processing & Evaluation</h1>
  <p align="center">
    <strong>End-to-end pipeline for Hindi Automatic Speech Recognition — from data preprocessing to lattice-based evaluation</strong>
  </p>
  <p align="center">
    <a href="#methodology">Methodology</a> •
    <a href="#results">Results</a> •
    <a href="#architecture">Architecture</a> •
    <a href="#getting-started">Getting Started</a> •
    <a href="#repository-structure">Repository Structure</a>
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?logo=huggingface&logoColor=black" alt="HuggingFace">
  <img src="https://img.shields.io/badge/Model-Whisper--Small-412991?logo=openai&logoColor=white" alt="Whisper">
  <img src="https://img.shields.io/badge/Language-Hindi_(hi)-FF6F00" alt="Hindi">
</p>

---

## Table of Contents

- [Project Overview](#project-overview)
- [Methodology](#methodology)
  - [Task 1 — Whisper Fine-Tuning & Error Analysis](#task-1--whisper-fine-tuning--error-analysis)
  - [Task 2 — ASR Output Cleanup Pipeline](#task-2--asr-output-cleanup-pipeline)
  - [Task 3 — Hindi Spelling Verification at Scale](#task-3--hindi-spelling-verification-at-scale)
  - [Task 4 — Lattice-Based ASR Evaluation](#task-4--lattice-based-asr-evaluation)
- [Results](#results)
- [Architecture](#architecture)
- [Getting Started](#getting-started)
- [Repository Structure](#repository-structure)
- [Technical Decisions & Trade-offs](#technical-decisions--trade-offs)
- [Limitations & Future Work](#limitations--future-work)
- [License](#license)

---

## Project Overview

This repository implements a comprehensive Hindi ASR research pipeline spanning **four interconnected tasks**:

1. **Whisper Fine-Tuning** — Fine-tuning OpenAI's Whisper-Small on ~10 hours of conversational Hindi audio, achieving a **44.6% relative WER reduction** on the FLEURS Hindi benchmark.
2. **ASR Post-Processing** — A cleanup pipeline featuring Hindi number normalization (with idiom preservation) and English word detection in Devanagari script.
3. **Spelling Verification** — A 12-layer rule-based classifier that categorizes **177,508 unique Hindi words** as correctly or incorrectly spelled, combining frequency analysis, morphological decomposition, and transliteration detection.
4. **Lattice-Based Evaluation** — A novel evaluation framework that replaces rigid single-reference WER with a lattice of valid transcription alternatives, reducing unfair penalties by up to **74.6%**.

> **Context:** Built as a research assignment for the AI Researcher Intern (Speech & Audio) role at Josh Talks. The dataset comprises real-world Hindi conversational recordings collected via the Josh Talks platform.

---

## Methodology

### Task 1 — Whisper Fine-Tuning & Error Analysis

#### 1.1 Data Preprocessing

| Stage | Description | Output |
|-------|-------------|--------|
| **URL Resolution** | Remapped legacy GCS paths (`joshtalks-data-collection/hq_data/hi`) to the active `upload_goai` bucket | 104 accessible recordings |
| **Download & Validation** | Fetched audio (`.wav`), transcription (`.json`), and metadata (`.json`) with retry logic | Raw data cached locally |
| **Audio Segmentation** | Split recordings into utterance-level clips using sentence-level timestamps from transcription JSONs | **4,929 segments** (1–30s each) |
| **Text Normalization** | Unicode NFC normalization, punctuation removal, whitespace normalization, lowercasing of Latin characters | Clean reference transcripts |
| **Train/Val Split** | 90/10 split stratified by `recording_id` to prevent data leakage | 4,436 train / 493 val |

#### 1.2 Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Base Model | `openai/whisper-small` | Task requirement; 244M parameters |
| Learning Rate | 1×10⁻⁵ | Conservative rate to avoid catastrophic forgetting |
| Effective Batch Size | 16 | `per_device=4 × gradient_accumulation=4` |
| Epochs | 7 (with early stopping) | Patience = 3 evaluations |
| Encoder Freezing | Epoch 1 only | Custom `EncoderUnfreezeCallback` — freezes encoder initially (36.5% params trainable), unfreezes after epoch 1 for full adaptation |
| Precision | Auto-detected | bf16 + tf32 on Ampere+ GPUs; fp16 on older hardware |
| Gradient Checkpointing | Enabled | ~50% VRAM reduction at ~20% compute overhead |

#### 1.3 Error Analysis Methodology

- **Systematic Sampling:** Stratified sampling across 5 WER severity bands (Very High >0.8 to Very Low ≤0.1), 5 samples per stratum, totaling **25 utterances** (`random_state=42`)
- **Taxonomy Construction:** Six error categories emerged organically from the data:

| Category | Frequency | Description |
|----------|-----------|-------------|
| Phonetic Substitution | 32% | Near-homophone confusion (e.g., `क्ष` → `श`) |
| Repetition/Hallucination | 20% | Attention collapse producing infinite token loops |
| Word Boundary Errors | 20% | Incorrect merging or splitting of words |
| Rare/Proper Noun Errors | 16% | OOV names and foreign words garbled |
| Inflection/Grammar | 12% | Correct root, wrong conjugation or gender |
| Number/Date Errors | 8% | Format mismatches between digits and Hindi words |

---

### Task 2 — ASR Output Cleanup Pipeline

#### 2a. Hindi Number Normalization

**Module:** [`src/number_normalizer.py`](src/number_normalizer.py)

- Complete vocabulary of all 100 unique Hindi number words (0–99), plus powers (सौ, हज़ार, लाख, करोड़)
- **Compound number parsing:** Greedy left-to-right accumulation algorithm handles expressions like `तीन सौ चौवन → 354`
- **Idiom preservation:** 14+ regex patterns protect idiomatic expressions from conversion (e.g., `दो-चार बातें`, `नौ दो ग्यारह`)

#### 2b. English Word Detection in Devanagari

**Module:** [`src/english_detector.py`](src/english_detector.py)

A three-signal hybrid detector:

| Signal | Method | Confidence |
|--------|--------|------------|
| Latin Script | Direct character detection | High |
| Lookup Table | ~150 curated English→Devanagari mappings | High |
| Suffix Patterns | Common English suffixes in Devanagari (`-शन`, `-मेंट`, `-नेस`, `-इंग`) | Medium |

Output format: `मेरा [EN]इंटरव्यू[/EN] बहुत अच्छा गया और मुझे [EN]जॉब[/EN] मिल गई`

---

### Task 3 — Hindi Spelling Verification at Scale

**Module:** [`src/spelling_checker.py`](src/spelling_checker.py)

Processed **177,508 unique words** through a 12-layer verification pipeline:

```
Input Word
    │
    ├─ Layer 1:  Dictionary lookup (curated ~1,200 + top-5,000 frequency words)
    ├─ Layer 2:  English transliteration lookup (~150 known mappings)
    ├─ Layer 3:  Devanagari structure validation (reject invalid sequences)
    ├─ Layer 4:  Mixed content detection (numbers/punctuation attached)
    ├─ Layer 5:  Mixed script detection (Latin characters in Devanagari)
    ├─ Layer 6:  Morphological decomposition (suffix → prefix → combined)
    ├─ Layer 7:  English suffix pattern detection
    ├─ Layer 8:  Number word detection (from normalizer vocabulary)
    ├─ Layer 9:  Nukta variant tolerance (ज ↔ ज़, फ ↔ फ़)
    ├─ Layer 10: Compound word analysis (split & validate parts)
    ├─ Layer 11: Short word tolerance (2-3 char pure Devanagari)
    └─ Layer 12: Default classification (unknown → incorrect, low confidence)
```

**Key Innovation — Frequency-Based Self-Derived Dictionary:** Leveraging Zipf's law, the top 5,000 most frequent words from the input corpus are treated as a reliable dictionary, providing ~4,500 additional validated entries beyond the curated vocabulary.

#### Manual Validation

50 low-confidence words were manually reviewed:

| System Verdict | Manual: Correct | Manual: Incorrect |
|----------------|:---------------:|:-----------------:|
| Correct (6)    | 6 ✓             | 0                 |
| Incorrect (44) | 18 ✗            | 26 ✓              |

**System accuracy on low-confidence sample: 64%** — Primary blind spots: proper nouns, unlisted English transliterations, complex verb morphology, and unsplit compound words.

---

### Task 4 — Lattice-Based ASR Evaluation

**Module:** [`src/lattice_builder.py`](src/lattice_builder.py)

#### The Problem

Standard WER uses a single rigid reference, unfairly penalizing models for valid alternative transcriptions (spelling variants, number formats, compound word forms).

#### The Solution — Lattice Construction

A **lattice** replaces the flat reference with a sequence of **bins**, each containing all valid alternatives for that position:

```
Flat:    [उसने] [चौदह]        [किताबें]                       [खरीदीं]
Lattice: [उसने] [चौदह | 14]   [किताबें | किताबे | पुस्तकें]   [खरीदीं | खरीदी]
```

**Algorithm:**

1. **Tokenize** all inputs (human reference + N model outputs)
2. **Progressive MSA** — align each model to a growing master alignment using Needleman-Wunsch (character-level similarity scoring)
3. **Trust mechanism** — words agreed upon by ≥3 models are added as valid lattice alternatives
4. **Lattice WER** — DP alignment where `match_cost = 0` if hypothesis word ∈ lattice bin

**Alignment unit:** Word-level — Hindi is space-delimited, making words the natural unit; subword would fragment morphology, phrase-level is too coarse.

---

## Results

### Task 1 — WER on FLEURS Hindi Test Set (844 utterances)

| Model | WER | Δ |
|-------|:---:|:---:|
| Whisper-Small (Pretrained Baseline) | 68.20% | — |
| **Whisper-Small (Fine-Tuned)** | **37.77%** | **−30.43 pp** |

> **Relative WER Reduction: 44.6%** — The fine-tuned model nearly halves the error rate despite being trained exclusively on conversational Hindi and evaluated on formal Hindi (FLEURS).

### Task 3 — Spelling Classification (177,508 words)

| Classification | Count | % |
|---------------|------:|-----:|
| Correctly Spelled | 49,120 | 27.7% |
| Incorrectly Spelled | 128,388 | 72.3% |

| Confidence | Count | % |
|-----------|------:|-----:|
| High | 16,806 | 9.5% |
| Medium | 23,112 | 13.0% |
| Low | 137,590 | 77.5% |

### Task 4 — Standard WER vs. Lattice WER

| Model | Standard WER | Lattice WER | Relative Improvement |
|-------|:-----------:|:-----------:|:--------------------:|
| Model H | 3.31% | 3.01% | 9.1% |
| Model i | 0.61% | 3.26% | −434.4%* |
| Model k | 10.18% | 3.02% | **70.3%** |
| Model l | 10.66% | 3.60% | **66.2%** |
| **Model m** | **19.56%** | **4.96%** | **74.6%** |
| Model n | 10.32% | 3.11% | **69.9%** |

> \*Model i anomaly: Its near-perfect standard WER (0.61%) was inflated by progressive MSA bin-boundary misalignment — a known limitation for nearly-perfect transcriptions.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                               │
│  FT Data CSV ─→ data_utils.py ─→ Download ─→ Segment ─→ Split  │
│  (104 recordings)   (URL fix)     (audio+JSON)  (4,929 clips)   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
┌──────────────┐  ┌───────────────┐  ┌──────────────────┐
│   TASK 1     │  │    TASK 2     │  │     TASK 3       │
│ whisper_     │  │ number_       │  │ spelling_        │
│ trainer.py   │  │ normalizer.py │  │ checker.py       │
│              │  │ english_      │  │ (12-layer        │
│ • Fine-tune  │  │ detector.py   │  │  pipeline)       │
│ • Evaluate   │  │              │  │                  │
│ • Error      │  │ • Normalize  │  │ • 177K words     │
│   analysis   │  │ • Tag [EN]   │  │ • Confidence     │
└──────┬───────┘  └──────┬────────┘  └────────┬─────────┘
       │                 │                    │
       ▼                 ▼                    ▼
┌──────────────────────────────────────────────────────┐
│                    TASK 4                              │
│  lattice_builder.py                                   │
│  • Needleman-Wunsch alignment                         │
│  • Progressive MSA                                    │
│  • Lattice construction & WER computation             │
└──────────────────────────────────────────────────────┘
```

---

## Getting Started

### Prerequisites

- **Python** ≥ 3.9
- **CUDA GPU** (recommended for Task 1; Tasks 2–4 run on CPU)
- **~15 GB disk space** (dataset downloads, model checkpoints)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/whisper-hindi-finetuning.git
cd whisper-hindi-finetuning

# Create virtual environment
python -m venv venv
source venv/bin/activate       # Linux/macOS
venv\Scripts\activate          # Windows

# Install dependencies
pip install -r requirements.txt
```

### Running the Tasks

```bash
# Task 1: Whisper Fine-Tuning (requires GPU, ~2-4 hours on A100)
jupyter notebook notebooks/Q1_whisper_finetuning.py

# Task 2: ASR Cleanup Pipeline (requires pretrained Whisper for ASR generation)
jupyter notebook notebooks/Q2_cleanup_pipeline.py

# Task 3: Spelling Verification (CPU sufficient, ~10 minutes)
python run_q3.py

# Task 4: Lattice-Based Evaluation (CPU sufficient, ~1 minute)
jupyter notebook notebooks/Q4_lattice_evaluation.py
```

### Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥ 2.0.0 | Deep learning framework |
| `transformers` | ≥ 4.36.0 | Whisper model & training |
| `datasets` | ≥ 2.16.0 | HuggingFace dataset management |
| `librosa` | ≥ 0.10.0 | Audio loading & processing |
| `jiwer` | ≥ 3.0.0 | WER computation |
| `indic-nlp-library` | ≥ 0.92 | Hindi NLP utilities |
| `pandas` | ≥ 2.0.0 | Data manipulation |

---

## Repository Structure

```
.
├── src/                              # Core Python modules
│   ├── data_utils.py                 # URL fixing, downloading, segmentation, normalization
│   ├── whisper_trainer.py            # Fine-tuning pipeline with encoder freezing & evaluation
│   ├── number_normalizer.py          # Hindi number word → digit conversion + idiom preservation
│   ├── english_detector.py           # English word detection in Devanagari text (3-signal)
│   ├── spelling_checker.py           # 12-layer Hindi spelling verification system
│   └── lattice_builder.py            # Lattice construction + lattice-based WER computation
│
├── notebooks/                        # Executable notebooks (with outputs)
│   ├── Q1_whisper_finetuning.py      # Full Task 1 pipeline
│   ├── Q2_cleanup_pipeline.py        # Number normalization + English detection
│   ├── Q3_spelling_verification.py   # 177K word classification
│   └── Q4_lattice_evaluation.py      # Lattice-based WER evaluation
│
├── data/
│   ├── input/                        # Input datasets
│   │   ├── ft_data.csv               # 104 recording metadata (Task 1)
│   │   ├── unique_words.csv          # 177K unique words (Task 3)
│   │   └── q4_task.csv               # Multi-model transcriptions (Task 4)
│   ├── raw/                          # Downloaded audio, transcriptions, metadata
│   └── processed/                    # Segmented audio clips
│
├── results/                          # All output files
│   ├── wer_results.csv               # Task 1: WER comparison table
│   ├── ft_result.csv                 # Task 1: WER in provided template format
│   ├── error_analysis.csv            # Task 1: 25 stratified error samples
│   ├── error_taxonomy.md             # Task 1: Full error taxonomy + fix proposals
│   ├── q2_cleanup_results.csv        # Task 2: Cleanup pipeline outputs
│   ├── q3_spelling_results.csv       # Task 3: 177K words with classifications (gitignored)
│   ├── q3_low_confidence_review.csv  # Task 3: 50 manually reviewed low-confidence words
│   ├── q4_lattice_wer_results.csv    # Task 4: Per-segment lattice WER
│   └── q4_wer_summary.csv           # Task 4: Aggregated WER comparison
│
├── docs/
│   └── final_submission.md           # Detailed submission document
│
├── models/whisper-small-hi/          # Fine-tuned model checkpoint (gitignored)
├── run_q3.py                         # Standalone runner for Task 3
├── requirements.txt                  # Python dependencies
└── .gitignore
```

---

## Technical Decisions & Trade-offs

| Decision | Rationale |
|----------|-----------|
| **Encoder freezing (Epoch 1)** | Prevents catastrophic forgetting of Whisper's pretrained audio features while the decoder adapts to Hindi-specific patterns. Reduces trainable parameters by ~60% initially. |
| **Recording-level train/val split** | Splitting by individual segments would leak speaker characteristics across sets, inflating validation scores. |
| **Frequency-based self-derived dictionary (Task 3)** | Avoids dependency on external Hindi dictionaries that may not cover conversational/code-switched vocabulary. Zipf's law guarantees high-frequency words are overwhelmingly correctly spelled. |
| **Word-level lattice alignment (Task 4)** | Hindi is predominantly space-delimited. Subword tokenization would fragment morphological structure; phrase-level would be too coarse to capture word-level variation. |
| **Rule-based spelling checker over ML** | For 177K isolated words without sentence context, rule-based approaches offer interpretability and confidence scoring. An ML approach would require a labeled training set that doesn't exist for this domain. |

---

## Limitations & Future Work

- **Task 1 (Fix Implementation):** The repetition suppression fix (`no_repeat_ngram_size=3`, `repetition_penalty=1.2`) was designed but not evaluated due to time constraints. Infrastructure for before/after comparison exists in `whisper_trainer.py`.
- **Task 2 (English Detection):** Detection rate on raw ASR output is low (2/100 transcripts) because Whisper outputs primarily in Devanagari. The detector is more valuable on human-written transcriptions with explicit code-switching.
- **Task 3 (Proper Nouns):** The system lacks a named entity dictionary, causing all proper nouns to be classified as misspelled. Integration with a Hindi NER model would address this.
- **Task 4 (Model i Anomaly):** Progressive MSA can introduce bin-boundary misalignment for near-perfect transcriptions. A center-star MSA or iterative refinement approach could mitigate this.

---

## License

This project was developed as a research assignment. All code is original work. The dataset is proprietary to Josh Talks and is not redistributed.

---

<p align="center">
  <sub>Built with 🔬 by Rajeev </sub>
</p>
