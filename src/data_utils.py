"""
Data Utilities Module
Handles URL fixing, audio downloading, and preprocessing for the dataset.
"""

import os
import re
import json
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# URL FIXING
# ============================================================================

# Old format: https://storage.googleapis.com/joshtalks-data-collection/hq_data/hi/{folder_id}/{recording_id}_audio.wav
# New format: https://storage.googleapis.com/upload_goai/{folder_id}/{recording_id}_audio.wav

OLD_BASE = "https://storage.googleapis.com/joshtalks-data-collection/hq_data/hi"
NEW_BASE = "https://storage.googleapis.com/upload_goai"


def fix_gcs_url(old_url: str) -> str:
    """
    Convert old GCS URL format to new upload_goai format.
    
    Old: https://storage.googleapis.com/joshtalks-data-collection/hq_data/hi/{folder}/{file}
    New: https://storage.googleapis.com/upload_goai/{folder}/{file}
    """
    if OLD_BASE in old_url:
        return old_url.replace(OLD_BASE, NEW_BASE)
    return old_url


def fix_all_urls_in_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Fix all GCS URLs in the dataset DataFrame."""
    df = df.copy()
    url_columns = ['rec_url_gcp', 'transcription_url_gcp', 'metadata_url_gcp']
    for col in url_columns:
        if col in df.columns:
            df[col] = df[col].apply(fix_gcs_url)
    return df


# ============================================================================
# DATA DOWNLOADING
# ============================================================================

def download_file(url: str, output_path: str, timeout: int = 60) -> bool:
    """
    Download a file from URL to local path.
    Returns True if successful, False otherwise.
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if os.path.exists(output_path):
            logger.debug(f"File already exists: {output_path}")
            return True
        
        response = requests.get(url, timeout=timeout, stream=True)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.debug(f"Downloaded: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False


def download_dataset(
    csv_path: str,
    output_dir: str,
    download_audio: bool = True,
    download_transcription: bool = True,
    download_metadata: bool = True
) -> pd.DataFrame:
    """
    Download all files from the dataset CSV.
    
    Args:
        csv_path: Path to the FT Data CSV file
        output_dir: Directory to save downloaded files
        download_audio: Whether to download audio files
        download_transcription: Whether to download transcription JSONs
        download_metadata: Whether to download metadata JSONs
    
    Returns:
        DataFrame with local file paths added
    """
    df = pd.read_csv(csv_path)
    df = fix_all_urls_in_dataframe(df)
    
    audio_dir = os.path.join(output_dir, "audio")
    transcription_dir = os.path.join(output_dir, "transcriptions")
    metadata_dir = os.path.join(output_dir, "metadata")
    
    local_audio_paths = []
    local_transcription_paths = []
    local_metadata_paths = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Downloading dataset"):
        recording_id = row['recording_id']
        
        # Audio
        if download_audio:
            audio_path = os.path.join(audio_dir, f"{recording_id}_audio.wav")
            download_file(row['rec_url_gcp'], audio_path)
            local_audio_paths.append(audio_path)
        else:
            local_audio_paths.append(None)
        
        # Transcription
        if download_transcription:
            trans_path = os.path.join(transcription_dir, f"{recording_id}_transcription.json")
            download_file(row['transcription_url_gcp'], trans_path)
            local_transcription_paths.append(trans_path)
        else:
            local_transcription_paths.append(None)
        
        # Metadata
        if download_metadata:
            meta_path = os.path.join(metadata_dir, f"{recording_id}_metadata.json")
            download_file(row['metadata_url_gcp'], meta_path)
            local_metadata_paths.append(meta_path)
        else:
            local_metadata_paths.append(None)
    
    df['local_audio_path'] = local_audio_paths
    df['local_transcription_path'] = local_transcription_paths
    df['local_metadata_path'] = local_metadata_paths
    
    return df


# ============================================================================
# TRANSCRIPTION PARSING
# ============================================================================

def parse_transcription_json(json_path: str) -> List[Dict]:
    """
    Parse a transcription JSON file.
    Returns list of segments with start_time, end_time, and text.
    
    Expected JSON structure (may vary):
    - Could be a list of segments with timestamps and text
    - Or a dict with a 'segments' key
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        segments = []
        
        # Handle different JSON structures
        if isinstance(data, list):
            # List of segments
            for item in data:
                seg = _extract_segment(item)
                if seg:
                    segments.append(seg)
        elif isinstance(data, dict):
            # Dict with segments key
            if 'segments' in data:
                for item in data['segments']:
                    seg = _extract_segment(item)
                    if seg:
                        segments.append(seg)
            elif 'text' in data:
                # Single transcription
                segments.append({
                    'start_time': data.get('start', data.get('start_time', 0)),
                    'end_time': data.get('end', data.get('end_time', 0)),
                    'text': data['text']
                })
            else:
                # Try to extract from arbitrary dict structure
                seg = _extract_segment(data)
                if seg:
                    segments.append(seg)
        
        return segments
    except Exception as e:
        logger.error(f"Failed to parse {json_path}: {e}")
        return []


def _extract_segment(item: dict) -> Optional[Dict]:
    """Extract start_time, end_time, text from a segment dict."""
    text = item.get('text', item.get('transcript', item.get('sentence', '')))
    if not text:
        return None
    
    start = item.get('start', item.get('start_time', item.get('begin', 0)))
    end = item.get('end', item.get('end_time', item.get('finish', 0)))
    
    return {
        'start_time': float(start),
        'end_time': float(end),
        'text': str(text).strip()
    }


# ============================================================================
# AUDIO SEGMENTATION
# ============================================================================

def segment_audio(
    audio_path: str,
    segments: List[Dict],
    output_dir: str,
    sample_rate: int = 16000,
    min_duration: float = 1.0,
    max_duration: float = 30.0
) -> List[Dict]:
    """
    Split a full audio file into segments based on transcription timestamps.
    
    Args:
        audio_path: Path to full audio WAV file
        segments: List of dicts with start_time, end_time, text
        output_dir: Directory to save segment WAV files
        sample_rate: Target sample rate (16kHz for Whisper)
        min_duration: Minimum segment duration in seconds
        max_duration: Maximum segment duration in seconds
    
    Returns:
        List of valid segments with local audio paths
    """
    import librosa
    import soundfile as sf
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load full audio
    audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    
    valid_segments = []
    recording_id = Path(audio_path).stem.replace('_audio', '')
    
    for i, seg in enumerate(segments):
        start_time = seg['start_time']
        end_time = seg['end_time']
        duration = end_time - start_time
        
        # Filter by duration
        if duration < min_duration or duration > max_duration:
            continue
        
        # Skip empty transcriptions
        if not seg['text'].strip():
            continue
        
        # Extract segment audio
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        
        # Bounds checking
        start_sample = max(0, start_sample)
        end_sample = min(len(audio), end_sample)
        
        if start_sample >= end_sample:
            continue
        
        segment_audio = audio[start_sample:end_sample]
        
        # Save segment
        seg_filename = f"{recording_id}_seg{i:04d}.wav"
        seg_path = os.path.join(output_dir, seg_filename)
        sf.write(seg_path, segment_audio, sample_rate)
        
        valid_segments.append({
            'audio_path': seg_path,
            'text': seg['text'].strip(),
            'duration': duration,
            'recording_id': recording_id,
            'segment_id': i,
            'start_time': start_time,
            'end_time': end_time
        })
    
    return valid_segments


# ============================================================================
# DATA FILTERING (Optimization)
# ============================================================================

def filter_bad_samples(
    segments: List[Dict],
    min_duration: float = 1.0,
    max_duration: float = 25.0,
    min_text_length: int = 2,
    remove_outliers: bool = True,
    outlier_sigma: float = 3.0,
) -> List[Dict]:
    """
    Remove segments that would harm training efficiency or quality.
    
    Filters:
      - Too short (<1s): insufficient audio context, mostly noise
      - Too long (>25s): causes OOM spikes and excessive padding
      - Empty/near-empty transcriptions: no learning signal
      - Statistical duration outliers (>3σ from mean)
    
    This typically removes ~5-15% of samples while improving convergence.
    
    Args:
        segments: List of segment dicts with 'duration', 'text' keys
        min_duration: Minimum segment duration in seconds
        max_duration: Maximum segment duration in seconds
        min_text_length: Minimum transcription character count
        remove_outliers: Whether to remove statistical outliers
        outlier_sigma: Number of std devs for outlier detection
    
    Returns:
        Filtered list of segments
    """
    original_count = len(segments)
    
    # Basic filters
    filtered = [
        s for s in segments
        if s.get('duration', 0) >= min_duration
        and s.get('duration', 0) <= max_duration
        and len(s.get('text', '').strip()) >= min_text_length
    ]
    
    # Statistical outlier removal
    if remove_outliers and filtered:
        durations = np.array([s['duration'] for s in filtered])
        mean_dur = np.mean(durations)
        std_dur = np.std(durations)
        lower = max(min_duration, mean_dur - outlier_sigma * std_dur)
        upper = min(max_duration, mean_dur + outlier_sigma * std_dur)
        filtered = [s for s in filtered if lower <= s['duration'] <= upper]
    
    removed = original_count - len(filtered)
    logger.info(
        f"🧹 Data filtering: {original_count} → {len(filtered)} segments "
        f"(removed {removed}, {100*removed/max(original_count,1):.1f}%)"
    )
    
    return filtered


# ============================================================================
# CACHED DATASET PREPROCESSING (Optimization)
# ============================================================================

def prepare_dataset_cached(dataset, processor, num_proc: int = 1, cache_dir: str = None):
    """
    Apply prepare_dataset with HuggingFace Arrow caching.
    
    Features are computed once and cached to disk. On subsequent runs,
    cached features are loaded in seconds instead of re-running librosa
    on every audio file.
    
    Args:
        dataset: HuggingFace Dataset (with 'audio_path' and 'sentence' columns)
        processor: WhisperProcessor instance
        num_proc: Number of parallel processes for .map() (1 for Windows)
        cache_dir: Optional cache directory override
    
    Returns:
        Preprocessed dataset with 'input_features' and 'labels' columns
    """
    from src.whisper_trainer import prepare_dataset
    from functools import partial
    
    map_fn = partial(prepare_dataset, processor=processor)
    
    logger.info(f"🔄 Preprocessing dataset ({len(dataset)} samples) with caching enabled...")
    
    processed = dataset.map(
        map_fn,
        remove_columns=dataset.column_names,
        num_proc=num_proc,
        desc="Extracting features",
        keep_in_memory=False,  # Save to disk cache
    )
    
    logger.info(f"✅ Preprocessing complete. Cached to: {processed.cache_files}")
    
    return processed


def add_input_length_column(dataset):
    """
    Add 'input_length' column for group_by_length batching.
    
    When used with `group_by_length=True` in training args, samples of
    similar length are batched together, reducing padding waste by 20-40%.
    
    Args:
        dataset: Preprocessed HuggingFace Dataset with 'input_features'
    
    Returns:
        Dataset with added 'input_length' column
    """
    def _get_length(example):
        example["input_length"] = len(example["input_features"][0]) if isinstance(
            example["input_features"], list
        ) else example["input_features"].shape[-1]
        return example
    
    dataset = dataset.map(_get_length, desc="Computing input lengths")
    logger.info("✅ Added input_length column for length-bucketed batching")
    return dataset


# ============================================================================
# DATASET BUILDING (HuggingFace format)
# ============================================================================

def build_hf_dataset(
    segments: List[Dict],
    train_ratio: float = 0.9,
    seed: int = 42
) -> Tuple:
    """
    Build HuggingFace Dataset from segments.
    Splits by recording_id to prevent data leakage.
    Includes duration for length-bucketed batching.
    
    Returns:
        (train_dataset, val_dataset)
    """
    from datasets import Dataset
    
    df = pd.DataFrame(segments)
    
    # Split by recording_id (not by individual segments)
    recording_ids = df['recording_id'].unique()
    np.random.seed(seed)
    np.random.shuffle(recording_ids)
    
    split_idx = int(len(recording_ids) * train_ratio)
    train_ids = set(recording_ids[:split_idx])
    val_ids = set(recording_ids[split_idx:])
    
    train_df = df[df['recording_id'].isin(train_ids)].reset_index(drop=True)
    val_df = df[df['recording_id'].isin(val_ids)].reset_index(drop=True)
    
    logger.info(f"Train: {len(train_df)} segments from {len(train_ids)} recordings")
    logger.info(f"Val: {len(val_df)} segments from {len(val_ids)} recordings")
    
    # Create HuggingFace datasets — include duration for length bucketing
    dataset_dict = {
        'audio_path': 'audio_path',
        'sentence': 'text',
        'recording_id': 'recording_id',
    }
    
    def _build_dataset(sub_df):
        d = {
            'audio_path': sub_df['audio_path'].tolist(),
            'sentence': sub_df['text'].tolist(),
            'recording_id': sub_df['recording_id'].tolist(),
        }
        if 'duration' in sub_df.columns:
            d['duration'] = sub_df['duration'].tolist()
        return Dataset.from_dict(d)
    
    train_dataset = _build_dataset(train_df)
    val_dataset = _build_dataset(val_df)
    
    return train_dataset, val_dataset


# ============================================================================
# TEXT NORMALIZATION (Hindi)
# ============================================================================

def normalize_hindi_text(text: str) -> str:
    """
    Normalize Hindi text for WER computation.
    - Remove punctuation
    - Normalize Unicode forms
    - Normalize whitespace
    - Lowercase (for any English mixed in)
    """
    import unicodedata
    
    # Unicode NFC normalization
    text = unicodedata.normalize('NFC', text)
    
    # Remove common punctuation (keep Devanagari characters, digits, spaces)
    text = re.sub(r'[।,?!;:\"\'""''…\-–—\(\)\[\]\{\}]', ' ', text)
    
    # Remove Devanagari danda and double danda
    text = text.replace('।', ' ').replace('॥', ' ')
    
    # Lowercase any English characters
    text = text.lower()
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


# ============================================================================
# QUICK STATS
# ============================================================================

def dataset_stats(csv_path: str) -> Dict:
    """Print quick stats about the dataset."""
    df = pd.read_csv(csv_path)
    
    total_duration_sec = df['duration'].sum()
    total_duration_hrs = total_duration_sec / 3600
    
    stats = {
        'num_recordings': len(df),
        'num_unique_users': df['user_id'].nunique(),
        'total_duration_seconds': total_duration_sec,
        'total_duration_hours': round(total_duration_hrs, 2),
        'avg_duration_seconds': round(df['duration'].mean(), 1),
        'min_duration_seconds': df['duration'].min(),
        'max_duration_seconds': df['duration'].max(),
    }
    
    logger.info(f"Dataset Stats:")
    for k, v in stats.items():
        logger.info(f"  {k}: {v}")
    
    return stats


if __name__ == "__main__":
    # Quick test
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'input', 'ft_data.csv')
    if os.path.exists(csv_path):
        stats = dataset_stats(csv_path)
        
        # Test URL fixing
        df = pd.read_csv(csv_path)
        sample_url = df.iloc[0]['rec_url_gcp']
        fixed_url = fix_gcs_url(sample_url)
        print(f"\nOriginal URL: {sample_url}")
        print(f"Fixed URL:    {fixed_url}")
