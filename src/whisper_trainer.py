"""
Whisper Fine-Tuning Trainer Module
Complete training pipeline for fine-tuning Whisper-Small on Hindi ASR data.
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
    WhisperFeatureExtractor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# DATA COLLATOR
# ============================================================================

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Custom data collator for Whisper fine-tuning.
    Pads input features and labels to the longest in the batch.
    """
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Split inputs and labels
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        # Pad input features
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Pad labels
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 to ignore in loss
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Remove BOS token if it was appended (Whisper automatically adds it)
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


# ============================================================================
# PREPROCESSING FUNCTIONS
# ============================================================================

def get_processor(model_name: str = "openai/whisper-small", language: str = "Hindi", task: str = "transcribe"):
    """
    Load and configure Whisper processor for Hindi transcription.
    """
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
    tokenizer = WhisperTokenizer.from_pretrained(model_name, language=language, task=task)
    processor = WhisperProcessor.from_pretrained(model_name, language=language, task=task)
    
    return processor, feature_extractor, tokenizer


def prepare_dataset(batch, processor):
    """
    Preprocessing function for each batch in the HuggingFace dataset.
    - Loads audio from file path using librosa
    - Extracts log-mel features from audio
    - Tokenizes the transcription text
    """
    import librosa
    
    # Load audio from file path
    audio_array, sr = librosa.load(batch["audio_path"], sr=16000)
    
    # Extract audio features
    batch["input_features"] = processor.feature_extractor(
        audio_array,
        sampling_rate=16000
    ).input_features[0]

    # Tokenize transcription
    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
    
    return batch


# ============================================================================
# METRIC COMPUTATION
# ============================================================================

def create_compute_metrics(processor):
    """
    Create the compute_metrics function for WER evaluation.
    """
    wer_metric = evaluate.load("wer")
    
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Replace -100 with pad token id
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        # Decode predictions and references
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}
    
    return compute_metrics


# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

def get_training_args(
    output_dir: str = "./whisper-small-hi",
    num_train_epochs: int = 7,
    per_device_train_batch_size: int = 16,
    per_device_eval_batch_size: int = 8,
    learning_rate: float = 1e-5,
    warmup_steps: int = 500,
    gradient_accumulation_steps: int = 1,
    eval_steps: int = 500,
    save_steps: int = 1000,
    fp16: bool = None,
) -> Seq2SeqTrainingArguments:
    """
    Configure training arguments for Whisper fine-tuning.
    """
    # Auto-detect fp16: only enable if CUDA is available
    if fp16 is None:
        fp16 = torch.cuda.is_available()
    
    return Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        num_train_epochs=num_train_epochs,
        fp16=fp16,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=225,
        logging_steps=25,
        report_to=["none"],
        push_to_hub=False,
        remove_unused_columns=False,
        label_names=["labels"],
        dataloader_num_workers=0,
    )


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_whisper(
    train_dataset,
    val_dataset,
    model_name: str = "openai/whisper-small",
    output_dir: str = "./whisper-small-hi",
    **training_kwargs
):
    """
    Full Whisper fine-tuning pipeline.
    
    Args:
        train_dataset: HuggingFace dataset (preprocessed with prepare_dataset)
        val_dataset: HuggingFace dataset (preprocessed with prepare_dataset)
        model_name: Base model to fine-tune
        output_dir: Where to save checkpoints
        **training_kwargs: Override default training arguments
    
    Returns:
        (trainer, model, processor)
    """
    # Load model and processor
    processor, feature_extractor, tokenizer = get_processor(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    
    # Configure model for Hindi
    model.generation_config.language = "Hindi"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    
    # Data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )
    
    # Metrics
    compute_metrics = create_compute_metrics(processor)
    
    # Training arguments
    training_args = get_training_args(output_dir=output_dir, **training_kwargs)
    
    # Trainer
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
    )
    
    logger.info("Starting training...")
    trainer.train()
    
    # Save best model
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    
    logger.info(f"Model saved to {output_dir}")
    
    return trainer, model, processor


# ============================================================================
# INFERENCE / EVALUATION
# ============================================================================

def transcribe_audio(
    audio_path: str,
    model,
    processor,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    language: str = "Hindi",
) -> str:
    """
    Transcribe a single audio file using Whisper model.
    """
    import librosa
    
    # Load audio
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    
    # Process
    input_features = processor.feature_extractor(
        audio, sampling_rate=16000, return_tensors="pt"
    ).input_features.to(device)
    
    model = model.to(device)
    
    # Generate
    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            language=language,
            task="transcribe",
            max_length=225,
        )
    
    # Decode
    transcription = processor.tokenizer.batch_decode(
        predicted_ids, skip_special_tokens=True
    )[0]
    
    return transcription


def evaluate_on_fleurs(
    model,
    processor,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 8,
) -> Dict:
    """
    Evaluate model on FLEURS Hindi test set.
    
    Returns:
        Dict with WER and predictions
    """
    from datasets import load_dataset
    from jiwer import wer as compute_wer
    from src.data_utils import normalize_hindi_text
    from tqdm import tqdm
    
    # Load FLEURS Hindi test set
    logger.info("Loading FLEURS Hindi test set...")
    fleurs = load_dataset("google/fleurs", "hi_in", split="test")
    
    model = model.to(device)
    model.eval()
    
    all_predictions = []
    all_references = []
    
    for i in tqdm(range(0, len(fleurs), batch_size), desc="Evaluating on FLEURS"):
        batch = fleurs[i:i + batch_size]
        
        # Extract audio arrays from FLEURS Audio feature dicts
        audio_arrays = [a["array"] for a in batch["audio"]]
        
        # Process batch
        input_features = processor.feature_extractor(
            audio_arrays,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        ).input_features.to(device)
        
        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                language="Hindi",
                task="transcribe",
                max_length=225,
            )
        
        predictions = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
        references = batch["transcription"]
        
        # Normalize
        predictions = [normalize_hindi_text(p) for p in predictions]
        references = [normalize_hindi_text(r) for r in references]
        
        all_predictions.extend(predictions)
        all_references.extend(references)
    
    # Compute WER
    overall_wer = compute_wer(all_references, all_predictions)
    
    # Per-sentence WER for error analysis
    sentence_wers = []
    for pred, ref in zip(all_predictions, all_references):
        try:
            s_wer = compute_wer([ref], [pred])
        except:
            s_wer = 1.0
        sentence_wers.append(s_wer)
    
    logger.info(f"FLEURS Hindi WER: {overall_wer:.4f} ({overall_wer*100:.2f}%)")
    
    return {
        'overall_wer': overall_wer,
        'predictions': all_predictions,
        'references': all_references,
        'sentence_wers': sentence_wers,
    }
