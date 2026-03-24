"""
Whisper Fine-Tuning Trainer Module (Optimized)
Complete training pipeline for fine-tuning Whisper-Small on Hindi ASR data.

Optimizations applied:
  - Gradient checkpointing (50% less VRAM)
  - bf16/tf32 auto-detection for Ampere+ GPUs
  - torch.compile for graph-mode speedup
  - Gradient accumulation (effective batch=32)
  - Encoder freezing callback (freeze epoch 1, unfreeze after)
  - Early stopping (patience=3)
  - Smarter eval/save frequency
  - Parallel data loading
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
    EarlyStoppingCallback,
    TrainerCallback,
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
# ENCODER FREEZING CALLBACK
# ============================================================================

class EncoderUnfreezeCallback(TrainerCallback):
    """
    Freezes the encoder at training start, unfreezes after `unfreeze_after_epoch`.
    
    Rationale: The pretrained encoder already excels at audio feature extraction.
    Freezing it initially:
      - Cuts trainable params by ~60% for the first epoch → much faster
      - Prevents catastrophic forgetting of pretrained representations
      - Lets the decoder (Hindi-specific) warm up first
    After unfreezing, end-to-end fine-tuning continues for full performance.
    """
    def __init__(self, unfreeze_after_epoch: int = 1):
        self.unfreeze_after_epoch = unfreeze_after_epoch
        self._frozen = False

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if model is not None:
            for param in model.model.encoder.parameters():
                param.requires_grad = False
            self._frozen = True
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            logger.info(
                f"🧊 Encoder FROZEN. Trainable: {trainable:,} / {total:,} params "
                f"({100*trainable/total:.1f}%). Will unfreeze after epoch {self.unfreeze_after_epoch}."
            )

    def on_epoch_begin(self, args, state, control, model=None, **kwargs):
        current_epoch = int(state.epoch) if state.epoch else 0
        if self._frozen and current_epoch >= self.unfreeze_after_epoch and model is not None:
            for param in model.model.encoder.parameters():
                param.requires_grad = True
            self._frozen = False
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            logger.info(
                f"🔥 Encoder UNFROZEN at epoch {current_epoch}. "
                f"Trainable: {trainable:,} / {total:,} params ({100*trainable/total:.1f}%)."
            )


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
# HARDWARE DETECTION HELPERS
# ============================================================================

def _detect_precision():
    """
    Auto-detect the best precision mode for the current hardware.
    Returns (fp16, bf16, tf32) booleans.
    
    - Ampere+ (compute capability >= 8.0): bf16=True, tf32=True
    - Older CUDA GPUs: fp16=True
    - CPU: all False
    """
    if not torch.cuda.is_available():
        return False, False, False
    
    capability = torch.cuda.get_device_capability()
    is_ampere_or_newer = capability[0] >= 8  # A100, A10, RTX 30xx, RTX 40xx, T4 is 7.5
    
    if is_ampere_or_newer:
        logger.info(f"⚡ Ampere+ GPU detected (compute {capability[0]}.{capability[1]}). Using bf16 + tf32.")
        return False, True, True  # fp16=False, bf16=True, tf32=True
    else:
        logger.info(f"🔋 Pre-Ampere GPU detected (compute {capability[0]}.{capability[1]}). Using fp16.")
        return True, False, False  # fp16=True, bf16=False, tf32=False


# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

import os

def get_training_args(
    output_dir: str = "./whisper-small-hi",
    num_train_epochs: int = 4,
    per_device_train_batch_size: int = 16,
    per_device_eval_batch_size: int = 8,
    learning_rate: float = 1e-5,
    warmup_steps: int = 200,
    gradient_accumulation_steps: int = 2,
    eval_steps: int = 1000,
    save_steps: int = 1000,
    fp16: bool = None,
    bf16: bool = None,
    tf32: bool = None,
    dataloader_num_workers: int = 0 if os.name == 'nt' else 4,
    torch_compile: bool = False,
) -> Seq2SeqTrainingArguments:
    """
    Configure optimized training arguments for Whisper fine-tuning.
    
    Key optimizations vs. original:
      - Epochs: 7 → 4 (with early stopping)
      - Gradient accumulation: 1 → 2 (effective batch = 32)
      - bf16/tf32 auto-detected for Ampere+ GPUs
      - Parallel data loading (num_workers=4)
      - Less frequent eval (500 → 1000 steps)
      - Warmup reduced (500 → 200 steps)
      - torch.compile support
    """
    # Auto-detect precision if not explicitly set
    if fp16 is None and bf16 is None:
        fp16, bf16, tf32_auto = _detect_precision()
        if tf32 is None:
            tf32 = tf32_auto
    else:
        fp16 = fp16 or False
        bf16 = bf16 or False
        if tf32 is None:
            tf32 = False
    
    # Enable tf32 globally if detected
    if tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Auto-detect torch.compile support
    if torch_compile and not hasattr(torch, 'compile'):
        logger.warning("torch.compile not available (requires PyTorch 2.0+). Disabling.")
        torch_compile = False
    
    args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        num_train_epochs=num_train_epochs,
        fp16=fp16,
        bf16=bf16,
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
        dataloader_num_workers=dataloader_num_workers,
        dataloader_pin_memory=True,
        torch_compile=torch_compile,
        optim="adamw_torch",
    )
    
    logger.info(
        f"📋 Training config: epochs={num_train_epochs}, "
        f"effective_batch={per_device_train_batch_size * gradient_accumulation_steps}, "
        f"fp16={fp16}, bf16={bf16}, tf32={tf32}, "
        f"compile={torch_compile}, workers={dataloader_num_workers}"
    )
    
    return args


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_whisper(
    train_dataset,
    val_dataset,
    model_name: str = "openai/whisper-small",
    output_dir: str = "./whisper-small-hi",
    freeze_encoder_epochs: int = 1,
    early_stopping_patience: int = 3,
    **training_kwargs
):
    """
    Optimized Whisper fine-tuning pipeline.
    
    Args:
        train_dataset: HuggingFace dataset (preprocessed with prepare_dataset)
        val_dataset: HuggingFace dataset (preprocessed with prepare_dataset)
        model_name: Base model to fine-tune
        output_dir: Where to save checkpoints
        freeze_encoder_epochs: Freeze encoder for this many epochs (0 to disable)
        early_stopping_patience: Stop if WER doesn't improve for N evals
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
    
    # ── Optimization: Gradient checkpointing ──
    # Trades ~20% compute for ~50% VRAM reduction
    model.config.use_cache = False  # Required when gradient checkpointing is on
    model.gradient_checkpointing_enable()
    logger.info("✅ Gradient checkpointing enabled (saves ~50% VRAM)")
    
    # Data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )
    
    # Metrics
    compute_metrics = create_compute_metrics(processor)
    
    # Training arguments
    training_args = get_training_args(output_dir=output_dir, **training_kwargs)
    
    # ── Callbacks ──
    callbacks = []
    
    # Early stopping: halt if WER doesn't improve for N evals
    callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))
    logger.info(f"✅ Early stopping enabled (patience={early_stopping_patience} evals)")
    
    # Encoder freezing: freeze for first N epochs
    if freeze_encoder_epochs > 0:
        callbacks.append(EncoderUnfreezeCallback(unfreeze_after_epoch=freeze_encoder_epochs))
        logger.info(f"✅ Encoder will be frozen for first {freeze_encoder_epochs} epoch(s)")
    
    # Trainer
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
        callbacks=callbacks,
    )
    
    logger.info("🚀 Starting optimized training...")
    trainer.train()
    
    # Save best model
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    
    logger.info(f"💾 Model saved to {output_dir}")
    
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
