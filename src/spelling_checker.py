"""
Hindi Spelling Checker
=======================
Multi-layer spelling verification for ~177K unique Hindi words.
Classifies each word as correct/incorrect with confidence scoring.
"""

import re
import os
import csv
import json
import unicodedata
from typing import Dict, List, Tuple, Optional, Set
from collections import Counter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# DEVANAGARI CHARACTER VALIDATION
# ============================================================================

# Valid Devanagari Unicode ranges
DEVANAGARI_CONSONANTS = set(chr(c) for c in range(0x0915, 0x093A))  # क-ह
DEVANAGARI_VOWELS = set(chr(c) for c in range(0x0904, 0x0915))      # अ-औ
DEVANAGARI_MATRAS = set(chr(c) for c in range(0x093E, 0x094E))       # Vowel signs
DEVANAGARI_VIRAMA = {'\u094D'}  # हलंत ्
DEVANAGARI_ANUSVARA = {'\u0902'}  # अनुस्वार ं
DEVANAGARI_VISARGA = {'\u0903'}   # विसर्ग ः
DEVANAGARI_NUKTA = {'\u093C'}     # नुक्ता ़
DEVANAGARI_CHANDRABINDU = {'\u0901'}  # चंद्रबिंदु ँ

ALL_DEVANAGARI = (DEVANAGARI_CONSONANTS | DEVANAGARI_VOWELS | DEVANAGARI_MATRAS | 
                  DEVANAGARI_VIRAMA | DEVANAGARI_ANUSVARA | DEVANAGARI_VISARGA | 
                  DEVANAGARI_NUKTA | DEVANAGARI_CHANDRABINDU)

# Common invalid character sequences in Devanagari
INVALID_PATTERNS = [
    r'[\u093E-\u094D]{3,}',           # Three or more consecutive matras/modifiers
    r'\u094D\u094D',                    # Double virama
    r'[\u093E-\u094C]\u093E',          # Matra followed by aa matra (usually invalid)
]


def is_valid_devanagari_structure(word: str) -> bool:
    """
    Check if a word has valid Devanagari character structure.
    Returns True if structure looks valid, False if it contains
    obviously invalid character sequences.
    """
    for pattern in INVALID_PATTERNS:
        if re.search(pattern, word):
            return False
    return True


def strip_punctuation(word: str) -> str:
    """Remove attached punctuation from a word."""
    return re.sub(r'[।,?!;:\"\'""''…\-–—\(\)\[\]\{\}\.\,]', '', word).strip()


# ============================================================================
# CORE HINDI VOCABULARY (High-frequency words)
# ============================================================================

# Top ~500 most common Hindi words — known correct spellings
# This acts as a baseline "dictionary" layer
CORE_HINDI_VOCAB = {
    # Pronouns
    'मैं', 'मेरा', 'मेरी', 'मेरे', 'मुझे', 'मैंने', 'हम', 'हमें', 'हमारा', 'हमारी', 'हमारे',
    'तुम', 'तुम्हारा', 'तुम्हें', 'तू', 'तेरा', 'तेरी',
    'आप', 'आपका', 'आपकी', 'आपके', 'आपको', 'आपने', 'आपसे',
    'वह', 'वो', 'उसका', 'उसकी', 'उसके', 'उसको', 'उसमें', 'उसने', 'उसे', 'उससे', 'उसी',
    'ये', 'यह', 'इस', 'इसका', 'इसकी', 'इसके', 'इसको', 'इसमें', 'इससे', 'इसे', 'इसी',
    'वे', 'उन', 'उनका', 'उनकी', 'उनके', 'उनको', 'उन्हें', 'उन्होंने', 'उनसे',
    'कोई', 'कुछ', 'सब', 'सभी', 'हर', 'कौन', 'क्या',
    
    # Verbs (common forms)
    'है', 'हैं', 'था', 'थी', 'थे', 'हो', 'होता', 'होती', 'होते', 'होगा', 'होगी',
    'होना', 'होने', 'होनी', 'होंगे', 'हुआ', 'हुई', 'हुए',
    'करना', 'करता', 'करती', 'करते', 'करें', 'करो', 'किया', 'किए', 'करने', 'करके',
    'करता', 'करेगा', 'करेंगे', 'करनी',
    'जाना', 'जाता', 'जाती', 'जाते', 'जाए', 'जाएगा', 'जाएंगे', 'जाओ', 'जाएं',
    'आना', 'आता', 'आती', 'आते', 'आया', 'आई', 'आए', 'आएगा',
    'देना', 'देता', 'देती', 'देते', 'दिया', 'दी', 'दिए', 'देंगे', 'देखना',
    'लेना', 'लेता', 'लेती', 'लेते', 'लिया', 'ली', 'लिए', 'लेंगे',
    'बोलना', 'बोलता', 'बोलती', 'बोलते', 'बोला', 'बोले', 'बोलिए', 'बोलने',
    'कहना', 'कहता', 'कहती', 'कहते', 'कहा', 'कहें', 'कहीं',
    'मिलना', 'मिलता', 'मिलती', 'मिलते', 'मिला', 'मिली', 'मिले', 'मिलेगा', 'मिलने',
    'रहना', 'रहता', 'रहती', 'रहते', 'रहा', 'रही', 'रहे', 'रहेगा', 'रहेंगे', 'रहने',
    'पढ़ना', 'पढ़ाई', 'पढ़ने', 'पढ़',
    'खाना', 'खाते', 'खाने', 'खाया', 'खा',
    'सोचना', 'सोचते', 'सोचा', 'सोच',
    'देखना', 'देखता', 'देखते', 'देखा', 'देखी', 'देखे', 'देखने', 'देखो', 'देख',
    'बताना', 'बताते', 'बताया', 'बता', 'बताइए', 'बताईए', 'बताएं', 'बताओ', 'बताने',
    'रखना', 'रखते', 'रखा', 'रख', 'रखने',
    'सुन', 'सुना', 'सुनना', 'सीख', 'सीखा', 'सीखना', 'सीखने',
    
    # Postpositions
    'का', 'की', 'के', 'को', 'में', 'से', 'पर', 'पे', 'तक', 'ने',
    'के लिए', 'के बारे', 'के साथ', 'के बाद',
    
    # Conjunctions & Particles
    'और', 'या', 'लेकिन', 'तो', 'भी', 'ही', 'न', 'ना', 'नहीं', 'मत',
    'कि', 'जो', 'जब', 'तब', 'अगर', 'अब', 'फिर', 'तभी',
    'क्योंकि', 'इसलिए', 'ताकि', 'जैसे', 'वैसे', 'ऐसे',
    
    # Adverbs
    'बहुत', 'बहोत', 'काफी', 'ज्यादा', 'ज़्यादा', 'कम', 'थोड़ा', 'थोड़ी', 'थोड़े',
    'अभी', 'आज', 'कल', 'पहले', 'बाद', 'यहां', 'यहाँ', 'वहां', 'वहाँ',
    'जल्दी', 'धीरे', 'हमेशा', 'कभी', 'अक्सर', 'रोज',
    'सिर्फ', 'बस', 'ठीक', 'अच्छा', 'अच्छी', 'अच्छे',
    'सही', 'गलत', 'बहुत', 'इतना', 'इतनी', 'इतने', 'उतना', 'जितना',
    'ऊपर', 'नीचे', 'अंदर', 'बाहर', 'आगे', 'पीछे', 'दूर', 'पास',
    
    # Nouns (common)
    'लोग', 'लोगों', 'आदमी', 'बच्चे', 'बच्चों', 'बच्चा', 'दोस्त', 'दोस्तों', 'दोस्ती',
    'घर', 'स्कूल', 'काम', 'बात', 'बातें', 'चीज', 'चीजें', 'चीज़', 'चीज़ें',
    'समय', 'दिन', 'रात', 'सुबह', 'शाम', 'साल', 'महीने', 'घंटे', 'घंटा', 'मिनट',
    'पानी', 'खाना', 'दाल', 'चावल', 'रोटी', 'सब्जी', 'चाय',
    'शहर', 'गांव', 'देश', 'जगह', 'रास्ता', 'रास्ते',
    'भाई', 'मम्मी', 'पापा', 'पिता', 'माता', 'परिवार',
    'नाम', 'पैसे', 'पैसा', 'रुपए', 'नौकरी',
    'भाषा', 'शादी', 'त्योहार', 'त्यौहार',
    'शौक', 'आदत', 'अनुभव', 'जीवन', 'जिंदगी',
    'मदद', 'कोशिश', 'तरीका', 'तरीके', 'वजह', 'कारण',
    'विचार', 'सपना', 'याद', 'यादें',
    
    # Adjectives
    'बड़ा', 'बड़ी', 'बड़े', 'छोटा', 'छोटी', 'छोटे',
    'नया', 'नई', 'नए', 'पुराना', 'पुरानी', 'पुराने',
    'अलग', 'खास', 'खुश', 'खुशी',
    'पूरा', 'पूरी', 'पूरे', 'सारा', 'सारी', 'सारे',
    'पहला', 'पहली', 'दूसरा', 'दूसरी', 'दूसरे',
    
    # Numbers
    'एक', 'दो', 'तीन', 'चार', 'पांच', 'छह', 'सात', 'आठ', 'नौ', 'दस',
    'बीस', 'तीस', 'चालीस', 'पचास', 'साठ', 'सत्तर', 'अस्सी', 'नब्बे', 'सौ', 'हजार',
    'पंद्रह', 'पच्चीस', 'बारहवीं',
    
    # Fillers & Interjections
    'हां', 'हाँ', 'हम्म', 'ह्म्म', 'अच्छा', 'ठीक', 'ओके',
    'अरे', 'हेलो', 'हैलो', 'हलो', 'यार', 'भई',
    'जी', 'सर', 'मैम', 'मेम',
    'धन्यवाद', 'नमस्कार', 'नमस्ते',
    'हा', 'हे', 'ओ', 'ए', 'अ', 'उम्म', 'उह', 'आह', 'अह',
    
    # Common derived words  
    'बिल्कुल', 'बिलकुल', 'एकदम', 'वगैरह', 'वगैरा',
    'शायद', 'लगभग', 'बाकी', 'अलावा', 'बिना',
    'चलो', 'चलिए', 'देखिए', 'लीजिए', 'कीजिए', 'बोलिए',
    'हांजी', 'अपन', 'अपना', 'अपनी', 'अपने',
    'मतलब', 'हिसाब', 'तरह', 'प्रकार',
    'दिमाग', 'दिल', 'हाथ',
    'वही', 'यही', 'कहीं', 'जहां', 'कहां',
    'वहीं', 'इधर', 'उधर',
}


# ============================================================================
# HINDI MORPHOLOGICAL SUFFIXES
# ============================================================================

# Common Hindi suffixes for morphological decomposition
HINDI_SUFFIXES = [
    # Verb suffixes
    'ता', 'ती', 'ते', 'ना', 'ने', 'नी', 'ें', 'ों', 'ूं', 'ूँ',
    'कर', 'के', 'ा', 'ी', 'े', 'ो', 'ूं', 'ाओ',
    # Noun suffixes
    'ों', 'ें', 'ओं', 'ियों', 'ियां', 'ियाँ',
    # Adjective suffixes
    'दार', 'वाला', 'वाली', 'वाले',
]


# ============================================================================
# SPELLING CHECKER
# ============================================================================

class HindiSpellingChecker:
    """
    Multi-layer spelling verification for Hindi words.
    
    Layers:
    1. Core vocabulary lookup (high confidence)
    2. Devanagari structure validation
    3. English transliteration detection
    4. Morphological decomposition
    5. Frequency & pattern-based heuristics
    """
    
    def __init__(self, additional_vocab: Optional[Set[str]] = None):
        self.vocab = CORE_HINDI_VOCAB.copy()
        if additional_vocab:
            self.vocab.update(additional_vocab)
        
        # English words in Devanagari (from english_detector module)
        from src.english_detector import ENGLISH_DEVANAGARI_MAP
        self.english_devanagari = set(ENGLISH_DEVANAGARI_MAP.keys())
    
    def check_word(self, word: str) -> Dict:
        """
        Classify a single word as correct/incorrect with confidence.
        
        Returns:
            {
                'word': original word,
                'classification': 'correct' | 'incorrect',
                'confidence': 'high' | 'medium' | 'low',
                'reason': explanation string,
                'layer': which layer made the decision
            }
        """
        # Strip attached punctuation
        clean = strip_punctuation(word)
        
        if not clean:
            return {
                'word': word,
                'classification': 'correct',
                'confidence': 'high',
                'reason': 'Punctuation-only token',
                'layer': 'punctuation'
            }
        
        # Layer 1: Core vocabulary lookup
        if clean in self.vocab:
            return {
                'word': word,
                'classification': 'correct',
                'confidence': 'high',
                'reason': f'Found in core Hindi vocabulary',
                'layer': 'dictionary'
            }
        
        # Layer 2: English transliteration detection
        if clean in self.english_devanagari:
            return {
                'word': word,
                'classification': 'correct',
                'confidence': 'high',
                'reason': f'Recognized English word in Devanagari script (per transcription guidelines)',
                'layer': 'english_transliteration'
            }
        
        # Layer 3: Devanagari structure validation
        if not is_valid_devanagari_structure(clean):
            return {
                'word': word,
                'classification': 'incorrect',
                'confidence': 'high',
                'reason': f'Invalid Devanagari character sequence',
                'layer': 'structure_validation'
            }
        
        # Layer 4: Morphological decomposition
        for suffix in sorted(HINDI_SUFFIXES, key=len, reverse=True):
            if clean.endswith(suffix) and len(clean) > len(suffix) + 1:
                root = clean[:-len(suffix)]
                if root in self.vocab:
                    return {
                        'word': word,
                        'classification': 'correct',
                        'confidence': 'medium',
                        'reason': f'Root "{root}" + suffix "{suffix}" (morphologically valid)',
                        'layer': 'morphology'
                    }
        
        # Layer 5: Check if it's a number word
        from src.number_normalizer import ALL_NUMBER_WORDS
        if clean in ALL_NUMBER_WORDS:
            return {
                'word': word,
                'classification': 'correct',
                'confidence': 'high',
                'reason': 'Hindi number word',
                'layer': 'number'
            }
        
        # Layer 6: Single Devanagari character (valid filler/particle)
        if len(clean) == 1 and any(c in ALL_DEVANAGARI for c in clean):
            return {
                'word': word,
                'classification': 'correct',
                'confidence': 'medium',
                'reason': 'Single Devanagari character (likely filler/particle)',
                'layer': 'single_char'
            }
        
        # Layer 7: Contains non-Devanagari characters (mixed script)
        has_latin = bool(re.search(r'[a-zA-Z]', clean))
        if has_latin:
            return {
                'word': word,
                'classification': 'incorrect',
                'confidence': 'medium',
                'reason': 'Mixed script (contains Latin characters) — may be ASR artifact',
                'layer': 'mixed_script'
            }
        
        # Default: Unknown — classify as low confidence
        return {
            'word': word,
            'classification': 'incorrect',
            'confidence': 'low',
            'reason': 'Not found in any dictionary or pattern — possibly misspelling, dialect, or unrecognized word',
            'layer': 'unknown'
        }
    
    def check_word_list(self, words: List[str]) -> List[Dict]:
        """Check a list of words and return classifications."""
        results = []
        for word in words:
            results.append(self.check_word(word))
        return results
    
    def check_csv(
        self,
        csv_path: str,
        output_csv_path: str,
        word_column: str = 'word'
    ) -> Dict:
        """
        Process the entire unique words CSV file.
        
        Returns summary stats and writes output CSV.
        """
        import pandas as pd
        
        df = pd.read_csv(csv_path)
        words = df[word_column].astype(str).tolist()
        
        logger.info(f"Checking {len(words)} words...")
        
        results = self.check_word_list(words)
        
        # Build output DataFrame
        out_df = pd.DataFrame({
            'word': [r['word'] for r in results],
            'classification': [r['classification'] for r in results],
            'confidence': [r['confidence'] for r in results],
            'reason': [r['reason'] for r in results],
            'layer': [r['layer'] for r in results],
        })
        
        # Save
        out_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        logger.info(f"Results saved to {output_csv_path}")
        
        # Stats
        correct = sum(1 for r in results if r['classification'] == 'correct')
        incorrect = sum(1 for r in results if r['classification'] == 'incorrect')
        
        confidence_breakdown = Counter(r['confidence'] for r in results)
        layer_breakdown = Counter(r['layer'] for r in results)
        
        stats = {
            'total_words': len(words),
            'correct_spelling': correct,
            'incorrect_spelling': incorrect,
            'confidence_breakdown': dict(confidence_breakdown),
            'layer_breakdown': dict(layer_breakdown),
        }
        
        logger.info(f"Stats: {json.dumps(stats, indent=2, ensure_ascii=False)}")
        
        return stats


# ============================================================================
# DEMO
# ============================================================================

if __name__ == "__main__":
    checker = HindiSpellingChecker()
    
    test_words = [
        'बहुत', 'कंप्यूटर', 'अच्छा', 'हम्म', 'इंरव्यू',  # intentional misspelling
        'स्कूल', 'पढ़ाई', 'बिजनेस', 'बच्चों', 'xyzहां',  # mixed script
    ]
    
    print("Hindi Spelling Checker Demo")
    print("=" * 60)
    for word in test_words:
        result = checker.check_word(word)
        status = "✓" if result['classification'] == 'correct' else "✗"
        print(f"{status} '{word}' → {result['classification']} [{result['confidence']}] ({result['reason']})")
