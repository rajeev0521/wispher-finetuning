"""
English Word Detector for Hindi Text
======================================
Detects English words written in Devanagari script within Hindi transcripts.
Uses multi-signal fusion: dictionary lookup, transliteration, and frequency analysis.
"""

import re
import os
import json
from typing import List, Dict, Tuple, Optional, Set
from collections import Counter

# ============================================================================
# COMMON ENGLISH WORDS IN DEVANAGARI (Expanded Lookup Table)
# ============================================================================

# Frequently occurring English words transliterated to Devanagari in Hindi conversations
# This is a curated list — extend as needed from actual data analysis
ENGLISH_DEVANAGARI_MAP = {
    # Technology
    'फोन': 'phone', 'मोबाइल': 'mobile', 'कंप्यूटर': 'computer',
    'लैपटॉप': 'laptop', 'डेस्कटॉप': 'desktop', 'इंटरनेट': 'internet',
    'ऑनलाइन': 'online', 'ऑफलाइन': 'offline', 'वेबसाइट': 'website',
    'ऐप': 'app', 'सॉफ्टवेयर': 'software', 'वाईफाई': 'wifi',
    'सोशल': 'social', 'मीडिया': 'media', 'यूट्यूब': 'youtube',
    'गूगल': 'google', 'व्हाट्सएप': 'whatsapp', 'इंस्टाग्राम': 'instagram',
    'फेसबुक': 'facebook', 'वीडियो': 'video', 'ऑडियो': 'audio',
    'नेटवर्क': 'network', 'सर्वर': 'server', 'डेटा': 'data',
    
    # Education
    'स्कूल': 'school', 'कॉलेज': 'college', 'यूनिवर्सिटी': 'university',
    'क्लास': 'class', 'टीचर': 'teacher', 'स्टूडेंट': 'student',
    'एग्जाम': 'exam', 'टेस्ट': 'test', 'रिजल्ट': 'result',
    'डिग्री': 'degree', 'कोर्स': 'course', 'सब्जेक्ट': 'subject',
    'ट्यूशन': 'tuition', 'होमवर्क': 'homework', 'प्रोजेक्ट': 'project',
    
    # Work
    'जॉब': 'job', 'ऑफिस': 'office', 'कंपनी': 'company',
    'बिजनेस': 'business', 'मीटिंग': 'meeting', 'इंटरव्यू': 'interview',
    'सैलरी': 'salary', 'वर्क': 'work', 'बॉस': 'boss',
    'मैनेजर': 'manager', 'स्टाफ': 'staff', 'टीम': 'team',
    'टारगेट': 'target', 'प्रमोशन': 'promotion', 'रिज्यूमे': 'resume',
    
    # Common words
    'टाइम': 'time', 'प्रॉब्लम': 'problem', 'सॉल्यूशन': 'solution',
    'लाइफ': 'life', 'फैमिली': 'family', 'फ्रेंड': 'friend',
    'फ्रेंड्स': 'friends', 'पार्टी': 'party', 'ग्रुप': 'group',
    'लिस्ट': 'list', 'प्लान': 'plan', 'टॉपिक': 'topic',
    'पॉइंट': 'point', 'रिलेशनशिप': 'relationship', 'एक्सपीरियंस': 'experience',
    'एक्चुअली': 'actually', 'बेसिकली': 'basically', 'ऑब्वियसली': 'obviously',
    'डेफिनेटली': 'definitely', 'सीरियसली': 'seriously',
    
    # Lifestyle
    'डाइट': 'diet', 'जिम': 'gym', 'फिटनेस': 'fitness',
    'हेल्थ': 'health', 'हेल्दी': 'healthy', 'फैशन': 'fashion',
    'शॉपिंग': 'shopping', 'ब्रांड': 'brand', 'स्टाइल': 'style',
    'ट्रेंड': 'trend', 'लुक': 'look', 'कूल': 'cool',
    
    # Travel/Transport
    'ट्रेन': 'train', 'बस': 'bus', 'टैक्सी': 'taxi',
    'एयरपोर्ट': 'airport', 'फ्लाइट': 'flight', 'होटल': 'hotel',
    'ट्रिप': 'trip', 'टूर': 'tour', 'टिकट': 'ticket',
    'ड्राइवर': 'driver', 'पार्किंग': 'parking', 'रोड': 'road',
    
    # Food
    'फूड': 'food', 'रेस्टोरेंट': 'restaurant', 'मेनू': 'menu',
    'कॉफी': 'coffee', 'पिज्जा': 'pizza', 'बर्गर': 'burger',
    'चॉकलेट': 'chocolate', 'आइसक्रीम': 'ice cream', 'केक': 'cake',
    
    # Entertainment
    'मूवी': 'movie', 'फिल्म': 'film', 'म्यूजिक': 'music',
    'गेम': 'game', 'क्रिकेट': 'cricket', 'फुटबॉल': 'football',
    'डांस': 'dance', 'डांसिंग': 'dancing', 'सॉन्ग': 'song',
    
    # Misc common
    'थैंक': 'thank', 'सॉरी': 'sorry', 'प्लीज': 'please',
    'ओके': 'okay', 'हेलो': 'hello', 'बाय': 'bye',
    'नंबर': 'number', 'टाइप': 'type', 'लेवल': 'level',
    'सिंपल': 'simple', 'ईजी': 'easy', 'इजी': 'easy',
    'लाइक': 'like', 'शेयर': 'share', 'फॉलो': 'follow',
    'स्टार्ट': 'start', 'स्टॉप': 'stop', 'चेंज': 'change',
    'फर्स्ट': 'first', 'लास्ट': 'last', 'नेक्स्ट': 'next',
    'बेस्ट': 'best', 'वर्स्ट': 'worst', 'फुल': 'full',
    'फ्री': 'free', 'सेफ': 'safe', 'रिस्क': 'risk',
    'चांस': 'chance', 'लक': 'luck', 'सक्सेस': 'success',
    'फैक्ट': 'fact', 'डिटेल': 'detail', 'बैलेंस': 'balance',
    'फीडबैक': 'feedback', 'अपडेट': 'update', 'रिपोर्ट': 'report',
    'नॉर्मल': 'normal', 'स्पेशल': 'special', 'फाइनल': 'final',
    'कार': 'car', 'बाइक': 'bike', 'साइकिल': 'cycle',
    'ट्रेडिशनल': 'traditional', 'मॉडर्न': 'modern', 'कल्चर': 'culture',
    'गिफ्ट': 'gift', 'गिफ्टेड': 'gifted', 'लैंड': 'land',
    'फोटो': 'photo', 'कैमरा': 'camera', 'सेल्फी': 'selfie',
    'पैशन': 'passion', 'टैलेंट': 'talent', 'स्किल': 'skill',
    'फ्यूचर': 'future', 'कैरियर': 'career', 'गोल': 'goal',
    'बर्थडे': 'birthday', 'फेवरेट': 'favourite', 'पसंदीदा': None,  # This is Hindi
    'सेम': 'same', 'डिफरेंट': 'different',
    'ब्रेक': 'break', 'ट्रेनिंग': 'training',
    'हार्ट': 'heart', 'प्योर': 'pure', 'फेस': 'face',
    'बिहेव': 'behave', 'इंफॉर्मेशन': 'information',
    'फ्लोर': 'floor', 'लिफ्ट': 'lift',
    'स्ट्रगल': 'struggle', 'स्ट्रीट': 'street',
    'लैंग्वेज': 'language', 'एरिया': 'area',
    'माइंड': 'mind', 'सपोर्ट': 'support',
    'हेल्प': 'help', 'क्लियर': 'clear',
    'एक्सप्लोर': 'explore', 'सिम्पल': 'simple',
    'पेमेंट': 'payment', 'ऑर्डर': 'order',
    'डिलीवरी': 'delivery', 'रिव्यू': 'review',
    'इंडिया': 'India', 'कॉल': 'call',
}

# Remove entries where value is None (they are actually Hindi words)
ENGLISH_DEVANAGARI_MAP = {k: v for k, v in ENGLISH_DEVANAGARI_MAP.items() if v is not None}


# ============================================================================
# DEVANAGARI SCRIPT DETECTION
# ============================================================================

DEVANAGARI_RANGE = re.compile(r'[\u0900-\u097F]')
LATIN_RANGE = re.compile(r'[a-zA-Z]')


def is_devanagari(text: str) -> bool:
    """Check if text is primarily in Devanagari script."""
    devanagari_chars = len(DEVANAGARI_RANGE.findall(text))
    total_alpha = devanagari_chars + len(LATIN_RANGE.findall(text))
    if total_alpha == 0:
        return False
    return devanagari_chars / total_alpha > 0.5


def is_latin(text: str) -> bool:
    """Check if text contains Latin characters."""
    return bool(LATIN_RANGE.search(text))


# ============================================================================
# ENGLISH WORD DETECTOR
# ============================================================================

class EnglishWordDetector:
    """
    Detects English words in Hindi (Devanagari) text.
    
    Uses multi-signal approach:
    1. Direct Latin character detection (when ASR outputs English)
    2. Lookup table matching (known English→Devanagari mappings)
    3. Pattern-based detection (common English word patterns in Devanagari)
    """
    
    def __init__(self, custom_mappings: Optional[Dict] = None):
        self.mappings = ENGLISH_DEVANAGARI_MAP.copy()
        if custom_mappings:
            self.mappings.update(custom_mappings)
        
        # Build reverse map (English → Devanagari) for reference
        self.reverse_map = {v: k for k, v in self.mappings.items()}
    
    def detect(self, text: str) -> List[Dict]:
        """
        Detect English words in Hindi text.
        
        Returns list of dicts with:
            - word: the original word
            - english: the English equivalent (if known)
            - detection_method: how it was detected
            - position: word index in text
        """
        words = text.split()
        detections = []
        
        for i, word in enumerate(words):
            word_clean = re.sub(r'[।,?!;:\.\-]', '', word)
            
            if not word_clean:
                continue
            
            # Signal 1: Direct Latin character detection
            if is_latin(word_clean):
                detections.append({
                    'word': word,
                    'clean_word': word_clean,
                    'english': word_clean.lower(),
                    'detection_method': 'latin_script',
                    'confidence': 'high',
                    'position': i,
                })
                continue
            
            # Signal 2: Lookup table
            if word_clean in self.mappings:
                detections.append({
                    'word': word,
                    'clean_word': word_clean,
                    'english': self.mappings[word_clean],
                    'detection_method': 'lookup_table',
                    'confidence': 'high',
                    'position': i,
                })
                continue
            
            # Signal 3: Partial match heuristics
            # Check if word starts/ends with known English patterns in Devanagari
            for dev_word, eng_word in self.mappings.items():
                if len(word_clean) > 3 and len(dev_word) > 3:
                    # Check for suffix variations (e.g., "फोन" → "फ़ोन", "फोनों")
                    if word_clean.startswith(dev_word) and len(word_clean) - len(dev_word) <= 3:
                        detections.append({
                            'word': word,
                            'clean_word': word_clean,
                            'english': eng_word + ' (variant)',
                            'detection_method': 'partial_match',
                            'confidence': 'medium',
                            'position': i,
                        })
                        break
        
        return detections
    
    def tag_text(self, text: str) -> str:
        """
        Return tagged version of text with [EN]...[/EN] around English words.
        
        Example:
            Input:  "मेरा इंटरव्यू बहुत अच्छा गया"
            Output: "मेरा [EN]इंटरव्यू[/EN] बहुत अच्छा गया"
        """
        detections = self.detect(text)
        
        if not detections:
            return text
        
        # Get positions of English words
        en_positions = {d['position'] for d in detections}
        
        words = text.split()
        tagged_words = []
        
        for i, word in enumerate(words):
            if i in en_positions:
                tagged_words.append(f"[EN]{word}[/EN]")
            else:
                tagged_words.append(word)
        
        return ' '.join(tagged_words)
    
    def analyze_transcript(self, text: str) -> Dict:
        """
        Full analysis of a transcript for English words.
        Returns summary with tagged text, detection list, and stats.
        """
        detections = self.detect(text)
        tagged = self.tag_text(text)
        
        words = text.split()
        total_words = len(words)
        english_count = len(detections)
        
        return {
            'original': text,
            'tagged': tagged,
            'english_words': detections,
            'total_words': total_words,
            'english_word_count': english_count,
            'english_ratio': english_count / total_words if total_words > 0 else 0,
        }


# ============================================================================
# DEMO / TEST
# ============================================================================

if __name__ == "__main__":
    detector = EnglishWordDetector()
    
    test_cases = [
        "मेरा इंटरव्यू बहुत अच्छा गया और मुझे जॉब मिल गई",
        "ये प्रॉब्लम सॉल्व नहीं हो रहा",
        "एक सिंपल और सादा वे में",
        "पहले डेस्कटॉप था ना बाद में लैपटॉप आया",
        "मुझे म्यूजिक सुनना पसंद है गाना भी पसंद है एक्चुअली",
        "जी फीडबैक मिलने पर सुधार करना",
        "तो भारत तो एक खुद से ही गिफ्टेड लैंड है तो हमें इसको एक्सप्लोर करना चाहिए",
    ]
    
    print("English Word Detection in Hindi Text")
    print("=" * 60)
    for text in test_cases:
        result = detector.analyze_transcript(text)
        print(f"\nInput:  {result['original']}")
        print(f"Tagged: {result['tagged']}")
        print(f"English words found: {result['english_word_count']}/{result['total_words']}")
        for ew in result['english_words']:
            print(f"  → '{ew['clean_word']}' = {ew['english']} [{ew['detection_method']}, {ew['confidence']}]")
