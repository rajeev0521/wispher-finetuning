"""
Hindi Spelling Checker (v2 — Production-grade)
================================================
Multi-layer spelling verification for ~177K unique Hindi words.
Uses a frequency-ranked self-derived dictionary from the input corpus,
structural validation, morphological decomposition (multi-level),
English transliteration detection, and pattern-based heuristics.

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

# Invalid character sequence patterns
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


def is_pure_devanagari(word: str) -> bool:
    """Check if word is composed entirely of Devanagari characters."""
    for char in word:
        cp = ord(char)
        # Devanagari block: 0900-097F, Devanagari Extended: A8E0-A8FF
        if not ((0x0900 <= cp <= 0x097F) or (0xA8E0 <= cp <= 0xA8FF)):
            return False
    return True


def strip_punctuation(word: str) -> str:
    """Remove attached punctuation from a word."""
    return re.sub(r'[।,?!;:\"\'""''…\-–—\(\)\[\]\{\}\.\,]', '', word).strip()


# ============================================================================
# COMPREHENSIVE HINDI VOCABULARY
# ============================================================================

# ~1200+ verified correct Hindi words — expanded from the original ~500
CORE_HINDI_VOCAB = {
    # ---- Pronouns ----
    'मैं', 'मेरा', 'मेरी', 'मेरे', 'मुझे', 'मैंने', 'मुझमें', 'मुझसे', 'मुझको',
    'हम', 'हमें', 'हमारा', 'हमारी', 'हमारे', 'हमने', 'हमसे', 'हमको', 'हमलोग', 'हमे',
    'तुम', 'तुम्हारा', 'तुम्हारी', 'तुम्हारे', 'तुम्हें', 'तू', 'तेरा', 'तेरी', 'तेरे', 'तुझे',
    'आप', 'आपका', 'आपकी', 'आपके', 'आपको', 'आपने', 'आपसे',
    'वह', 'वो', 'उसका', 'उसकी', 'उसके', 'उसको', 'उसमें', 'उसने', 'उसे', 'उससे', 'उसी', 'उसमे',
    'ये', 'यह', 'इस', 'इसका', 'इसकी', 'इसके', 'इसको', 'इसमें', 'इससे', 'इसे', 'इसी',
    'वे', 'उन', 'उनका', 'उनकी', 'उनके', 'उनको', 'उन्हें', 'उन्होंने', 'उनसे',
    'कोई', 'कुछ', 'सब', 'सभी', 'हर', 'कौन', 'क्या', 'सबको', 'सबके', 'सबसे',
    'खुद', 'स्वयं', 'अपना', 'अपनी', 'अपने', 'अपन',
    'कहीं', 'कहां', 'कहाँ', 'जहां', 'जहाँ', 'वहां', 'वहाँ', 'यहां', 'यहाँ', 'वहीं', 'यही', 'वही',

    # ---- Verbs (comprehensive conjugations) ----
    'है', 'हैं', 'था', 'थी', 'थे', 'हो', 'होता', 'होती', 'होते', 'होगा', 'होगी',
    'होना', 'होने', 'होनी', 'होंगे', 'हुआ', 'हुई', 'हुए', 'हूं', 'हूँ', 'हु', 'हू',
    'करना', 'करता', 'करती', 'करते', 'करें', 'करो', 'किया', 'किए', 'करने', 'करके',
    'करेगा', 'करेंगे', 'करनी', 'करा', 'करी', 'करेगी', 'करोगे',
    'जाना', 'जाता', 'जाती', 'जाते', 'जाए', 'जाएगा', 'जाएंगे', 'जाओ', 'जाएं',
    'जा', 'गया', 'गई', 'गए', 'जाने', 'जाके', 'जाकर', 'जाएगी',
    'आना', 'आता', 'आती', 'आते', 'आया', 'आई', 'आए', 'आएगा', 'आ', 'आने', 'आओ',
    'देना', 'देता', 'देती', 'देते', 'दिया', 'दी', 'दिए', 'देंगे', 'दे', 'देने',
    'लेना', 'लेता', 'लेती', 'लेते', 'लिया', 'ली', 'लिए', 'लेंगे', 'ले', 'लो', 'लेके', 'लेकर', 'लेने',
    'बोलना', 'बोलता', 'बोलती', 'बोलते', 'बोला', 'बोले', 'बोलिए', 'बोलने', 'बोल',
    'कहना', 'कहता', 'कहती', 'कहते', 'कहा', 'कहें', 'कह', 'कही',
    'मिलना', 'मिलता', 'मिलती', 'मिलते', 'मिला', 'मिली', 'मिले', 'मिलेगा', 'मिलने', 'मिल',
    'रहना', 'रहता', 'रहती', 'रहते', 'रहा', 'रही', 'रहे', 'रहेगा', 'रहेंगे', 'रहने', 'रह',
    'पढ़ना', 'पढ़ाई', 'पढ़ने', 'पढ़', 'पढ़ा',
    'खाना', 'खाते', 'खाने', 'खाया', 'खा', 'खाता', 'खाती',
    'सोचना', 'सोचते', 'सोचा', 'सोच', 'सोचती',
    'देखना', 'देखता', 'देखती', 'देखते', 'देखा', 'देखी', 'देखे', 'देखने', 'देखो', 'देख',
    'बताना', 'बताते', 'बताया', 'बता', 'बताइए', 'बताईए', 'बताएं', 'बताओ', 'बताने',
    'रखना', 'रखते', 'रखा', 'रख', 'रखने', 'रखती',
    'सुन', 'सुना', 'सुनना', 'सुनी', 'सुने', 'सुनो',
    'सीख', 'सीखा', 'सीखना', 'सीखने', 'सीखी',
    'चलना', 'चलता', 'चलती', 'चलते', 'चला', 'चली', 'चले', 'चल', 'चलो', 'चलिए',
    'बनना', 'बनता', 'बनती', 'बनते', 'बना', 'बनी', 'बने', 'बन', 'बनाना', 'बनाया', 'बनाते', 'बनाने', 'बनाई',
    'लगना', 'लगता', 'लगती', 'लगते', 'लगा', 'लगी', 'लगे', 'लग', 'लगेगा',
    'मानना', 'मान', 'माना', 'मानते', 'मानी',
    'पाना', 'पाता', 'पाती', 'पाते', 'पा', 'पाएंगे',
    'सकना', 'सकता', 'सकती', 'सकते', 'सके',
    'चाहना', 'चाहता', 'चाहती', 'चाहते', 'चाहिए', 'चाहे', 'चाहेंगे',
    'मारना', 'मार', 'मारा', 'मारी',
    'निकलना', 'निकल', 'निकला', 'निकली', 'निकाल', 'निकाला',
    'पड़ना', 'पड़ता', 'पड़ती', 'पड़ा', 'पड़े', 'पड़ेगा', 'पड़',
    'रोक', 'रोका', 'रोकना',
    'समझ', 'समझना', 'समझा', 'समझते',
    'बदल', 'बदलना', 'बदला', 'बदलाव',
    'खेल', 'खेलना', 'खेलते', 'खेलने', 'खेला',
    'बढ़', 'बढ़ना', 'बढ़ा', 'बढ़िया',
    'छोड़', 'छोड़ना', 'छोड़ा', 'छोड़ी',
    'डाल', 'डालना', 'डाला',
    'पहुंच', 'पहुँच', 'पहुंचना',
    'बैठ', 'बैठना', 'बैठे', 'बैठा', 'बैठी',
    'उठ', 'उठना', 'उठा', 'उठाना',
    'भूल', 'भूलना',
    'टूट', 'टूटना', 'तोड़', 'तोड़ना',
    'खरीद', 'खरीदना', 'खरीदा',
    'बुला', 'बुलाना', 'बुलाया',
    'सो', 'सोना', 'सोया',
    'रो', 'रोना', 'रोया',
    'पी', 'पीना', 'पीने',
    'लिख', 'लिखना', 'लिखा', 'लिखी',
    'पूछ', 'पूछना', 'पूछा',
    'बेच', 'बेचना', 'बेचा',
    'चुका', 'चुकी', 'चुके',

    # ---- Postpositions ----
    'का', 'की', 'के', 'को', 'में', 'से', 'पर', 'पे', 'तक', 'ने',
    'बारे', 'साथ', 'बाद', 'लिए', 'बीच', 'ऊपर', 'नीचे', 'अंदर', 'बाहर',
    'आगे', 'पीछे', 'दूर', 'पास', 'बिना', 'तरफ',

    # ---- Conjunctions & Particles ----
    'और', 'या', 'लेकिन', 'तो', 'भी', 'ही', 'न', 'ना', 'नहीं', 'नही', 'मत',
    'कि', 'जो', 'जब', 'तब', 'अगर', 'अब', 'फिर', 'तभी',
    'क्योंकि', 'इसलिए', 'ताकि', 'जैसे', 'वैसे', 'ऐसे', 'जैसा', 'वैसा', 'ऐसा', 'ऐसी',
    'जिस', 'जिसे', 'जिसको', 'जिसके', 'जिससे', 'जिसमें', 'जिसने',
    'जितना', 'जितने', 'जितनी', 'उतना',
    'इन', 'इनका', 'इनकी', 'इनके',

    # ---- Adverbs ----
    'बहुत', 'बहोत', 'काफी', 'ज्यादा', 'ज़्यादा', 'कम', 'थोड़ा', 'थोड़ी', 'थोड़े',
    'अभी', 'आज', 'कल', 'पहले', 'बाद', 'जल्दी', 'धीरे',
    'हमेशा', 'कभी', 'अक्सर', 'रोज', 'रोजाना',
    'सिर्फ', 'बस', 'ठीक', 'अच्छा', 'अच्छी', 'अच्छे',
    'सही', 'गलत', 'इतना', 'इतनी', 'इतने',
    'ज्यादातर', 'लगभग', 'करीब',
    'साफ', 'सच', 'शायद',
    'एकदम', 'बिल्कुल', 'बिलकुल',

    # ---- Nouns (common) ----
    'लोग', 'लोगों', 'लोगो', 'आदमी', 'इंसान',
    'बच्चे', 'बच्चों', 'बच्चा', 'दोस्त', 'दोस्तों', 'दोस्ती',
    'घर', 'स्कूल', 'काम', 'बात', 'बातें', 'बाते', 'बातचीत',
    'चीज', 'चीजें', 'चीज़', 'चीज़ें', 'चीजे', 'चीज़े', 'चीजों', 'चीज़ों',
    'समय', 'दिन', 'रात', 'सुबह', 'शाम', 'साल', 'महीने', 'महीना',
    'घंटे', 'घंटा', 'मिनट', 'हफ्ता', 'हफ्ते',
    'पानी', 'खाना', 'दाल', 'चावल', 'रोटी', 'सब्जी', 'चाय', 'दूध', 'फल',
    'शहर', 'गांव', 'गाँव', 'देश', 'जगह', 'रास्ता', 'रास्ते', 'रोड', 'सड़क',
    'भाई', 'मम्मी', 'पापा', 'पिता', 'माता', 'परिवार', 'मां', 'माँ', 'बहन',
    'नाम', 'पैसे', 'पैसा', 'रुपए', 'रुपये', 'नौकरी',
    'भाषा', 'शादी', 'त्योहार', 'त्यौहार', 'होली', 'दिवाली',
    'शौक', 'आदत', 'अनुभव', 'जीवन', 'जिंदगी', 'ज़िंदगी',
    'मदद', 'कोशिश', 'तरीका', 'तरीके', 'वजह', 'कारण',
    'विचार', 'सपना', 'याद', 'यादें', 'यात्रा',
    'दिमाग', 'दिल', 'हाथ', 'पैर', 'आंख', 'आँख', 'कान', 'मुंह', 'मुँह', 'शरीर',
    'कंपनी', 'ऑफिस', 'बाजार', 'दुकान', 'मार्केट',
    'किताब', 'कागज', 'पेड़', 'मंदिर', 'मस्जिद',
    'दुनिया', 'समाज', 'सरकार', 'राजनीति',
    'संस्कृति', 'इतिहास', 'विज्ञान',
    'समस्या', 'हाल', 'माहौल', 'मौसम', 'बारिश', 'गर्मी', 'सर्दी',
    'फोन', 'मोबाइल', 'कंप्यूटर', 'इंटरनेट',
    'गाड़ी', 'ट्रेन', 'बस', 'बाइक', 'कार', 'साइकिल',
    'फोटो', 'वीडियो', 'गाना', 'गाने', 'फिल्म', 'मूवी',
    'आवाज', 'बचपन', 'आजकल', 'पल', 'वक्त',
    'सामान', 'सामने', 'बीच', 'मन', 'दर्द',
    'गलती', 'मेहनत', 'तैयारी', 'तैयार',
    'योजना', 'प्लान', 'टेस्ट', 'एग्जाम',
    'जानकारी', 'रुचि', 'ताकत', 'कमी',
    'छुट्टी', 'पसंद', 'पसंदीदा', 'खुशी', 'खुश',
    'सफर', 'गेम', 'खेल',
    'विषय', 'टॉपिक',
    'नंबर', 'ग्रुप', 'लड़की', 'लड़के', 'लड़का',
    'बर्थडे', 'पार्टी',
    'रूम', 'एरिया',
    'दिक्कत', 'मजा', 'मज़ा', 'मस्ती',
    'आराम', 'प्यार',

    # ---- Adjectives ----
    'बड़ा', 'बड़ी', 'बड़े', 'छोटा', 'छोटी', 'छोटे',
    'नया', 'नई', 'नए', 'पुराना', 'पुरानी', 'पुराने',
    'अलग', 'खास', 'मुश्किल', 'आसान',
    'पूरा', 'पूरी', 'पूरे', 'सारा', 'सारी', 'सारे',
    'पहला', 'पहली', 'दूसरा', 'दूसरी', 'दूसरे', 'तीसरा', 'तीसरी',
    'बुरा', 'बुरी', 'बेहतर', 'बेस्ट',
    'कितना', 'कितनी', 'कितने',
    'अगला', 'अगली',
    'खराब',

    # ---- Numbers ----
    'एक', 'दो', 'तीन', 'चार', 'पांच', 'पाँच', 'छह', 'छः', 'सात', 'आठ', 'नौ', 'दस',
    'ग्यारह', 'बारह', 'तेरह', 'चौदह', 'पंद्रह', 'सोलह', 'सत्रह', 'अठारह', 'उन्नीस',
    'बीस', 'इक्कीस', 'बाईस', 'तेईस', 'चौबीस', 'पच्चीस', 'छब्बीस', 'सत्ताईस', 'अट्ठाईस', 'उनतीस',
    'तीस', 'इकतीस', 'बत्तीस', 'तैंतीस', 'चौंतीस', 'पैंतीस', 'छत्तीस', 'सैंतीस', 'अड़तीस', 'उनतालीस',
    'चालीस', 'इकतालीस', 'बयालीस', 'तैंतालीस', 'चौवालीस', 'पैंतालीस', 'छियालीस', 'सैंतालीस', 'अड़तालीस', 'उनचास',
    'पचास', 'इक्यावन', 'बावन', 'तिरपन', 'चौवन', 'पचपन', 'छप्पन', 'सत्तावन', 'अट्ठावन', 'उनसठ',
    'साठ', 'इकसठ', 'बासठ', 'तिरसठ', 'चौंसठ', 'पैंसठ', 'छियासठ', 'सड़सठ', 'अड़सठ', 'उनहत्तर',
    'सत्तर', 'इकहत्तर', 'बहत्तर', 'तिहत्तर', 'चौहत्तर', 'पचहत्तर', 'छिहत्तर', 'सतहत्तर', 'अठहत्तर', 'उनासी',
    'अस्सी', 'इक्यासी', 'बयासी', 'तिरासी', 'चौरासी', 'पचासी', 'छियासी', 'सत्तासी', 'अट्ठासी', 'नवासी',
    'नब्बे', 'इक्यानवे', 'बानवे', 'तिरानवे', 'चौरानवे', 'पचानवे', 'छियानवे', 'सत्तानवे', 'अट्ठानवे', 'निन्यानवे',
    'सौ', 'हजार', 'हज़ार', 'लाख', 'करोड़', 'अरब',
    'पहले', 'दूसरे', 'तीसरे', 'आधा', 'आधी', 'डेढ़', 'ढाई', 'सवा', 'पौने',

    # ---- Fillers & Interjections ----
    'हां', 'हाँ', 'हम्म', 'ह्म्म', 'ओके', 'ओक',
    'अरे', 'हेलो', 'हैलो', 'हलो', 'यार', 'भई',
    'जी', 'सर', 'मैम', 'मेम',
    'धन्यवाद', 'नमस्कार', 'नमस्ते',
    'हा', 'हे', 'ओ', 'ए', 'अ', 'उम्म', 'उह', 'आह', 'अह', 'हुह',
    'हांजी', 'अच्छा', 'वाह',

    # ---- Common derived/compound words ----
    'वगैरह', 'वगैरा', 'बाकी', 'अलावा',
    'मतलब', 'हिसाब', 'तरह', 'प्रकार', 'रूप',
    'अलगअलग', 'आसपास',
    'यस', 'नो',
    'दोनों', 'कब', 'किस', 'किसी',

    # ---- Common English loanwords in Devanagari (frequently used in Hindi) ----
    'टाइम', 'टॉपिक', 'टाइप', 'लाइफ', 'लाइक', 'लाइन', 'लेवल', 'लास्ट', 'लेट',
    'फ्रेंड', 'फ्रेंड्स', 'फैमिली', 'फील', 'फ्री',
    'सोशल', 'मीडिया', 'ऑनलाइन', 'यूट्यूब', 'इंटरव्यू',
    'बिजनेस', 'प्रॉब्लम', 'क्लास', 'क्लियर',
    'नेक्स्ट', 'फर्स्ट', 'सेम', 'ब्रेक',
    'एक्सपीरियंस', 'एक्चुअली', 'बेसिकली',
    'स्ट्रीट', 'नॉर्मल', 'लोकल',
    'फेमस', 'फेवरेट', 'फोकस',
    'कॉलेज', 'हेल्प', 'डेली', 'डे',
    'एंड', 'टू', 'बट', 'यू',
    'कॉल', 'फोन', 'गोल', 'रूल',
    'पॉइंट', 'माइंड', 'कल्चर',
    'वर्क', 'ट्रिप', 'स्टार्ट',
    'सपोर्ट', 'इंग्लिश', 'हिंदी',
    'लैंग्वेज', 'नेटवर्क', 'मेन',
    'शेयर', 'चेंज', 'इंडिया', 'दिल्ली', 'बिहार', 'जिम',
    'साइड', 'बुक', 'यूज',
    'चैनल', 'ऐप', 'सॉफ्टवेयर', 'टेक्नोलॉजी',
    'स्किल', 'प्रैक्टिस', 'परफॉर्मेंस',
    'स्कॉलरशिप', 'कैरियर',
    'फैक्ट', 'रिजल्ट', 'प्रोसेस',
    'प्रोड्यूसर', 'क्रिएटिव',
    'सिस्टम', 'मैनेज', 'मैनेजमेंट',
    'कंटेंट', 'कम्युनिकेशन',
    'डॉक्टर', 'इंजीनियर', 'टीचर',
    'एजुकेशन', 'मोटिवेशन',
    'प्रेजेंटेशन', 'क्वालिटी',
    'हॉस्पिटल', 'हॉस्टल',
    'ट्रांसपोर्ट', 'पेपर',
    'कैमरा', 'टीवी',
    'क्रिकेट', 'फुटबॉल',
    'मूड', 'स्ट्रेस',
    'स्टूडेंट', 'स्टूडेंट्स',
    'ट्यूशन', 'कोचिंग',
    'बजट', 'मार्क', 'मार्क्स',
    'सब्जेक्ट', 'प्रोजेक्ट',
    'इवेंट', 'प्रोग्राम',
    'अकाउंट', 'पासवर्ड',
    'फीचर', 'अपडेट',
}

# ============================================================================
# EXPANDED MORPHOLOGICAL ANALYSIS
# ============================================================================

# Multi-level Hindi suffixes for decomposition (ordered by length, longest first)
HINDI_SUFFIXES = [
    # Long compound suffixes
    'ियों', 'ियां', 'ियाँ', 'ाइए', 'ईए', 'ओं',
    'वाला', 'वाली', 'वाले',
    'कार', 'दार', 'गार',
    # Verb suffixes
    'ता', 'ती', 'ते', 'ना', 'ने', 'नी',
    'ें', 'ों', 'ूं', 'ूँ', 'ां', 'ाँ',
    'ेगा', 'ेगी', 'ेंगे', 'ेंगी',
    'ाओ', 'ओ', 'ईं',
    # Noun/Adj suffixes
    'पन', 'ता', 'आई', 'आस',
    'ा', 'ी', 'े', 'ो',
]

# Common Hindi prefixes
HINDI_PREFIXES = [
    'अन', 'बे', 'ना', 'निर', 'नि', 'प्र', 'अति', 'अध', 'परि',
    'सम', 'सु', 'कु', 'दुर', 'दुस', 'अभि', 'उप', 'अधि',
]


# ============================================================================
# FREQUENCY-BASED DICTIONARY BUILDER
# ============================================================================

def build_frequency_dictionary(word_list_path: str, top_n: int = 5000) -> Set[str]:
    """
    Build a dictionary from the top-N words in a frequency-sorted word list.
    
    The input CSV is assumed to be sorted by frequency (most common first).
    High-frequency words in natural language are overwhelmingly correct spellings.
    
    Research basis: Zipf's law — the most frequent words in any corpus are almost
    always standard dictionary words. Words in the top 5000 by frequency cover
    ~95% of all running text and are nearly always correctly spelled.
    """
    import pandas as pd
    
    if not os.path.exists(word_list_path):
        logger.warning(f"Word list not found: {word_list_path}")
        return set()
    
    df = pd.read_csv(word_list_path)
    words = df.iloc[:, 0].astype(str).tolist()
    
    dictionary = set()
    for i, word in enumerate(words[:top_n]):
        clean = strip_punctuation(word)
        if clean and is_pure_devanagari(clean) and len(clean) >= 2:
            dictionary.add(clean)
    
    logger.info(f"Built frequency dictionary with {len(dictionary)} words from top {top_n}")
    return dictionary


# ============================================================================
# PATTERN DETECTORS
# ============================================================================

def is_english_transliteration(word: str) -> bool:
    """
    Detect if a word is likely an English transliteration in Devanagari.
    Uses multiple signals:
    1. Contains common English suffix patterns in Devanagari
    2. Contains consonant clusters rare in native Hindi but common in English
    """
    english_suffix_patterns = [
        'शन$', 'मेंट$', 'नेस$', 'टिव$', 'टिक$', 'इंग$',
        'ली$', 'फुल$', 'लेस$', 'एबल$', 'ेशन$',
        'र्स$', 'ट्स$', 'न्स$', 'क्स$',
        'िटी$', 'ेंस$', 'ेंट$', 'ैंड$',
    ]
    for pattern in english_suffix_patterns:
        if re.search(pattern, word):
            return True
    return False


def is_compound_word(word: str) -> bool:
    """Check if word contains a hyphen or underscore (compound/joined)."""
    return bool(re.search(r'[-_।.]', word))


def is_word_with_punctuation_attached(word: str) -> bool:
    """Check if word has punctuation mixed in (e.g., '499में', '।मेटा')."""
    return bool(re.search(r'[0-9।,.!?;:]', word))


def has_nukta_variant(word: str) -> bool:
    """
    Check if word contains nukta (़) characters.
    Words like ज़, फ़, क़ are Urdu-origin but valid in Hindi.
    """
    return '\u093C' in word


# ============================================================================
# SPELLING CHECKER (v2)
# ============================================================================

class HindiSpellingChecker:
    """
    Multi-layer spelling verification for Hindi words.
    
    Layers (in order of application):
    1. Punctuation/special token handling
    2. Core + frequency-based dictionary lookup (high confidence)
    3. English transliteration detection (high confidence)
    4. Devanagari structure validation (high confidence reject)
    5. Mixed content detection (numbers, punctuation attached)
    6. Multi-level morphological decomposition (medium confidence)
    7. English suffix pattern detection (medium confidence)
    8. Nukta/dialect variant tolerance (medium confidence)
    9. Single character particles (medium confidence)
    10. Default: unknown (low confidence)
    """
    
    def __init__(self, word_list_path: Optional[str] = None, freq_top_n: int = 5000):
        """
        Initialize the checker.
        
        Args:
            word_list_path: Path to frequency-sorted word CSV for self-derived dictionary
            freq_top_n: Number of top-frequency words to trust as correct
        """
        # Start with curated vocabulary
        self.vocab = CORE_HINDI_VOCAB.copy()
        
        # Add frequency-based dictionary from input corpus
        self.freq_dict = set()
        if word_list_path:
            self.freq_dict = build_frequency_dictionary(word_list_path, top_n=freq_top_n)
            self.vocab.update(self.freq_dict)
        
        # English words in Devanagari (from english_detector module)
        try:
            from src.english_detector import ENGLISH_DEVANAGARI_MAP
            self.english_devanagari = set(ENGLISH_DEVANAGARI_MAP.keys())
        except ImportError:
            self.english_devanagari = set()
        
        logger.info(f"Checker initialized: {len(self.vocab)} dictionary words, "
                     f"{len(self.english_devanagari)} English transliterations")
    
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
        # Strip attached punctuation for analysis
        clean = strip_punctuation(word)
        
        if not clean:
            return {
                'word': word,
                'classification': 'correct',
                'confidence': 'high',
                'reason': 'Punctuation-only token',
                'layer': 'punctuation'
            }
        
        # ----- Layer 1: Direct dictionary lookup (core + frequency) -----
        if clean in self.vocab:
            return {
                'word': word,
                'classification': 'correct',
                'confidence': 'high',
                'reason': 'Found in Hindi dictionary (curated + frequency-validated)',
                'layer': 'dictionary'
            }
        
        # ----- Layer 2: English transliteration lookup -----
        if clean in self.english_devanagari:
            return {
                'word': word,
                'classification': 'correct',
                'confidence': 'high',
                'reason': 'Recognized English word in Devanagari script',
                'layer': 'english_transliteration'
            }
        
        # ----- Layer 3: Structure validation — reject invalid -----
        if not is_valid_devanagari_structure(clean):
            return {
                'word': word,
                'classification': 'incorrect',
                'confidence': 'high',
                'reason': 'Invalid Devanagari character sequence (structural violation)',
                'layer': 'structure_invalid'
            }
        
        # ----- Layer 4: Mixed content (numbers/punctuation attached) -----
        if is_word_with_punctuation_attached(clean):
            return {
                'word': word,
                'classification': 'incorrect',
                'confidence': 'high',
                'reason': 'Contains mixed content (numbers/punctuation attached to word)',
                'layer': 'mixed_content'
            }
        
        # ----- Layer 5: Contains non-Devanagari (Latin chars) -----
        has_latin = bool(re.search(r'[a-zA-Z]', clean))
        if has_latin:
            return {
                'word': word,
                'classification': 'incorrect',
                'confidence': 'high',
                'reason': 'Mixed script (contains Latin characters) — ASR/transcription artifact',
                'layer': 'mixed_script'
            }
        
        # ----- Layer 6: Multi-level morphological decomposition -----
        # Try suffix stripping (longest match first)
        for suffix in sorted(HINDI_SUFFIXES, key=len, reverse=True):
            if clean.endswith(suffix) and len(clean) > len(suffix) + 1:
                root = clean[:-len(suffix)]
                if root in self.vocab:
                    return {
                        'word': word,
                        'classification': 'correct',
                        'confidence': 'medium',
                        'reason': f'Morphological match: root "{root}" + suffix "{suffix}"',
                        'layer': 'morphology_suffix'
                    }
        
        # Try prefix stripping
        for prefix in sorted(HINDI_PREFIXES, key=len, reverse=True):
            if clean.startswith(prefix) and len(clean) > len(prefix) + 1:
                stem = clean[len(prefix):]
                if stem in self.vocab:
                    return {
                        'word': word,
                        'classification': 'correct',
                        'confidence': 'medium',
                        'reason': f'Morphological match: prefix "{prefix}" + stem "{stem}"',
                        'layer': 'morphology_prefix'
                    }
        
        # Try prefix + suffix combination
        for prefix in HINDI_PREFIXES:
            if clean.startswith(prefix):
                remainder = clean[len(prefix):]
                for suffix in HINDI_SUFFIXES:
                    if remainder.endswith(suffix) and len(remainder) > len(suffix) + 1:
                        core = remainder[:-len(suffix)]
                        if core in self.vocab:
                            return {
                                'word': word,
                                'classification': 'correct',
                                'confidence': 'medium',
                                'reason': f'Morphological match: "{prefix}" + "{core}" + "{suffix}"',
                                'layer': 'morphology_combined'
                            }
        
        # ----- Layer 7: English suffix pattern (transliteration heuristic) -----
        if is_english_transliteration(clean) and is_pure_devanagari(clean):
            return {
                'word': word,
                'classification': 'correct',
                'confidence': 'medium',
                'reason': 'Likely English transliteration (detected English suffix pattern)',
                'layer': 'english_pattern'
            }
        
        # ----- Layer 8: Number word check -----
        try:
            from src.number_normalizer import ALL_NUMBER_WORDS
            if clean in ALL_NUMBER_WORDS:
                return {
                    'word': word,
                    'classification': 'correct',
                    'confidence': 'high',
                    'reason': 'Hindi number word',
                    'layer': 'number'
                }
        except ImportError:
            pass
        
        # ----- Layer 9: Nukta variant tolerance -----
        if has_nukta_variant(clean):
            # Try without nukta
            denukta = clean.replace('\u093C', '')
            if denukta in self.vocab:
                return {
                    'word': word,
                    'classification': 'correct',
                    'confidence': 'medium',
                    'reason': f'Nukta variant of known word "{denukta}" (Urdu-influenced spelling)',
                    'layer': 'nukta_variant'
                }
        else:
            # Try adding nukta to ज, फ, क
            for base, nukta in [('ज', 'ज़'), ('फ', 'फ़'), ('क', 'क़'), ('ख', 'ख़'), ('ग', 'ग़')]:
                if base in clean:
                    variant = clean.replace(base, nukta)
                    if variant in self.vocab:
                        return {
                            'word': word,
                            'classification': 'correct',
                            'confidence': 'medium',
                            'reason': f'Non-nukta variant of "{variant}" (standard Hindi spelling)',
                            'layer': 'nukta_variant'
                        }
        
        # ----- Layer 10: Compound word analysis -----
        if is_compound_word(word):
            parts = re.split(r'[-_]', clean)
            parts_valid = sum(1 for p in parts if p and (p in self.vocab or strip_punctuation(p) in self.vocab))
            if parts_valid >= len(parts) * 0.5:
                return {
                    'word': word,
                    'classification': 'correct',
                    'confidence': 'medium',
                    'reason': f'Compound word — {parts_valid}/{len(parts)} parts recognized',
                    'layer': 'compound'
                }
        
        # ----- Layer 11: Single Devanagari character (valid particle) -----
        if len(clean) == 1 and any(c in ALL_DEVANAGARI for c in clean):
            return {
                'word': word,
                'classification': 'correct',
                'confidence': 'medium',
                'reason': 'Single Devanagari character (filler/particle)',
                'layer': 'single_char'
            }
        
        # ----- Layer 12: Pure Devanagari, reasonable length -----
        # Words that are pure Devanagari with 2-3 chars are likely valid short words/verb forms
        if is_pure_devanagari(clean) and len(clean) <= 3:
            return {
                'word': word,
                'classification': 'correct',
                'confidence': 'low',
                'reason': 'Short pure-Devanagari word (likely valid verb form or particle)',
                'layer': 'short_word'
            }
        
        # ----- Default: Classify based on script purity -----
        if is_pure_devanagari(clean):
            return {
                'word': word,
                'classification': 'incorrect',
                'confidence': 'low',
                'reason': 'Not found in dictionary — possibly misspelling, dialect, proper noun, or rare word',
                'layer': 'unknown_devanagari'
            }
        else:
            return {
                'word': word,
                'classification': 'incorrect',
                'confidence': 'medium',
                'reason': 'Contains non-standard characters — likely transcription artifact',
                'layer': 'unknown_nonstandard'
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
            'correct_pct': round(100.0 * correct / len(words), 2) if words else 0,
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
        'बहुत', 'कंप्यूटर', 'अच्छा', 'हम्म', 'इंरव्यू',   # intentional misspelling
        'स्कूल', 'पढ़ाई', 'बिजनेस', 'बच्चों', 'xyzहां',     # mixed script
        'स्कॉलरशिप', 'एग्जाक्ट्ली', 'गोशालाएं',             # real words from Q3
        'अपङी', '499में', 'हांपकवान',                          # actual misspellings
    ]
    
    print("Hindi Spelling Checker v2 Demo")
    print("=" * 60)
    for word in test_words:
        result = checker.check_word(word)
        status = "✓" if result['classification'] == 'correct' else "✗"
        print(f"{status} '{word}' → {result['classification']} [{result['confidence']}] "
              f"({result['layer']}: {result['reason']})")
