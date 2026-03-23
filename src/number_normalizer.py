"""
Hindi Number Normalizer
========================
Converts spoken Hindi number words into digits.
Handles simple, compound, and edge cases (idioms, hyphenated pairs).
"""

import re
from typing import List, Tuple, Optional

# ============================================================================
# HINDI NUMBER WORD MAPPINGS
# ============================================================================

# Basic units
UNITS = {
    'शून्य': 0, 'एक': 1, 'दो': 2, 'तीन': 3, 'चार': 4,
    'पाँच': 5, 'पांच': 5, 'छह': 6, 'छः': 6, 'सात': 7,
    'आठ': 8, 'नौ': 9, 'नो': 9,
}

# 10-19 (each has a unique word in Hindi)
TEENS = {
    'दस': 10, 'ग्यारह': 11, 'बारह': 12, 'तेरह': 13, 'चौदह': 14,
    'पंद्रह': 15, 'सोलह': 16, 'सत्रह': 17, 'अठारह': 18, 'उन्नीस': 19,
}

# 20-99 (Hindi has unique words for each number 20-99)
TENS_UNIQUE = {
    'बीस': 20, 'इक्कीस': 21, 'बाईस': 22, 'तेईस': 23, 'चौबीस': 24,
    'पच्चीस': 25, 'छब्बीस': 26, 'सत्ताईस': 27, 'अट्ठाईस': 28, 'उनतीस': 29,
    'तीस': 30, 'इकतीस': 31, 'बत्तीस': 32, 'तैंतीस': 33, 'चौंतीस': 34,
    'पैंतीस': 35, 'छत्तीस': 36, 'सैंतीस': 37, 'अड़तीस': 38, 'उनतालीस': 39,
    'चालीस': 40, 'इकतालीस': 41, 'बयालीस': 42, 'तैंतालीस': 43, 'चौवालीस': 44,
    'पैंतालीस': 45, 'छियालीस': 46, 'सैंतालीस': 47, 'अड़तालीस': 48, 'उनचास': 49,
    'पचास': 50, 'इक्यावन': 51, 'बावन': 52, 'तिरपन': 53, 'चौवन': 54,
    'पचपन': 55, 'छप्पन': 56, 'सत्तावन': 57, 'अट्ठावन': 58, 'उनसठ': 59,
    'साठ': 60, 'इकसठ': 61, 'बासठ': 62, 'तिरसठ': 63, 'चौंसठ': 64,
    'पैंसठ': 65, 'छियासठ': 66, 'सड़सठ': 67, 'अड़सठ': 68, 'उनहत्तर': 69,
    'सत्तर': 70, 'इकहत्तर': 71, 'बहत्तर': 72, 'तिहत्तर': 73, 'चौहत्तर': 74,
    'पचहत्तर': 75, 'छिहत्तर': 76, 'सतहत्तर': 77, 'अठहत्तर': 78, 'उन्यासी': 79,
    'अस्सी': 80, 'इक्यासी': 81, 'बयासी': 82, 'तिरासी': 83, 'चौरासी': 84,
    'पचासी': 85, 'छियासी': 86, 'सतासी': 87, 'अट्ठासी': 88, 'नवासी': 89,
    'नब्बे': 90, 'इक्यानवे': 91, 'बानवे': 92, 'तिरानवे': 93, 'चौरानवे': 94,
    'पंचानवे': 95, 'छियानवे': 96, 'सत्तानवे': 97, 'अट्ठानवे': 98, 'निन्यानवे': 99,
}

# Powers / multipliers
POWERS = {
    'सौ': 100,
    'हज़ार': 1000, 'हजार': 1000, 'हज़ार': 1000,
    'लाख': 100000,
    'करोड़': 10000000, 'करोड': 10000000,
    'अरब': 1000000000, 'अरब': 1000000000,
}

# Combine all single-token number words
ALL_NUMBER_WORDS = {}
ALL_NUMBER_WORDS.update(UNITS)
ALL_NUMBER_WORDS.update(TEENS)
ALL_NUMBER_WORDS.update(TENS_UNIQUE)
ALL_NUMBER_WORDS.update(POWERS)

# Half values
HALF_WORDS = {'डेढ़': 1.5, 'ढाई': 2.5, 'साढ़े': 'prefix_half', 'सवा': 'prefix_quarter'}

# ============================================================================
# IDIOMATIC / NON-LITERAL NUMBER EXPRESSIONS
# ============================================================================

# Patterns where numbers are used idiomatically and should NOT be converted
IDIOMATIC_PATTERNS = [
    r'दो[-\s]चार\s+(बातें|बात|लोग|दिन|मिनट)',   # "दो-चार बातें" = a few things
    r'चार[-\s]पांच\s+(बातें|लोग|दिन)',             # "चार-पांच लोग" ≈ a few people (borderline)
    r'दो[-\s]दो\s+हाथ',                             # "दो-दो हाथ" = to fight/confrontation
    r'एक[-\s]दो\s+(बार|दिन|लोग|बातें)',            # "एक-दो बार" = once or twice (idiomatic)
    r'दो[-\s]टूक',                                   # "दो-टूक" = blunt/straight
    r'तीन[-\s]तेरह',                                 # "तीन-तेरह" = scattered
    r'चार[-\s]चाँद',                                 # "चार चाँद" = to add glory
    r'एक[-\s]एक',                                    # "एक-एक" = each one (distributive)
    r'नौ[-\s]दो[-\s]ग्यारह',                         # "नौ दो ग्यारह" = to flee
    r'सात[-\s]पांच',                                  # colloquial idioms
    r'दो[-\s]राय',                                    # "दो राय" = two opinions
    r'एक[-\s]तरफ',                                   # "एक तरफ" = on one side
    r'एक[-\s]साथ',                                   # "एक साथ" = together
    r'एक[-\s]दूसरे',                                 # "एक दूसरे" = each other
]


def is_idiomatic(text: str, start_idx: int, words: List[str]) -> bool:
    """
    Check if a number word at start_idx is part of an idiomatic expression.
    Uses a substring match around the number word position.
    """
    # Get context window (5 words around the number)
    context_start = max(0, start_idx - 2)
    context_end = min(len(words), start_idx + 5)
    context = ' '.join(words[context_start:context_end])
    
    for pattern in IDIOMATIC_PATTERNS:
        if re.search(pattern, context):
            return True
    return False


# ============================================================================
# COMPOUND NUMBER PARSER
# ============================================================================

def parse_number_value(word: str) -> Optional[int]:
    """Get numeric value of a single Hindi number word, or None if not a number."""
    word_clean = word.strip()
    if word_clean in ALL_NUMBER_WORDS:
        return ALL_NUMBER_WORDS[word_clean]
    return None


def parse_compound_number(words: List[str], start_idx: int) -> Tuple[Optional[int], int]:
    """
    Parse a compound Hindi number starting at start_idx.
    
    Returns (value, num_words_consumed) or (None, 0) if not a number.
    
    Examples:
        ["तीन", "सौ", "चौवन"] → (354, 3)
        ["एक", "हज़ार"] → (1000, 2)
        ["पच्चीस"] → (25, 1)
        ["दो", "लाख", "तीन", "हज़ार", "चार", "सौ", "पचास"] → (203450, 7)
    """
    if start_idx >= len(words):
        return None, 0
    
    first_val = parse_number_value(words[start_idx])
    if first_val is None:
        return None, 0
    
    # Collect consecutive number words
    total = 0
    current = first_val
    consumed = 1
    
    i = start_idx + 1
    while i < len(words):
        val = parse_number_value(words[i])
        if val is None:
            break
        
        if val in POWERS.values():
            # Multiplier: "तीन सौ" = 3 * 100
            if current == 0:
                current = 1  # "सौ" alone means 100
            current *= val
            consumed += 1
        elif val >= 100 and val in POWERS.values():
            # Another power → add current to total and start fresh
            total += current
            current = val
            consumed += 1
        else:
            # Additive: "तीन सौ चौवन" → 300 + 54
            # Check if this is an additive continuation
            if current > val:
                # Current is bigger → additive (300 + 54)
                total += current
                current = val
                consumed += 1
            else:
                # Not a continuation, stop
                break
        
        i += 1
    
    total += current
    return total, consumed


# ============================================================================
# MAIN NORMALIZER
# ============================================================================

class HindiNumberNormalizer:
    """
    Converts Hindi number words to digits in text.
    
    Handles:
    - Simple: दो → 2, दस → 10, सौ → 100
    - Compound: तीन सौ चौवन → 354, एक हज़ार → 1000
    - Edge cases: Preserves idiomatic expressions like "दो-चार बातें"
    """
    
    def __init__(self, preserve_idioms: bool = True):
        self.preserve_idioms = preserve_idioms
    
    def normalize(self, text: str) -> str:
        """
        Convert Hindi number words to digits in the given text.
        Preserves idiomatic expressions when preserve_idioms=True.
        """
        words = text.split()
        result = []
        i = 0
        
        while i < len(words):
            word_clean = words[i].strip()
            
            # Check if this is a number word
            if parse_number_value(word_clean) is not None:
                # Check for idiomatic usage
                if self.preserve_idioms and is_idiomatic(text, i, words):
                    result.append(words[i])
                    i += 1
                    continue
                
                # Try to parse compound number
                value, consumed = parse_compound_number(words, i)
                
                if value is not None:
                    result.append(str(value))
                    i += consumed
                else:
                    result.append(words[i])
                    i += 1
            else:
                result.append(words[i])
                i += 1
        
        return ' '.join(result)
    
    def normalize_with_annotations(self, text: str) -> List[dict]:
        """
        Like normalize(), but also returns annotations showing what was changed.
        Useful for generating before/after examples.
        """
        words = text.split()
        annotations = []
        result = []
        i = 0
        
        while i < len(words):
            word_clean = words[i].strip()
            
            if parse_number_value(word_clean) is not None:
                if self.preserve_idioms and is_idiomatic(text, i, words):
                    result.append(words[i])
                    annotations.append({
                        'original': words[i],
                        'converted': words[i],
                        'type': 'preserved_idiom'
                    })
                    i += 1
                    continue
                
                value, consumed = parse_compound_number(words, i)
                if value is not None:
                    original_words = ' '.join(words[i:i+consumed])
                    result.append(str(value))
                    annotations.append({
                        'original': original_words,
                        'converted': str(value),
                        'type': 'number_conversion'
                    })
                    i += consumed
                else:
                    result.append(words[i])
                    i += 1
            else:
                result.append(words[i])
                i += 1
        
        return {
            'original': text,
            'normalized': ' '.join(result),
            'annotations': [a for a in annotations if a.get('type') == 'number_conversion' or a.get('type') == 'preserved_idiom'],
        }


# ============================================================================
# DEMO / TEST
# ============================================================================

if __name__ == "__main__":
    normalizer = HindiNumberNormalizer()
    
    test_cases = [
        "मेरे पास दो किताबें हैं",
        "तीन सौ चौवन रुपए",
        "एक हज़ार पांच सौ",
        "पच्चीस लोग आए",
        "दो-चार बातें करनी हैं",
        "नौ दो ग्यारह हो गए",
        "उसने दस किताबें पढ़ीं",
        "सौ प्रतिशत सही",
    ]
    
    print("Hindi Number Normalization Examples")
    print("=" * 60)
    for text in test_cases:
        result = normalizer.normalize_with_annotations(text)
        print(f"\nInput:  {result['original']}")
        print(f"Output: {result['normalized']}")
        for ann in result['annotations']:
            print(f"  → [{ann['type']}] '{ann['original']}' → '{ann['converted']}'")
