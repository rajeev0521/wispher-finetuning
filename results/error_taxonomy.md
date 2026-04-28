# Q1: Error Taxonomy & Fix Proposals

## Error Sampling Strategy (Q1d)

**Method:** Stratified sampling by WER severity across 5 strata (Very High >0.8, High 0.5–0.8, Medium 0.3–0.5, Low 0.1–0.3, Very Low ≤0.1). We sampled 5 from each stratum (or all available if <5). Total = **25 error utterances** systematically sampled.

---

## Error Taxonomy (Q1e)

Categories emerged from examining the 25 sampled error utterances from fine-tuned Whisper-Small on FLEURS Hindi.

---

### Category 1: Repetition / Hallucination Loops (Most Frequent)

The model gets stuck repeating a single word/syllable indefinitely. This is a known Whisper failure mode caused by attention collapse.

| # | Reference | Model Output | Reasoning |
|---|-----------|-------------|-----------|
| 1 | यू.एस. कॉर्प्स ऑफ़ इंजीनियर्स ने अंदाजा लगाया कि 6 इंच बारिश... | जी जी जी जी जी जी जी जी जी जी जी जी... | Attention collapsed into a single token loop. The English abbreviation "U.S." likely triggered confusion. |
| 2 | सावधान रहें कि कपड़े को बहुत गर्म न होने दें... | सावधान रही कि कपड़े को बहुत गरम ना होनी दे चलो चलो चलो चलो... | Started correctly, but after deviation ("होनी दे" vs "होने दें"), collapsed into "चलो" repetition. |
| 3 | जिसके परिणामस्वरूप मंच पर कलाकार... | जिसके परढ़ाजाजाजाजा... | Complex compound word "परिणामस्वरूप" triggered an early loop of "जा" syllable repetition. |

**Frequency:** ~5/25 samples (20%). The most damaging error type — causes WER >1.0.

---

### Category 2: Phonetic Substitution (Homophone/Near-Homophone Confusion)

Similar-sounding Hindi words are swapped. The model recognizes the phoneme correctly but maps it to the wrong lexical item.

| # | Reference | Model Output | Reasoning |
|---|-----------|-------------|-----------|
| 1 | क्षेत्रों → शेत्रों | "क्ष" cluster simplified to "श" — phonetically close in casual speech |
| 2 | शताब्दी → सतावती | "शत" → "सत" (aspiration lost), "ब्दी" → "वती" (similar vowel pattern) |
| 3 | बकवास → बबास | Geminate confusion: "कव" → "ब" (labial merger) |
| 4 | शिकारी → सिकारी | "श" → "स" — classic dental vs. palatal sibilant confusion |
| 5 | मूर्खतापूर्ण → मुझव तपना | Complex Sanskritized compound completely decomposed into simpler phonetic matches |

**Frequency:** ~8/25 samples (32%). The most frequent category overall.

---

### Category 3: Word Boundary Errors (Merging / Splitting)

Words incorrectly merged (two words → one) or split (one word → two).

| # | Reference | Model Output | Reasoning |
|---|-----------|-------------|-----------|
| 1 | सुपर जी → सुपरजी | Two words merged into one compound — contextually reasonable but differs from reference |
| 2 | बचाव दल → बचावदल | Same merging pattern; "बचाव दल" (rescue team) fused |
| 3 | साथ क्रिया → साथक्रिया | Space dropped at word boundary |
| 4 | उपोष्णकटिबंधीय → उपोष्ण न कटिवंदीद | Long compound word incorrectly split and mutated |
| 5 | नाकाबंदी → नाकावंदिश | Compound distorted — "बं" → "वं", ending changed |

**Frequency:** ~5/25 samples (20%).

---

### Category 4: Rare / Proper Noun and Foreign Word Errors

Names, technical terms, and borrowed words not in training vocabulary get mangled.

| # | Reference | Model Output | Reasoning |
|---|-----------|-------------|-----------|
| 1 | शेंगेन ज़ोन → शिंगें जॉन | Proper noun "Schengen" phonetically adapted differently |
| 2 | ओल्डरिच जेलिनेक → ऑलडरिच जैलिमनेक | Czech name garbled — model has no prior for this name |
| 3 | स्फिंक्स → स्फिंग्स | Rare English loanword — final cluster "क्स" → "ग्स" |
| 4 | नेक्रोपोलिस → नेक्रोपोलस | Greek-origin word — vowel dropped |

**Frequency:** ~4/25 samples (16%).

---

### Category 5: Inflection / Grammatical Errors

Correct root word but wrong verb conjugation, gender marking, or case ending.

| # | Reference | Model Output | Reasoning |
|---|-----------|-------------|-----------|
| 1 | रहें → रही | Subjunctive → past tense feminine |
| 2 | महत्वपूर्ण → महत्वपोर्ण | Vowel in suffix distorted — "पूर्ण" → "पोर्ण" |
| 3 | पूरे → पूरी | Masculine plural → feminine singular gender mismatch |

**Frequency:** ~3/25 samples (12%).

---

### Category 6: Number / Date Expression Errors

Numerals, dates, and quantities are misrecognized or output in wrong format.

| # | Reference | Model Output | Reasoning |
|---|-----------|-------------|-----------|
| 1 | 11 35 बजे → गेरा पैसालीस बजे | Digits in reference, model outputs Hindi words but incorrectly ("11:35" → garbled) |
| 2 | 1963 → उन्नीस सहत्यूत सट | Year in digits, model attempted Hindi number words but badly mangled |
| 3 | 2009 → दो हजार नौ | Digits → correct Hindi words — actually valid but counts as error since reference uses digits |

**Frequency:** ~2/25 samples (8%).

---

## Top-3 Fix Proposals (Q1f)

### Fix 1: Repetition Suppression (for Hallucination Loops — 20% of errors)

**Problem:** Whisper's autoregressive decoder gets stuck in attention loops, repeating tokens.

**Actionable Fix:**
- Enable **repetition penalty** during generation (`repetition_penalty=1.2–1.5` in `model.generate()`)
- Add a **no-repeat n-gram** constraint (`no_repeat_ngram_size=3`) to prevent any 3-gram from appearing twice
- Implement **length penalty** (`length_penalty=1.0`) to discourage outputs that are much longer than expected for the input audio duration
- Post-processing: detect runs of 3+ identical consecutive words and truncate

**Expected Impact:** Would eliminate ~100% of Type 1 errors (hallucination loops), reducing overall WER by ~5–10%.

---

### Fix 2: Text Normalization Alignment (for Number/Format Mismatches — 8% of errors)

**Problem:** Reference uses digits ("1963") while model outputs Hindi words ("उन्नीस सौ तिरसठ") or vice versa. Both are valid but WER punishes the mismatch.

**Actionable Fix:**
- Apply **number normalization** to both reference and hypothesis before WER computation
- Convert all Hindi number words → digits (or all digits → Hindi words) consistently
- Normalize "ज़" ↔ "ज", "फ़" ↔ "फ" (nukta variants) before comparison

**Expected Impact:** Would eliminate false-positive errors from format mismatches. Est. 2–3% WER improvement.

---

### Fix 3: Data Augmentation with FLEURS-style Formal Hindi (for Phonetic Substitution — 32% of errors)

**Problem:** The training data is conversational Hindi but FLEURS test data is formal/literary Hindi with Sanskritized vocabulary (e.g., "क्षतिग्रस्त", "उपोष्णकटिबंधीय") that the model hasn't seen.

**Actionable Fix:**
- **Mix training data**: Add Hindi Common Voice or FLEURS train split to the fine-tuning data (multi-domain training)
- **Text augmentation**: Generate phonetic variants of rare words in training transcriptions to expose the model to more vocabulary
- **Curriculum learning**: Train first on conversational data (current dataset), then fine-tune further on formal Hindi data

**Expected Impact:** Would reduce phonetic substitution errors on formal vocabulary by ~30–50%, est. 5–8% WER improvement on FLEURS.
