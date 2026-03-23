# Task Assignment | AI Researcher Intern- Speech & Audio | Josh Talks

## Question-1

### Background

You are provided with ~10 hours of Hindi ASR training data @FT Data - data.csv in the format shown below  
(audio + transcription metadata)

**Important Note:** The Url’s mentioned above in the the question and further questions might not work, PFB the instructions for modifying the same  

**Instructions to access the data :** this is the example of a new transcription URL  
<https://storage.googleapis.com/upload_goai/967179/825780_transcription.json>, the recording and metadata follows the same format, please modify the other URL's while processing the data  

### Dataset Schema Description

- user_id – Identifier for the speaker/user associated with the audio (anonymized).  
- recording_id – Unique identifier for the specific audio recording within the dataset.  
- language – Language label of the audio (e.g., "hi" for Hindi).  
- duration – Duration of the audio recording (in seconds). Useful for filtering or batching.  
- rec_url_gcp – URL link to the raw audio file stored on cloud (e.g., Google Cloud Storage). This is the main audio input for training/evaluation.  
- transcription_url – URL to the ground-truth transcription text corresponding to the audio file. This is the label to be used for fine-tuning.  
- metadata_url – URL to additional metadata about the recording (may include device type, noise level, accents, or collection conditions). Optional for training, but can help in analysis.  

### Your Task

a) Preprocess the dataset and share what you did to process the data and make it ready for training.  

b) Fine-tune Whisper-small on this dataset and evaluate both the pretrained Whisper-small baseline and your fine-tuned model on the Hindi portion of the FLEURS test dataset.  

c) Report the Word Error Rate (WER) in a structured table format. @FT Result - Sheet1.csv  

d) Systematically sample at least 25 utterances where your fine-tuned model still produces errors. Describe your sampling strategy (e.g., every Nth error, stratified by severity). Do not cherry-pick examples.  

e) Build an error taxonomy from what you observe. Categories should emerge from the data itself. For each category, provide 3–5 concrete examples showing: the reference transcript, your model's output, and your reasoning about the cause of the error.  

f) For your top 3 most frequent error types, propose a specific, actionable fix. Sometimes collecting more data is not sufficient.  

g) Implement at least one of your proposed fixes within the assignment timeframe. Show before/after results on a targeted subset of your error examples.  

---

## Question-2

Raw ASR output from Hindi conversations is messy, numbers come out as words, English words spoken in conversation are not always identified or handled correctly etc. Before ASR output is usable for any downstream task, it needs to be cleaned up. This question asks you to build a cleanup pipeline with two specific operations and carefully check where each one helps and where it makes things worse.  

Keep in mind our transcription guideline: English words spoken in the conversation are transcribed in Devanagari script. For example, "computer" spoken in English should appear as "कंप्यटू र." The Devanagari transcription counts as the correct spelling, not an error.  

### Data

Using the same ~10-hour dataset, generate raw ASR transcripts by running the pretrained whisper-small (before your Q1 fine-tuning) on the audio segments. Pair each raw ASR output with the human reference transcription from the dataset's JSON files.  

### Your Task

Build a pipeline that takes raw ASR output and performs the following operations:  

#### a) Number Normalization

Convert spoken Hindi number words into digits.  

- Simple cases: दो → 2, दस → 10, सौ → 100  
- Compound numbers: तीन सौ चौवन → 354, पच्चीस → 25, एक हज़ार → 1000  
- Edge cases: how do you handle numbers used in idioms or phrases where conversion would be wrong? (e.g., "दो-चार बातें" should probably stay as-is, not become "2-4 बातें")  

Provide 4-5 before/after examples from your actual data showing correct conversions, and 2-3 examples of tricky edge cases where you had to make a judgment call. Explain your reasoning for each edge case.  

#### b) English Word Detection

Identify which words in the Hindi transcript are actually English words spoken in the conversation.  

This is important because:  

- English words need different handling in downstream processing  
- They may need script normalization (Roman ↔ Devanagari)  
- They are common in real Hindi conversation ("मेरा interview अच्छा गया", "येproblem solve नहीं हो रहा")  

For each transcript, output a tagged version where English words are marked. For example:  

- Input: "मेरा इंटरव्यूबहुत अच्छा गया और मझु ेजॉब मि ल गई"  
- Output: "मेरा [EN]इंटरव्य[ू/EN] बहुत अच्छा गया और मझु े[EN]जॉब[/EN] मि ल गई"  

---

## Question-3

### Background

In a subset of our Hindi conversational dataset, which was human transcribed we have identified ~1,77,000 unique words @Unique Words Data - Sheet1.csv. Some of these words are obvious spelling mistakes.  

Our goal is to improve transcription accuracy of our dataset. One proposed approach is to separate these words into two groups:  

- Words that have 100% accurate spelling  
- Words that are incorrect because they contain spelling mistakes  

The idea is that once we identify the words with errors, we can go back to the corresponding audio segments and selectively re-human transcribe only those segments, rather than redoing the entire dataset.  

Keep in mind that in our transcription guidelines: English words spoken in the conversation are transcribed in Devanagari script. For example, “computer” spoken in English should appear as “कंप्यटू र.” In such cases, the Devanagari transcription counts as the correct spelling, not an error.  

### Your Task

a) Identify which of the 1,75,000 words are correctly spelled vs. incorrectly spelled. Share the approach you undertook to come to this conclusion.  

b) For every word your system classifies, also output a confidence score (high/medium/low) with a brief reason  

c) Review 40-50 words from your “low confidence” bucket. How many did your system get right vs. wrong? What does this tell you about where your approach breaks down?  

d) Identify at least 1-2 specific word categories where your system is unreliable and explain why.  

### Deliverables

a. Share the final number of unique correct spelled words in the dataset  
b. A google sheet containing 2 columns, one with the the final list of unique words and second making them as ‘correct spelling’ and ‘incorrect spelling’  

We are looking for your ability to combine linguistic reasoning with practical data-cleaning strategies that balance accuracy and efficiency.  

---

## Question - 4

In ASR evaluation, comparing model output against a single, rigid
Ground Truth string unfairly penalizes valid transcriptions. Speech often has multiple
correct written representations. A Lattice addresses this by replacing a flat string
with a sequential list of "bins." Each bin represents a specific alignment position and
contains all valid lexical, phonetical, and spelling variations for that point in the
audio. @Question 4 - Task.csv, you are given transcriptions from five ASR models for the same audio
and a human reference, which may contain errors

### Example

If the spoken audio is:  
"उसनेचौदह कि ताबेंखरीदीं" (He bought 14 books)  

A rigid reference transcript might just be:  
["उसने", "चौदह", "कि ताबें", "खरीदीं"]  

A lattice representation captures valid alternatives:  
[["उसने"], ["चौदह", "14"], ["कि ताबें", "कि ताबे", "पस्ुतकें"], ["खरीदीं", "खरीदी"]]  

### Your Task

Design an approach (theory + pseudocode/code) to:  

- Construct a lattice that captures all valid transcription alternatives from the model outputs.  
- Handle insertions, deletions, and substitutions in a way that does not unfairly penalize models when the reference is wrong.  
- Decide when to trust model agreement over the reference.  

Choose and justify the alignment unit (word / subword / phrase). Then compute WER for each model using lattice based transcription and model output.  

Your method should reduce WER for models that were unfairly penalized and keep it unchanged for the others.  
