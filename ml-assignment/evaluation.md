# Evaluation

##  Trigram Language Model – Design Explanation

This document briefly explains the reasoning and implementation strategy behind the trigram language model.

---

### 1.  Data Preprocessing

- **Sentence Splitting:**  
  Text is split using a regex that detects `.`, `!`, and `?`. This lightweight approach works well for general sentence boundaries without needing external libraries.

- **Tokenization:**  
  Tokens are extracted using `re.findall(r"\b\w+\b")` after converting text to lowercase. Only alphanumeric words are retained, ensuring consistency and simplicity.

- **Padding Strategy:**  
  Each sentence is wrapped as:  
  ```
  <s> <s> w1 w2 ... wn </s>
  ```
  Using two start tokens (`<s>`) ensures that the first word of the sentence always forms a complete trigram context. The end token (`</s>`) helps in deciding sentence termination during generation.

---

### 2.  Model Design

- **Data Structures:**  
  The model maintains:
  ```
  trigram_counts[(w1, w2)][w3] → number of occurrences
  context_totals[(w1, w2)] → total occurrences of context
  ```
  This enables efficient lookup of `P(w3 | w1, w2)`.

- **Vocabulary Handling:**  
  A `vocab` set is built during training. A placeholder `<unk>` token exists for future unknown words, though all tokens from training text are currently known.

---

### 3.  Training Process (`fit()`)

1. Clean and tokenize sentences.
2. Add padding.
3. For each `(w1, w2, w3)` trigram in a sequence:
   - Increment count in `trigram_counts`
   - Update `context_totals`
4. If no valid tokens are present, the model remains in "untrained" mode.

---

### 4.  Text Generation (`generate()`)

1. Start from `(<s>, <s>)`.
2. Retrieve possible `w3` candidates based on the trained model.
3. Sample one word using probability distribution based on counts:
   ```
   p(w3) = count(w3) / total count for context
   ```
4. Stop if:
   - The end token (`</s>`) is predicted
   - Maximum length is reached
   - Context is unseen

If no words are generated (e.g., very short input), return an empty string instead of raising an error.

---

### 5.  Edge Case Handling

| Case | Expected Output |
|------|-----------------|
| Empty input text | `""` |
| Very short input | Attempts sensible generation |
| No matching context | Stops gracefully |

These scenarios were unit tested and validated.

---

### 6.  Optional: Scaled Dot-Product Attention

A pure **NumPy implementation** of:
\[
\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
\]

**Key Features:**
- Supports batching via NumPy broadcasting
- Implements a mask option using `-1e9` trick before softmax
- No external ML libraries used
- Demo script prints output and attention weights

---

### 7.  Summary of Choices

| Component | Approach |
|----------|----------|
| Sentence splitting | Regex-based |
| Tokenization | Alphanumeric + lowercase |
| Padding | `<s> <s> ... </s>` |
| Data structure | Nested dictionary |
| Sampling | Probabilistic random |
| Unknown tokens | `<unk>` placeholder |
| Attention (optional) | NumPy implementation |

---

###  Final Remarks

- The model is clear, maintainable, and follows assignment constraints.
- Successfully passes provided test cases.
- Optional task completed without using ML frameworks.

---


