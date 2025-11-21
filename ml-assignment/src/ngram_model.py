import random
import re
from typing import Dict, Tuple, List, Optional


class TrigramModel:
    def __init__(self) -> None:
        # Mapping from (w1, w2) -> {w3: count}
        self._trigram_counts: Dict[Tuple[str, str], Dict[str, int]] = {}
        # Total number of times each (w1, w2) context was seen
        self._context_totals: Dict[Tuple[str, str], int] = {}
        # Vocabulary observed during training (excluding special tokens)
        self._vocab = set()

        # Special tokens for sentence boundaries / unknown words
        self._start_token = "<s>"
        self._end_token = "</s>"
        self._unk_token = "<unk>"

        self._is_trained: bool = False

    def _split_into_sentences(self, text: str) -> List[str]:
        # Normalize whitespace
        text = text.replace("\n", " ")
        # Split on ., ! or ?; keep only non-empty pieces
        parts = re.split(r"[.!?]+", text)
        return [p.strip() for p in parts if p.strip()]

    def _tokenize(self, sentence: str) -> List[str]:
        sentence = sentence.lower()
        # This finds sequences of alphanumeric characters as tokens.
        return re.findall(r"\b\w+\b", sentence)

    def _build_vocabulary(self, tokenized_sentences: List[List[str]]) -> None:
        vocab = set()
        for sent in tokenized_sentences:
            vocab.update(sent)
        self._vocab = vocab

    def _prepare_training_sequences(self, text: str) -> List[List[str]]:
        sentences = self._split_into_sentences(text)
        tokenized = [self._tokenize(s) for s in sentences]
        # Filter out completely empty sentences
        tokenized = [t for t in tokenized if t]

        if not tokenized:
            return []

        # Build vocabulary from the clean tokenized sentences
        self._build_vocabulary(tokenized)

        sequences: List[List[str]] = []
        for sent in tokenized:
            seq = [self._start_token, self._start_token]
            # In principle we could map rare / unknown tokens to <unk>.
            # Here everything from training is known, so we just append.
            seq.extend(sent)
            seq.append(self._end_token)
            sequences.append(seq)
        return sequences


    def fit(self, text: str) -> None:
        
        # Reset any previous state
        self._trigram_counts.clear()
        self._context_totals.clear()
        self._vocab.clear()
        self._is_trained = False

        if not text or not text.strip():
            # Nothing to learn from; keep the model in "untrained" state.
            return

        sequences = self._prepare_training_sequences(text)
        if not sequences:
            return

        for seq in sequences:
            # Iterate over triples (w1, w2, w3)
            for i in range(len(seq) - 2):
                w1, w2, w3 = seq[i], seq[i + 1], seq[i + 2]
                context = (w1, w2)

                # Update trigram counts
                if context not in self._trigram_counts:
                    self._trigram_counts[context] = {}
                self._trigram_counts[context][w3] = (
                    self._trigram_counts[context].get(w3, 0) + 1
                )

                # Update context totals
                self._context_totals[context] = self._context_totals.get(context, 0) + 1

        self._is_trained = bool(self._trigram_counts)

    def _sample_next(self, context: Tuple[str, str]) -> Optional[str]:
        
        options = self._trigram_counts.get(context)
        if not options:
            return None

        words = list(options.keys())
        counts = list(options.values())
        total = float(sum(counts))

        # Draw a random point in [0, 1) and walk the cumulative
        # probability mass until we cross it.
        r = random.random()
        cumulative = 0.0
        for word, count in zip(words, counts):
            cumulative += count / total
            if r <= cumulative:
                return word

        # Fallback (should rarely happen due to floating point)
        return words[-1]

    def generate(self, max_length: int = 50) -> str:
        
        if not self._is_trained:
            return ""

        # Start from two start tokens
        context = (self._start_token, self._start_token)
        generated_tokens: List[str] = []

        for _ in range(max_length):
            next_word = self._sample_next(context)
            if next_word is None:
                # We never saw this context; stop generation.
                break
            if next_word == self._end_token:
                # Reached end of sentence.
                break

            generated_tokens.append(next_word)
            # Slide the context window forward
            context = (context[1], next_word)

        # If nothing could be generated (very tiny or degenerate corpus),
        # return an empty string instead of raising an error.
        if not generated_tokens:
            return ""

        return " ".join(generated_tokens)
