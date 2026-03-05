"""
Chapter 4: Sentence Buffer

Accumulates streaming LLM tokens and yields complete sentences.
This is the bridge between the LLM (which generates tokens) and
TTS (which needs complete sentences for natural speech).

The sentence buffer handles:
  - Detecting sentence boundaries (.!?)
  - Avoiding false positives (Mr., Dr., 3.14, U.S.A.)
  - Flushing remaining text when the stream ends
  - Minimum chunk length (don't send 1-word sentences to TTS)

Usage:
    python chapters/04_streaming_pipeline/sentence_buffer.py
"""

import re


class SentenceBuffer:
    """
    Accumulates streaming text tokens and yields complete sentences.

    Usage:
        buffer = SentenceBuffer()
        for token in llm_stream:
            sentences = buffer.add(token)
            for sentence in sentences:
                # Send to TTS
                tts.synthesize(sentence)
        # Flush remaining
        final = buffer.flush()
        if final:
            tts.synthesize(final)
    """

    # Common abbreviations that end with periods but aren't sentence endings
    ABBREVIATIONS = {
        "mr", "mrs", "ms", "dr", "prof", "sr", "jr",
        "st", "ave", "blvd", "rd",
        "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
        "inc", "ltd", "corp", "co", "dept",
        "vs", "etc", "approx", "appt",
        "e.g", "i.e", "a.m", "p.m",
    }

    def __init__(self, min_length: int = 10):
        """
        Args:
            min_length: Minimum character count before yielding a sentence.
                       Prevents very short fragments (e.g., "Hi." alone).
        """
        self.buffer = ""
        self.min_length = min_length

    def add(self, token: str) -> list[str]:
        """
        Add a token to the buffer and return any complete sentences.

        Args:
            token: A text token from the LLM stream

        Returns:
            List of complete sentences (may be empty)
        """
        self.buffer += token
        return self._extract_sentences()

    def flush(self) -> str | None:
        """Flush any remaining text in the buffer. Call when stream ends."""
        remaining = self.buffer.strip()
        self.buffer = ""
        return remaining if remaining else None

    def _extract_sentences(self) -> list[str]:
        """Extract complete sentences from the buffer."""
        sentences = []

        while True:
            # Find potential sentence boundary
            match = re.search(r'[.!?][\s"]', self.buffer)
            if not match:
                # Also check for sentence-ending punctuation at end of buffer
                # (followed by nothing yet — wait for next token to confirm)
                break

            boundary_pos = match.start() + 1  # Include the punctuation
            candidate = self.buffer[:boundary_pos].strip()

            # Check if this is a real sentence boundary
            if self._is_sentence_boundary(candidate):
                if len(candidate) >= self.min_length:
                    sentences.append(candidate)
                    self.buffer = self.buffer[boundary_pos:].lstrip()
                else:
                    # Too short — keep accumulating
                    break
            else:
                # False positive (abbreviation, number) — skip past it
                # Move past this period and continue searching
                break

        return sentences

    def _is_sentence_boundary(self, text: str) -> bool:
        """Check if the text ending represents a real sentence boundary."""
        if not text:
            return False

        # Get the last word before the punctuation
        words = text.rstrip(".!?").rsplit(None, 1)
        if not words:
            return True

        last_word = words[-1].lower().rstrip(".")

        # Check against abbreviations
        if last_word in self.ABBREVIATIONS:
            return False

        # Check for decimal numbers (e.g., "3.14")
        if re.search(r'\d\.$', text.rstrip()):
            return False

        return True

    def reset(self):
        """Clear the buffer."""
        self.buffer = ""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_sentence_buffer():
    """Test the sentence buffer with various inputs."""
    print("=" * 60)
    print("Sentence Buffer Tests")
    print("=" * 60)

    tests = [
        (
            "Simple sentences",
            ["Hello, ", "how are ", "you? ", "I'm ", "doing ", "great. ", "Thanks ", "for asking!"],
            ["Hello, how are you?", "I'm doing great."],  # "Thanks for asking!" flushed at end
        ),
        (
            "Abbreviations (Dr., Mr.)",
            ["Please see ", "Dr. ", "Johnson at ", "3 PM. ", "He's expecting ", "you."],
            ["Please see Dr. Johnson at 3 PM."],
        ),
        (
            "Numbers with decimals",
            ["The temperature ", "is 98.6 ", "degrees. ", "That's normal."],
            ["The temperature is 98.6 degrees."],
        ),
        (
            "Exclamation and question",
            ["That's amazing! ", "Don't you ", "think so? ", "I certainly ", "do."],
            ["That's amazing!", "Don't you think so?"],
        ),
    ]

    all_passed = True
    for name, tokens, expected_sentences in tests:
        buffer = SentenceBuffer(min_length=10)
        all_sentences = []

        for token in tokens:
            sentences = buffer.add(token)
            all_sentences.extend(sentences)

        remaining = buffer.flush()
        if remaining:
            all_sentences.append(remaining)

        # Check just the non-flushed sentences
        extracted = all_sentences[: len(expected_sentences)]
        passed = extracted == expected_sentences

        status = "PASS" if passed else "FAIL"
        print(f"\n  [{status}] {name}")
        print(f"    Tokens: {tokens}")
        print(f"    Expected: {expected_sentences}")
        print(f"    Got:      {extracted}")
        if remaining:
            print(f"    Flushed:  '{remaining}'")

        if not passed:
            all_passed = False

    print(f"\n  {'All tests passed!' if all_passed else 'Some tests failed.'}")
    return all_passed


def demo_streaming_simulation():
    """Simulate LLM streaming and show sentence detection timing."""
    import time

    print("\n" + "=" * 60)
    print("Streaming Simulation")
    print("=" * 60)

    # Simulate LLM generating tokens one at a time
    text = (
        "Hello! I've checked Dr. Johnson's availability for you. "
        "He has openings on Tuesday at 10 AM and 3 PM. "
        "Would you prefer the morning or afternoon slot? "
        "Both are 30-minute appointments."
    )

    # Split into "tokens" (words)
    tokens = []
    for word in text.split(" "):
        tokens.append(word + " ")

    buffer = SentenceBuffer(min_length=10)
    print(f"\n  Simulating {len(tokens)} tokens:\n")

    t_start = time.time()
    for i, token in enumerate(tokens):
        # Simulate ~30ms inter-token delay
        time.sleep(0.03)

        sentences = buffer.add(token)
        for sentence in sentences:
            elapsed = (time.time() - t_start) * 1000
            print(f"  [{elapsed:>6.0f}ms] → SENTENCE: {sentence}")
            print(f"           (would send to TTS now)")

    remaining = buffer.flush()
    if remaining:
        elapsed = (time.time() - t_start) * 1000
        print(f"  [{elapsed:>6.0f}ms] → FLUSH: {remaining}")

    print(f"\n  Each sentence would start TTS synthesis immediately,")
    print(f"  overlapping with continued LLM generation.")


if __name__ == "__main__":
    test_sentence_buffer()
    demo_streaming_simulation()
