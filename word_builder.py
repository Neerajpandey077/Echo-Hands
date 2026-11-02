"""
ECHO HANDS — Word Builder (Phase 1)
Accurately accumulates recognized letters into words with hold-to-confirm debouncing.
Supports all letters including 'E' and 'A' without conflicts.
"""

import time


class WordBuilder:
    def __init__(self, hold_seconds=0.7, grace_seconds=0.3):
        self.current_word = ""
        self.completed_words = []
        self.candidate_letter = ""
        self.letter_start_time = None
        self.last_seen_time = None
        self.committed = False
        self.hold_seconds = hold_seconds
        self.grace_seconds = grace_seconds
        self.progress = 0.0
        self.word = self.current_word

    def get_sentence(self):
        """Returns the sentence built from completed words and the current word."""
        parts = [word for word in self.completed_words if word]
        if self.current_word:
            parts.append(self.current_word)
        return " ".join(parts)

    def update(self, letter, fingers=None):
        """
        Updates the candidate letter tracking.
        When a letter is held steadily for `hold_seconds`, it is committed to `self.current_word`.
        """
        now = time.time()

        if letter:
            if letter != self.candidate_letter:
                # Switched to a new letter -> start fresh hold cycle
                self.candidate_letter = letter
                self.letter_start_time = now
                self.last_seen_time = now
                self.committed = False
                self.progress = 0.0
            else:
                # Same letter sustained
                self.last_seen_time = now
                if not self.committed and self.letter_start_time is not None:
                    elapsed = now - self.letter_start_time
                    self.progress = min(1.0, elapsed / self.hold_seconds)
                    if elapsed >= self.hold_seconds:
                        self.current_word += letter
                        self.word = self.current_word
                        self.committed = True
                        self.progress = 1.0

        else:
            # No letter recognized in this frame (hand relaxed or transition)
            if self.last_seen_time and (now - self.last_seen_time < self.grace_seconds):
                # Within grace window -> tolerate single frame drops
                pass
            else:
                # Grace period expired -> reset candidate
                self.candidate_letter = ""
                self.letter_start_time = None
                self.committed = False
                self.progress = 0.0

        return self.current_word

    def complete_word(self):
        """Moves the current word into the completed sentence buffer with a single space."""
        if not self.current_word.strip():
            return self.get_display_text()

        self.completed_words.append(self.current_word.strip())
        self.current_word = ""
        self.word = ""
        self.candidate_letter = ""
        self.letter_start_time = None
        self.committed = False
        self.progress = 0.0
        return self.get_display_text()

    def add_space(self):
        """Manual word separator. Keeps word boundaries explicit and user-controlled."""
        return self.complete_word()

    def undo_space(self):
        """
        Reverses the last add_space() — pops the last completed word back into
        current_word so the user can continue editing it.
        Returns: the restored word string.
        """
        if not self.completed_words:
            return self.current_word

        # If current word has partial content, discard it (user chose to undo the space)
        restored = self.completed_words.pop()
        self.current_word = restored
        self.word = self.current_word
        self.candidate_letter = ""
        self.letter_start_time = None
        self.committed = False
        self.progress = 0.0
        return self.current_word

    def get_display_text(self):
        """
        Returns the entire unified text typed so far as a single continuous string.
        If space was pressed, includes a trailing space so the next word continues smoothly.
        """
        if self.completed_words and self.current_word:
            return " ".join(self.completed_words) + " " + self.current_word
        elif self.completed_words:
            return " ".join(self.completed_words) + " "
        else:
            return self.current_word

    def backspace(self):
        """
        Deletes the last character. If current_word is empty but completed_words exist,
        removes the space and restores the last word for editing.
        """
        if self.current_word:
            self.current_word = self.current_word[:-1]
            self.word = self.current_word
        elif self.completed_words:
            self.current_word = self.completed_words.pop()
            self.word = self.current_word

        self.candidate_letter = ""
        self.letter_start_time = None
        self.committed = False
        self.progress = 0.0
        return self.get_display_text()

    def clear(self):
        """Clears the entire current word and completed sentence."""
        self.current_word = ""
        self.completed_words = []
        self.word = self.current_word
        self.candidate_letter = ""
        self.letter_start_time = None
        self.last_seen_time = None
        self.committed = False
        self.progress = 0.0
        return self.current_word
