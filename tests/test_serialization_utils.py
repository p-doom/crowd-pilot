#!/usr/bin/env python3
"""
Synthetic tests for `serialization_utils.py`.

These tests stress `_apply_change` and `_compute_changed_block_lines`
to ensure that file state remains consistent even after many edits
insert / replace / delete) and that the diff computation can be used
to reconstruct the `after` state from the `before` state.
"""

from __future__ import annotations

import random
import string
import unittest

from crowd_pilot.serialization_utils import _apply_change, _compute_changed_block_lines


def _random_text(rng: random.Random, max_len: int = 40) -> str:
    """Generate random text including spaces and newlines."""
    alphabet = string.ascii_letters + string.digits + " \t"
    # Bias towards shorter snippets but occasionally longer ones
    length = rng.randint(0, max_len)
    # Sprinkle in a few explicit '\n' and real newlines
    chars = []
    for _ in range(length):
        if rng.random() < 0.05:
            # Literal backslash-n, as it would appear in CSV
            chars.append("\\n")
        elif rng.random() < 0.05:
            # Actual newline character
            chars.append("\n")
        else:
            chars.append(rng.choice(alphabet))
    return "".join(chars)


class ApplyChangeRandomizedTest(unittest.TestCase):
    """Randomized tests for `_apply_change` to check state consistency."""

    def _naive_apply(self, base: str, offset: int, length: int, new_text: str) -> str:
        """Naive implementation mirroring `_apply_change` semantics."""
        text = "" if new_text is None else str(new_text)
        # Match the newline normalization used in `_apply_change`
        text = text.replace("\\n", "\n").replace("\\r", "\r")
        if offset > len(base):
            base = base + (" " * (offset - len(base)))
        return base[:offset] + text + base[offset + length :]

    def test_apply_change_random_sequences(self) -> None:
        """
        Apply hundreds of random edits (insert/replace/delete) and
        check that `_apply_change` matches the naive reference.
        """
        rng = random.Random(12345)

        for _ in range(50):  # 50 independent runs
            base_expected = _random_text(rng, max_len=200)
            base_actual = base_expected

            for _ in range(300):  # up to 300 edits per run
                # Choose an offset in the current string
                if base_expected:
                    offset = rng.randint(0, len(base_expected))
                else:
                    offset = 0

                # Randomly choose an operation type
                op = rng.choice(["insert", "delete", "replace"])

                if op == "insert":
                    length = 0
                    new_text = _random_text(rng, max_len=40)
                elif op == "delete":
                    if len(base_expected) == 0:
                        continue
                    max_len = len(base_expected) - offset
                    length = rng.randint(0, max_len)
                    new_text = ""
                else:  # replace
                    if len(base_expected) == 0:
                        length = 0
                    else:
                        max_len = len(base_expected) - offset
                        length = rng.randint(0, max_len)
                    new_text = _random_text(rng, max_len=40)

                base_expected = self._naive_apply(base_expected, offset, length, new_text)
                base_actual = _apply_change(base_actual, offset, length, new_text)

                self.assertEqual(
                    base_actual.rstrip("\n"),
                    base_expected.rstrip("\n"),
                    msg=f"Mismatch after op={op}, offset={offset}, length={length}",
                )


class ComputeChangedBlockLinesTest(unittest.TestCase):
    """Tests for `_compute_changed_block_lines` using synthetic before/after pairs."""

    def _apply_block(
        self,
        before: str,
        start_before: int,
        end_before: int,
        repl_lines: list[str],
    ) -> str:
        """Apply a contiguous block replacement to `before`."""
        before_lines = before.splitlines()
        # Convert 1-based inclusive indices to Python slices
        start_idx = max(0, start_before - 1)
        end_idx = min(len(before_lines), end_before)
        new_lines = before_lines[:start_idx] + repl_lines + before_lines[end_idx:]
        return "\n".join(new_lines)

    def test_changed_block_round_trip_single_region(self) -> None:
        """
        Construct `after` from `before` via a single contiguous region
        replacement and ensure `_compute_changed_block_lines` returns a
        patch that reproduces `after`.
        """
        rng = random.Random(67890)

        for _ in range(100):  # 100 random before/after pairs
            # Build a non-empty "before" text with multiple lines
            num_lines = rng.randint(3, 40)
            before_lines = [
                _random_text(rng, max_len=40) or f"line-{i}" for i in range(num_lines)
            ]
            before = "\n".join(before_lines)

            # Choose a single contiguous region to replace
            start_before = rng.randint(1, num_lines)
            end_before = rng.randint(start_before, num_lines)

            # Replacement may be empty (pure deletion) or some new lines
            repl_count = rng.randint(0, 10)
            repl_lines = [
                _random_text(rng, max_len=40) or f"repl-{i}" for i in range(repl_count)
            ]

            after_lines = (
                before_lines[: start_before - 1] + repl_lines + before_lines[end_before:]
            )
            after = "\n".join(after_lines)

            (
                s_before,
                e_before,
                _s_after,
                _e_after,
                diff_repl_lines,
            ) = _compute_changed_block_lines(before, after)

            patched = self._apply_block(before, s_before, e_before, diff_repl_lines)

            self.assertEqual(
                patched.rstrip("\n"),
                after.rstrip("\n"),
                msg=(
                    f"Failed to reconstruct `after` from `before` using "
                    f"computed block ({s_before},{e_before})"
                ),
            )


if __name__ == "__main__":
    unittest.main()


