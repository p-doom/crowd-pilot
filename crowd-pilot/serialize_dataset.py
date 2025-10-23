#!/usr/bin/env python3
"""
CSV sessions -> Parquet shards for MaxText Grain pretraining.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, cast, Dict

import pandas as pd  # type: ignore
from datasets import Dataset, load_dataset  # type: ignore



@dataclass
class SerializeConfig:
    output_dir: str
    shard_size: int
    target_chars: int
    min_doc_chars: int
    max_docs: Optional[int]
    long_pause_threshold_ms: int
    csv_root: Optional[str]


def _clean_text(text: str) -> str:
    # Normalize line endings and strip trailing spaces; preserve tabs/newlines.
    return text.replace("\r\n", "\n").replace("\r", "\n").rstrip()


def _fenced_block(path: str, language: Optional[str], content: str) -> str:
    lang = (language or "").lower()
    return f"```{lang}\n{content}\n```\n"


def _apply_change(content: str, offset: int, length: int, new_text: str) -> str:
    # Mirrors crowd_code_player.replay_file.apply_change
    base = str(content)
    text = str(new_text) if pd.notna(new_text) else ""
    text = text.replace("\\n", "\n").replace("\\r", "\r")
    if offset > len(base):
        base = base + (" " * (offset - len(base)))
    return base[:offset] + text + base[offset + length:]




def _session_to_parts(
    df: pd.DataFrame,
    long_pause_threshold_ms: int,
) -> List[str]:
    file_states: Dict[str, str] = {}
    terminal_state: str = ""
    per_file_event_counts: Dict[str, int] = {}
    per_file_cursor_positions: Dict[str, Tuple[int, int]] = {}  # (offset, length) for each file
    last_time_ms: Optional[int] = None

    parts: List[str] = []

    for i in range(len(df)):
        row = df.iloc[i]
        file_path: str = row["File"]
        event_time: int = row["Time"]
        language: Optional[str] = row["Language"]

        # Long pause detection
        if last_time_ms is not None:
            delta = event_time - last_time_ms
            if delta > long_pause_threshold_ms:
                # TODO (f.srambical): think about whether we want to emit this as an observation or not
                parts.append(f"<obs long_pause ms=\"{delta}\" />")
        last_time_ms = event_time

        event_type = row["Type"]

        match event_type:
            case "tab":
                # File switch event
                parts.append(f"<act focus file=\"{file_path}\" />")
                
                # If Text is present, this is the first time opening the file
                # and the entire file content is captured
                text = row["Text"]
                if pd.notna(text):
                    file_content = str(text).replace("\\n", "\n").replace("\\r", "\r")
                    file_states[file_path] = file_content
                    parts.append(f"// observation: file={file_path}")
                    parts.append(_fenced_block(file_path, language, _clean_text(file_content)))

            case "terminal_command":
                # Terminal command execution
                command = row["Text"]
                command_str = str(command).replace("\\n", "\n").replace("\\r", "\r")
                parts.append(f"<act terminal_command />")
                parts.append(_fenced_block(file_path, "bash", _clean_text(command_str)))

            case "terminal_output":
                # Terminal output capture
                output = row["Text"]
                output_str = str(output).replace("\\n", "\n").replace("\\r", "\r")
                parts.append(f"<obs terminal_output />")
                parts.append(_fenced_block(file_path, None, _clean_text(output_str)))

            case "terminal_focus":
                # Terminal focus event
                parts.append(f"<act focus target=\"terminal\" />")

            case "git_branch_checkout":
                # Git branch checkout event
                branch_info = row["Text"]
                branch_str = str(branch_info).replace("\\n", "\n").replace("\\r", "\r")
                parts.append(f"<act git_branch_checkout />")
                parts.append(f"// git: {_clean_text(branch_str)}")

            case "selection_command" | "selection_mouse" | "selection_keyboard":
                # Handle cursor movement
                offset = row["RangeOffset"]
                length = row["RangeLength"]
                old_cursor = per_file_cursor_positions.get(file_path, (0, 0))
                new_cursor = (offset, length)
                per_file_cursor_positions[file_path] = new_cursor
                
                # Emit cursor movement observation if position changed
                if old_cursor != new_cursor:
                    parts.append(f"<act cursor file=\"{file_path}\" offset=\"{offset}\" len=\"{length}\" />")

            case "content":
                # Handle file edit events
                offset = row["RangeOffset"]
                length = row["RangeLength"]
                new_text = row["Text"]
                new_text_str = str(new_text) if pd.notna(new_text) else ""

                operation = "noop"
                if length == 0 and new_text_str:
                    operation = "insert"
                elif length > 0 and not new_text_str:
                    operation = "delete"
                elif length > 0 and new_text_str:
                    operation = "replace"

                parts.append(f"<act {operation} file=\"{file_path}\" offset=\"{offset}\" len=\"{length}\" />")

                if new_text_str and (operation == "insert" or operation == "replace"):
                    parts.append(_fenced_block(file_path, language, _clean_text(new_text_str)))

                before = file_states.get(file_path, "")
                after = _apply_change(before, offset, length, new_text)
                file_states[file_path] = after
                per_file_event_counts[file_path] = per_file_event_counts.get(file_path, 0) + 1

                # Update cursor position after edit (cursor moves to end of inserted/replaced text)
                per_file_cursor_positions[file_path] = (offset + len(new_text_str), 0)

            case _:
                raise ValueError(f"Unknown event type: {event_type}")

    return parts


def _chunk_parts(parts: List[str], target_chars: int, min_doc_chars: int) -> List[str]:
    """Chunk by aggregating whole parts without splitting parts.

    Each chunk is a concatenation of complete parts whose total length is as
    close as possible to target_chars without exceeding it (except when a
    single part alone exceeds target_chars, in which case it forms its own
    chunk). No overlap is introduced between consecutive chunks.
    """
    if target_chars <= 0:
        joined = "\n".join(parts).strip()
        return [joined] if len(joined) >= min_doc_chars else []

    chunks_out: List[str] = []
    current: List[str] = []
    current_len = 0

    def flush_current():
        nonlocal current, current_len
        if not current:
            return
        joined = "\n".join(current).strip()
        if len(joined) >= min_doc_chars:
            chunks_out.append(joined)
        current = []
        current_len = 0

    for p in parts:
        p_len = len(p) + 1  # account for newline joining
        if current_len + p_len <= target_chars or not current:
            current.append(p)
            current_len += p_len
        else:
            flush_current()
            # If the part itself exceeds target, still include it to preserve atomicity
            current.append(p)
            current_len = p_len
    flush_current()
    return chunks_out

 
def load_hf_csv(hf_path: str, split: str) -> Dataset:
    loaded = load_dataset(hf_path, split=split)

    assert isinstance(loaded, Dataset), "Expected a Dataset from load_dataset"
    return loaded


def _discover_local_sessions(root: Path) -> List[Path]:
    # Recursively find all CSV files
    paths: List[Path] = []
    for p in root.rglob("*.csv"):
        if p.is_file():
            paths.append(p)
    paths.sort()
    return paths


def to_parquet(
    cfg: SerializeConfig,
) -> None:
    os.makedirs(cfg.output_dir, exist_ok=True)

    required_cols = ["Sequence", "Time", "File", "RangeOffset", "RangeLength", "Text", "Language", "Type"]

    # Build a list of per-session DataFrames
    session_dataframes: List[pd.DataFrame] = []
    root = Path(cast(str, cfg.csv_root)).expanduser().resolve()
    csv_files = _discover_local_sessions(root)
    assert csv_files, f"No CSV files found under {root}"
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        missing_local = [c for c in required_cols if c not in df.columns]
        assert not missing_local, f"Missing required CSV columns in {csv_file}: {missing_local}"
        session_dataframes.append(df)

    rows_out = 0
    shard_idx = 0
    buffer: List[str] = []
    docs_written = 0

    def flush_buffer(idx: int):
        nonlocal buffer, rows_out
        if not buffer:
            return
        df_out = pd.DataFrame({"text": buffer})
        out_path = Path(cfg.output_dir) / f"shard_{idx:05d}.parquet"
        table = df_out
        table.to_parquet(out_path, index=False)
        rows_out += len(df_out)
        buffer = []

    for session_df in session_dataframes:
        session_df = pd.DataFrame(session_df.copy())
        parts = _session_to_parts(
            session_df,
            long_pause_threshold_ms=cfg.long_pause_threshold_ms,
        )
        chunks = _chunk_parts(parts, cfg.target_chars, cfg.min_doc_chars)
        if not chunks:
            continue
        buffer.extend(chunks)
        docs_written += len(chunks)
        if len(buffer) >= cfg.shard_size:
            flush_buffer(shard_idx)
            shard_idx += 1
        if cfg.max_docs and docs_written >= cfg.max_docs:
            break

    if buffer:
        flush_buffer(shard_idx)

    print(f"Wrote {rows_out} documents to {cfg.output_dir}")


def parse_args() -> SerializeConfig:
    p = argparse.ArgumentParser(description="Serialize HF CSV sessions to Parquet for MaxText Grain")
    p.add_argument("--csv_root", type=str, required=True, help="Root directory containing per-session CSV files")
    p.add_argument("--output_dir", type=str, required=True, help="Output directory for Parquet shards")
    p.add_argument("--shard_size", type=int, default=20000, help="Rows per Parquet shard")
    # FIXME(f.srambical): It is awkward that the target number is in character-space instead of in token-space.
    p.add_argument("--target_chars", type=int, default=8192, help="Target characters per document chunk. This should be ~3-4x the max token length of the model you are using.")
    p.add_argument("--min_doc_chars", type=int, default=1024, help="Minimum characters to keep a chunk")
    p.add_argument("--max_docs", type=int, default=None, help="Stop after writing this many unique docs")
    p.add_argument("--long_pause_threshold_ms", type=int, default=120000, help="Threshold (ms) to annotate long pauses and emit a keyframe")
    args = p.parse_args()
    return SerializeConfig(
        output_dir=args.output_dir,
        shard_size=args.shard_size,
        target_chars=args.target_chars,
        min_doc_chars=args.min_doc_chars,
        max_docs=args.max_docs,
        long_pause_threshold_ms=args.long_pause_threshold_ms,
        csv_root=(args.csv_root if args.csv_root else None),
    )


def main() -> None:
    cfg = parse_args()
    to_parquet(cfg)


if __name__ == "__main__":
    main()


