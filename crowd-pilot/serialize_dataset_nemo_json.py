#!/usr/bin/env python3
"""
CSV sessions -> JSONL (newline-delimited JSON) for NeMo 2.0 SFT function calling.

Each session is converted to a multi-turn conversation where:
- User role: terminal output (<stdout>...</stdout> content)
- Assistant role: terminal commands (bash commands)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple, cast
import random

import pandas as pd

from serialization_utils import (
    SerializeConfig,
    session_to_nemo_conversation_chunks,
    _discover_local_sessions,
)


def to_nemo_jsonl(cfg: SerializeConfig) -> None:
    """Convert CSV sessions to NeMo JSONL format."""
    os.makedirs(cfg.output_dir, exist_ok=True)

    required_cols = ["Sequence", "Time", "File", "RangeOffset", "RangeLength", "Text", "Language", "Type"]

    session_dataframes: List[Tuple[pd.DataFrame, str]] = []
    root = Path(cast(str, cfg.csv_root)).expanduser().resolve()
    csv_files = _discover_local_sessions(root)
    assert csv_files, f"No CSV files found under {root}"
    
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        missing_local = [c for c in required_cols if c not in df.columns]
        assert not missing_local, f"Missing required CSV columns in {csv_file}: {missing_local}"
        session_dataframes.append((df, str(csv_file)))

    random.seed(42)
    random.shuffle(session_dataframes)
    
    total_sessions = len(session_dataframes)
    val_count = int(total_sessions * cfg.val_ratio)
    train_count = total_sessions - val_count

    train_conversations = []
    val_conversations = []
    
    session_turn_counts_all: List[int] = []
    session_turn_counts_kept: List[int] = []
    session_char_counts_all: List[int] = []
    session_char_counts_kept: List[int] = []
    skipped_short_sessions = 0
    docs_written = 0

    for i, (session_df, session_path) in enumerate(session_dataframes):
        if cfg.max_docs and docs_written >= cfg.max_docs:
            break
            
        session_df = pd.DataFrame(session_df.copy())

        assert cfg.target_chars > 0, "target_chars must be positive"
        conversation_chunks = session_to_nemo_conversation_chunks(
            session_df,
            cfg.target_chars,
        )

        assert len(conversation_chunks) >= 1, "At least one conversation chunk should be produced"

        total_turns = sum(len(chunk) for chunk in conversation_chunks)
        session_turn_counts_all.append(total_turns)
        session_chars = sum(
            len(turn.get("value", ""))
            for chunk in conversation_chunks
            for turn in chunk
        )
        session_char_counts_all.append(session_chars)

        if total_turns < cfg.min_session_turns:
            print(f"[warning] Session {session_path} is too short ({total_turns} turns)")
            skipped_short_sessions += 1
            continue
        
        session_turn_counts_kept.append(total_turns)
        session_char_counts_kept.append(session_chars)

        for chunk in conversation_chunks:
            if cfg.max_docs and docs_written >= cfg.max_docs:
                break

            record = {
                "mask": "User",
                "system": "",
                "conversations": chunk,
            }

            if i < train_count:
                train_conversations.append(record)
            else:
                val_conversations.append(record)

            docs_written += 1

        if cfg.max_docs and docs_written >= cfg.max_docs:
            break

    train_path = Path(cfg.output_dir) / "train.jsonl"
    with open(train_path, 'w', encoding='utf-8') as f:
        for record in train_conversations:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    val_path = Path(cfg.output_dir) / "val.jsonl"
    with open(val_path, 'w', encoding='utf-8') as f:
        for record in val_conversations:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    def _print_turn_stats(label: str, values: List[int]) -> None:
        if not values:
            print(f"[debug] No {label} sessions for turn count stats.")
            return
        count = len(values)
        total = sum(values)
        avg = total / count
        min_len = min(values)
        max_len = max(values)
        print(
            f"[debug] {label.capitalize()} sessions turn stats: "
            f"count={count}, avg_turns={avg:.1f}, min_turns={min_len}, max_turns={max_len}"
        )

    def _print_char_stats(label: str, values: List[int]) -> None:
        if not values:
            print(f"[debug] No {label} sessions for character stats.")
            return
        count = len(values)
        total = sum(values)
        avg = total / count
        min_len = min(values)
        max_len = max(values)
        print(
            f"[debug] {label.capitalize()} sessions char stats: "
            f"count={count}, avg_chars={avg:.1f}, min_chars={min_len}, max_chars={max_len}"
        )

    _print_turn_stats("all", session_turn_counts_all)
    _print_turn_stats("kept", session_turn_counts_kept)
    _print_char_stats("all", session_char_counts_all)
    _print_char_stats("kept", session_char_counts_kept)
    
    print(f"\n[summary]")
    print(f"  Total sessions processed: {total_sessions}")
    print(f"  Sessions kept: {len(session_turn_counts_kept)}")
    print(f"  Skipped (too few turns): {skipped_short_sessions}")
    print(f"  Train conversations: {len(train_conversations)}")
    print(f"  Val conversations: {len(val_conversations)}")
    print(f"  Output: {cfg.output_dir}/{{train,val}}.jsonl")


def parse_args() -> SerializeConfig:
    p = argparse.ArgumentParser(
        description="Serialize CSV sessions to JSONL for NeMo 2.0 SFT function calling"
    )
    p.add_argument("--csv_root", type=str, required=True, 
                   help="Root directory containing per-session CSV files")
    p.add_argument("--output_dir", type=str, required=True, 
                   help="Output directory for JSONL files")
    p.add_argument("--min_session_turns", type=int, default=10, 
                   help="Minimum number of turns to keep a session")
    p.add_argument("--max_docs", type=int, default=None, 
                   help="Stop after writing this many unique docs")
    p.add_argument("--val_ratio", type=float, default=0.10, 
                   help="Fraction of sessions to route to validation [0,1)")
    p.add_argument(
        "--target_chars",
        type=int,
        default=8192*3,
        help="Approximate max characters per conversation chunk (<=0 disables chunking)",
    )
    # These parameters are not used for JSONL output but kept for compatibility with SerializeConfig
    p.add_argument("--shard_size", type=int, default=20000, help="(unused for JSONL)")
    p.add_argument("--overlap_chars", type=int, default=128, help="(unused for JSONL)")
    p.add_argument("--arrayrecord_group_size", type=int, default=1, help="(unused for JSONL)")
    
    args = p.parse_args()
    return SerializeConfig(
        output_dir=args.output_dir,
        shard_size=args.shard_size,
        target_chars=args.target_chars,
        overlap_chars=args.overlap_chars,
        min_session_turns=args.min_session_turns,
        max_docs=args.max_docs,
        csv_root=(args.csv_root if args.csv_root else None),
        val_ratio=args.val_ratio,
        arrayrecord_group_size=args.arrayrecord_group_size,
    )


def main() -> None:
    cfg = parse_args()
    to_nemo_jsonl(cfg)


if __name__ == "__main__":
    main()

