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
import random
from pathlib import Path
from typing import List, Tuple, cast, Optional
from dataclasses import dataclass

import pandas as pd

from serialization_utils import (
    session_to_nemo_conversation_chunks,
    _discover_local_sessions,
)

@dataclass
class SerializeConfig:
    output_dir: str
    max_chars_per_conversation: int
    max_chars_per_turn: int
    min_conversation_turns: int
    max_docs: Optional[int]
    csv_root: Optional[str]
    val_ratio: float


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
    
    turn_counts: List[int] = []
    char_counts: List[int] = []
    docs_written = 0

    for i, (session_df, _) in enumerate(session_dataframes):
        if cfg.max_docs and docs_written >= cfg.max_docs:
            break
            
        session_df = pd.DataFrame(session_df.copy())

        assert cfg.max_chars_per_conversation > 0, "max_chars_per_conversation must be positive"
        assert cfg.max_chars_per_turn > 0, "max_chars_per_turn must be positive"
        conversation_chunks = session_to_nemo_conversation_chunks(
            session_df,
            cfg.max_chars_per_conversation,
            max_chars_per_turn=cfg.max_chars_per_turn,
            min_conversation_turns=cfg.min_conversation_turns,
        )

        # Per-chunk statistics (for reporting)
        per_chunk_turns = [len(chunk) for chunk in conversation_chunks]
        per_chunk_chars = [
            sum(len(turn.get("value", "")) for turn in chunk)
            for chunk in conversation_chunks
        ]
        
        turn_counts.extend(per_chunk_turns)
        char_counts.extend(per_chunk_chars)

        for chunk in conversation_chunks:
            if cfg.max_docs and docs_written >= cfg.max_docs:
                break

            record = {
                "mask": "User",
                "system": "You are a helpful assistant that can interact multiple times with a computer shell to solve programming tasks.\nYour response must contain exactly ONE bash code block with ONE command (or commands connected with && or ||).\n\nFormat your response as shown in <format_example>.\n\n<format_example>\n```bash\nyour_command_here\n```\n</format_example>\n\nFailure to follow these rules will cause your response to be rejected.",
                "conversations": chunk,
            }

            if i < train_count:
                train_conversations.append(record)
            else:
                val_conversations.append(record)

            docs_written += 1

        if cfg.max_docs and docs_written >= cfg.max_docs:
            break

    train_path = Path(cfg.output_dir) / "training.jsonl"
    with open(train_path, 'w', encoding='utf-8') as f:
        for record in train_conversations:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    val_path = Path(cfg.output_dir) / "validation.jsonl"
    with open(val_path, 'w', encoding='utf-8') as f:
        for record in val_conversations:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    def _compute_stats(values: List[int]) -> dict | None:
        if not values:
            return None
        count = len(values)
        total = sum(values)
        avg = total / count if count > 0 else 0.0
        sorted_vals = sorted(values)
        mid = count // 2
        if count % 2 == 1:
            median = float(sorted_vals[mid])
        else:
            median = 0.5 * (sorted_vals[mid - 1] + sorted_vals[mid])
        min_len = min(values)
        max_len = max(values)
        return {
            "count": count,
            "total": total,
            "median": median,
            "avg": avg,
            "min": min_len,
            "max": max_len,
        }

    def _print_turn_stats(label: str, values: List[int]) -> None:
        if not values:
            print(f"[debug] No {label} chunks for turn count stats.")
            return
        stats = _compute_stats(values)
        assert stats is not None
        print(
            f"[debug] {label.capitalize()} chunks turn stats: "
            f"count={stats['count']}, median_turns={stats['median']:.1f}, avg_turns={stats['avg']:.1f}, "
            f"min_turns={stats['min']}, max_turns={stats['max']}"
        )

    def _print_char_stats(label: str, values: List[int]) -> None:
        if not values:
            print(f"[debug] No {label} chunks for character stats.")
            return
        stats = _compute_stats(values)
        assert stats is not None
        print(
            f"[debug] {label.capitalize()} chunks char stats: "
            f"count={stats['count']}, median_chars={stats['median']:.1f}, avg_chars={stats['avg']:.1f}, "
            f"min_chars={stats['min']}, max_chars={stats['max']}"
        )

    print(f"\n[summary]")
    print(f"  Total sessions processed: {total_sessions}")
    print(f"  Train conversations: {len(train_conversations)}")
    print(f"  Val conversations: {len(val_conversations)}")
    print(f"  Output: {cfg.output_dir}/{{training,validation}}.jsonl")

    metadata = {
        "config": {
            "csv_root": cfg.csv_root,
            "output_dir": cfg.output_dir,
            "min_conversation_turns": cfg.min_conversation_turns,
            "max_docs": cfg.max_docs,
            "val_ratio": cfg.val_ratio,
            "max_chars_per_conversation": cfg.max_chars_per_conversation,
            "max_chars_per_turn": cfg.max_chars_per_turn,
        },
        "counts": {
            "total_sessions": total_sessions,
            "train_conversations": len(train_conversations),
            "val_conversations": len(val_conversations),
            "docs_written": docs_written,
        },
        "turn_stats": _compute_stats(turn_counts),
        "char_stats": _compute_stats(char_counts),
        "files": {
            "train_path": str(train_path),
            "val_path": str(val_path),
        },
    }
    metadata_path = Path(cfg.output_dir) / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as mf:
        json.dump(metadata, mf, ensure_ascii=False, indent=2)
    print(f"  Metadata: {metadata_path}")


def parse_args() -> SerializeConfig:
    p = argparse.ArgumentParser(
        description="Serialize CSV sessions to JSONL for NeMo 2.0 SFT function calling"
    )
    p.add_argument("--csv_root", type=str, required=True, 
                   help="Root directory containing per-session CSV files")
    p.add_argument("--output_dir", type=str, required=True, 
                   help="Output directory for JSONL files")
    p.add_argument("--min_conversation_turns", type=int, default=5, 
                   help="Minimum number of turns to keep a conversation chunk")
    p.add_argument("--max_docs", type=int, default=None, 
                   help="Stop after writing this many unique docs")
    p.add_argument("--val_ratio", type=float, default=0.10, 
                   help="Fraction of sessions to route to validation [0,1)")
    p.add_argument(
        "--max_chars_per_conversation",
        type=int,
        default=8192*16*3,
        help="Maximum characters per conversation chunk",
    )
    p.add_argument("--max_chars_per_turn", type=int, default=8192*3, help="Maximum characters per turn")
    args = p.parse_args()
    return SerializeConfig(
        output_dir=args.output_dir,
        max_chars_per_conversation=args.max_chars_per_conversation,
        max_chars_per_turn=args.max_chars_per_turn,
        min_conversation_turns=args.min_conversation_turns,
        max_docs=args.max_docs,
        csv_root=(args.csv_root if args.csv_root else None),
        val_ratio=args.val_ratio,
    )


def main() -> None:
    cfg = parse_args()
    to_nemo_jsonl(cfg)


if __name__ == "__main__":
    main()

