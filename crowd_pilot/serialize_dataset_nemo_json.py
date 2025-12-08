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
from transformers import AutoTokenizer

from serialization_utils import session_to_nemo_conversations

@dataclass
class SerializeConfig:
    output_dir: str
    max_tokens_per_conversation: int
    max_tokens_per_message: int
    min_conversation_messages: int
    csv_root: Optional[str]
    val_ratio: float
    tokenizer_model: str


def _discover_local_sessions(root: Path) -> List[Path]:
    # Recursively find all CSV files
    paths: List[Path] = []
    for p in root.rglob("*.csv"):
        if p.is_file():
            paths.append(p)
    paths.sort()
    return paths


def to_nemo_jsonl(cfg: SerializeConfig) -> None:
    """Convert CSV sessions to NeMo JSONL format."""
    assert cfg.max_tokens_per_conversation > 0, "max_tokens_per_conversation must be positive"
    assert cfg.max_tokens_per_message > 0, "max_tokens_per_message must be positive"
    assert cfg.min_conversation_messages > 0, "min_conversation_messages must be positive"
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    print(f"Loading tokenizer from {cfg.tokenizer_model}...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_model)

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
    
    message_counts: List[int] = []
    token_counts: List[int] = []
    conversations_written = 0

    for i, (session_df, _) in enumerate(session_dataframes):
        conversations, per_conversation_tokens = session_to_nemo_conversations(
            session_df,
            cfg.max_tokens_per_conversation,
            max_tokens_per_message=cfg.max_tokens_per_message,
            min_conversation_messages=cfg.min_conversation_messages,
            tokenizer=tokenizer,
        )

        # Per-conversation statistics (for reporting)
        per_conversation_messages = [len(conversation) for conversation in conversations]
        
        message_counts.extend(per_conversation_messages)
        token_counts.extend(per_conversation_tokens)

        for conversation in conversations:
            record = {
                "mask": "User",
                "system": "You are a helpful assistant that can interact multiple times with a computer shell to solve programming tasks.\nYour response must contain exactly ONE bash code block with ONE command (or commands connected with && or ||).\n\nFormat your response as shown in <format_example>.\n\n<format_example>\n```bash\nyour_command_here\n```\n</format_example>\n\nFailure to follow these rules will cause your response to be rejected.",
                "conversations": conversation,
            }

            if i < train_count:
                train_conversations.append(record)
            else:
                val_conversations.append(record)

            conversations_written += 1

    train_path = Path(cfg.output_dir) / "training.jsonl"
    with open(train_path, 'w', encoding='utf-8') as f:
        for record in train_conversations:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    val_path = Path(cfg.output_dir) / "validation.jsonl"
    with open(val_path, 'w', encoding='utf-8') as f:
        for record in val_conversations:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"\n[summary]")
    print(f"  Total sessions processed: {total_sessions}")
    print(f"  Train conversations: {len(train_conversations)}")
    print(f"  Val conversations: {len(val_conversations)}")
    print(f"  Output: {cfg.output_dir}/{{training,validation}}.jsonl")

    total_messages = sum(message_counts)
    total_tokens = sum(token_counts)
    count = len(message_counts)

    metadata = {
        "config": {
            "csv_root": cfg.csv_root,
            "output_dir": cfg.output_dir,
            "min_conversation_messages": cfg.min_conversation_messages,
            "val_ratio": cfg.val_ratio,
            "max_tokens_per_conversation": cfg.max_tokens_per_conversation,
            "max_tokens_per_message": cfg.max_tokens_per_message,
            "tokenizer_model": cfg.tokenizer_model,
        },
        "counts": {
            "total_sessions": total_sessions,
            "train_conversations": len(train_conversations),
            "val_conversations": len(val_conversations),
            "conversations_written": conversations_written,
        },
        "stats": {
            "messages": {
                "total": total_messages,
                "avg": total_messages / count if count > 0 else 0,
                "min": min(message_counts) if message_counts else 0,
                "max": max(message_counts) if message_counts else 0,
            },
            "tokens": {
                "total": total_tokens,
                "avg": total_tokens / count if count > 0 else 0,
                "min": min(token_counts) if token_counts else 0,
                "max": max(token_counts) if token_counts else 0,
            },
        },
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
    p.add_argument("--min_conversation_messages", type=int, default=5, 
                   help="Minimum number of messages to keep a conversation chunk")
    p.add_argument("--val_ratio", type=float, default=0.10, 
                   help="Fraction of sessions to route to validation [0,1)")
    p.add_argument(
        "--max_tokens_per_conversation",
        type=int,
        default=8192,
        help="Maximum tokens per conversation chunk",
    )
    p.add_argument("--max_tokens_per_message", type=int, default=2048, help="Maximum tokens per message")
    p.add_argument("--tokenizer_model", type=str, required=True, help="Path or name of the HuggingFace tokenizer model")
    
    args = p.parse_args()
    return SerializeConfig(
        output_dir=args.output_dir,
        max_tokens_per_conversation=args.max_tokens_per_conversation,
        max_tokens_per_message=args.max_tokens_per_message,
        min_conversation_messages=args.min_conversation_messages,
        csv_root=(args.csv_root if args.csv_root else None),
        val_ratio=args.val_ratio,
        tokenizer_model=args.tokenizer_model,
    )


def main() -> None:
    cfg = parse_args()
    to_nemo_jsonl(cfg)


if __name__ == "__main__":
    main()

