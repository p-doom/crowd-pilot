#!/usr/bin/env python3
"""
Common utilities for dataset serialization scripts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import difflib
import re
import pandas as pd
from datasets import Dataset, load_dataset


_ANSI_CSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
_ANSI_OSC_TERMINATED_RE = re.compile(r"\x1b\][\s\S]*?(?:\x07|\x1b\\)")
_ANSI_OSC_LINE_FALLBACK_RE = re.compile(r"\x1b\][^\n]*$")
_BRACKETED_PASTE_ENABLE = "\x1b[?2004h"
_BRACKETED_PASTE_DISABLE = "\x1b[?2004l"
_OSC_633 = "\x1b]633;"
_OSC_0 = "\x1b]0;"



@dataclass
class ConversationState:
    """
    Mutable state used while constructing conversations.
    """
    conversations: List[List[Dict[str, str]]]
    max_tokens_per_conversation: int
    max_tokens_per_message: int
    min_conversation_messages: int
    tokenizer: Any
    conversation_token_counts: List[int] = field(default_factory=list)
    current_conversation: List[Dict[str, str]] = field(default_factory=list)
    current_tokens: int = 0
    files_opened_in_conversation: set[str] = field(default_factory=set)

    def finalize_conversation(self) -> None:
        """
        Finalize the current conversation: check constraints and append if valid.
        Then reset state for the next conversation.
        """
        if self.current_conversation:
            is_long_enough = len(self.current_conversation) >= self.min_conversation_messages
            has_user = any(msg.get("from") == "User" for msg in self.current_conversation)
            has_assistant = any(msg.get("from") == "Assistant" for msg in self.current_conversation)

            if is_long_enough and has_user and has_assistant:
                self.conversations.append(self.current_conversation)
                self.conversation_token_counts.append(self.current_tokens)
        
        self.current_conversation = []
        self.current_tokens = 0
        self.files_opened_in_conversation.clear()

    def append_message(self, message: Dict[str, str]) -> None:
        value = message["value"]
        
        tokens = self.tokenizer.encode(value)
        num_tokens = len(tokens)

        if num_tokens > self.max_tokens_per_message:
            tokens = tokens[:self.max_tokens_per_message]
            value = self.tokenizer.decode(tokens)
            message["value"] = value
            num_tokens = self.max_tokens_per_message

        if self.current_tokens + num_tokens > self.max_tokens_per_conversation:
            self.finalize_conversation()

        self.current_conversation.append(message)
        self.current_tokens += num_tokens

    def maybe_capture_file_contents(
        self,
        file_path: str,
        content: str,
    ) -> None:
        """
        Capture the contents of the given file in the current conversation if it hasn't been opened yet.
        """
        if file_path in self.files_opened_in_conversation:
            return
        cmd = f"cat -n {file_path}"
        self.append_message({
            "from": "Assistant",
            "value": _fenced_block("bash", _clean_text(cmd)),
        })
        output = _line_numbered_output(content)
        self.append_message({
            "from": "User",
            "value": f"<stdout>\n{output}\n</stdout>",
        })
        self.files_opened_in_conversation.add(file_path)


def _clean_text(text: str) -> str:
    # Normalize line endings and strip trailing spaces; preserve tabs/newlines.
    return text.replace("\r\n", "\n").replace("\r", "\n").rstrip()


def _fenced_block(language: Optional[str], content: str) -> str:
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


def _apply_backspaces(text: str) -> str:
    out: List[str] = []
    for ch in text:
        if ch == "\b":  # \x08
            if out:
                out.pop()
        else:
            out.append(ch)
    return "".join(out)


def _normalize_terminal_output(raw: str) -> str:
    """
    Normalize PTY/terminal output for training:
      - Apply backspaces (\x08)
      - Strip OSC (window title/shell integration) first, keeping BEL/ST terminators intact
      - Resolve carriage returns (\r) by keeping the last rewrite per line
      - Strip CSI (coloring etc.)
      - Finally drop any remaining BEL (\x07)
    """
    if not raw:
        return raw
    s = _apply_backspaces(raw)
    # Remove OSC sequences that are properly terminated (BEL or ST)
    s = _ANSI_OSC_TERMINATED_RE.sub("", s)
    # Fallback: drop any unterminated OSC up to end-of-line only
    s = "\n".join(_ANSI_OSC_LINE_FALLBACK_RE.sub("", line) for line in s.split("\n"))
    # Resolve carriage returns per line:
    # - If there are multiple rewrites, keep the last non-empty conversation
    # - If it's CRLF (ending with '\r' before '\n'), keep the content before '\r'
    resolved_lines: List[str] = []
    for seg in s.split("\n"):
        parts = seg.split("\r")
        chosen = ""
        # pick last non-empty part if available; else last part
        for p in reversed(parts):
            if p != "":
                chosen = p
                break
        if chosen == "" and parts:
            chosen = parts[-1]
        resolved_lines.append(chosen)
    s = "\n".join(resolved_lines)
    # Strip ANSI escape sequences
    s = _ANSI_CSI_RE.sub("", s)
    # Remove any remaining BEL beeps
    s = s.replace("\x07", "")
    return s


def _line_numbered_output(content: str, start_line: Optional[int] = None, end_line: Optional[int] = None) -> str:
    # FIXME (f.srambical): check whether this corresponds **exactly** to the output of cat -n {file_path} | sed -n '{vstart},{vend}p'
    lines = content.splitlines()
    total = len(lines)
    if total == 0:
        return ""
    s = 1 if start_line is None else max(1, min(start_line, total))
    e = total if end_line is None else max(1, min(end_line, total))
    assert e >= s, "End line number cannot be less than start line number! Likely a bug in the line numbering computation."
    buf: List[str] = []
    for idx in range(s, e + 1):
        buf.append(f"{idx:6}\t{lines[idx - 1]}")
    return "\n".join(buf)


def _compute_viewport(total_lines: int, center_line: int, radius: int) -> Tuple[int, int]:
    if total_lines <= 0:
        return (1, 0)
    start = max(1, center_line - radius)
    end = min(total_lines, center_line + radius)
    assert end >= start, "Viewport cannot have negative width! Likely a bug in the viewport computation."
    return (start, end)


def _escape_single_quotes_for_sed(text: str) -> str:
    # Close quote, add an escaped single quote, reopen quote: '"'"'
    return text.replace("'", "'\"'\"'")


def _compute_changed_block_lines(
    before: str, after: str
) -> Tuple[int, int, int, int, List[str]]:
    """
    Return 1-based start and end line numbers in 'before' that should be
    replaced, 1-based start and end line numbers in 'after' that contain
    the replacement, and the replacement lines from 'after'.

    For pure deletions, the replacement list may be empty.
    """
    before_lines = before.splitlines()
    after_lines = after.splitlines()
    sm = difflib.SequenceMatcher(a=before_lines, b=after_lines, autojunk=False)
    opcodes = [op for op in sm.get_opcodes() if op[0] != "equal"]
    assert opcodes, "Opcode list cannot be empty! Likely a bug in the diff computation."

    first = opcodes[0]
    last = opcodes[-1]
    # i1/i2 refer to 'before' indices, j1/j2 to 'after'
    start_before = max(1, first[1] + 1)
    end_before = last[2]  # no increment since we go from 'exclusive' to 'inclusive' indexing
    start_after = max(1, first[3] + 1)
    end_after = last[4]
    replacement_lines = after_lines[first[3] : last[4]]
    return (start_before, end_before, start_after, end_after, replacement_lines)


def session_to_nemo_conversations(
    df: pd.DataFrame,
    max_tokens_per_conversation: int,
    max_tokens_per_message: int,
    min_conversation_messages: int,
    tokenizer: Any,
    viewport_radius: int = 10,
    normalize_terminal_output: bool = True,
    coalesce_radius: int = 5,
) -> Tuple[List[List[Dict[str, str]]], List[int]]:
    """
    Convert a session DataFrame to one or more NeMo conversations.

    - Conversations are created by approximately limiting the total tokens
      across all `value` fields to `max_tokens_per_conversation`.
    - When a new conversation starts (after the first), the first time a file is
      referenced in that conversation we re-log the full file contents with
      `cat -n <file>` and numbered output so that each conversation is self-contained.
    """
    file_states: Dict[str, str] = {}
    per_file_viewport: Dict[str, Optional[Tuple[int, int]]] = {}

    conversations: List[List[Dict[str, str]]] = []
    conversation_token_counts: List[int] = []
    conversation_state = ConversationState(
        conversations=conversations,
        conversation_token_counts=conversation_token_counts,
        max_tokens_per_conversation=max_tokens_per_conversation,
        max_tokens_per_message=max_tokens_per_message,
        min_conversation_messages=min_conversation_messages,
        tokenizer=tokenizer,
    )

    terminal_output_buffer: List[str] = []
    pending_edits_before: Dict[str, Optional[str]] = {}
    pending_edit_regions: Dict[str, Optional[Tuple[int, int]]] = {}

    def _flush_terminal_output_buffer() -> None:
        if not terminal_output_buffer:
            return
        aggregated = "".join(terminal_output_buffer)
        out = aggregated
        if normalize_terminal_output:
            out = _normalize_terminal_output(out)
        cleaned = _clean_text(out)
        if cleaned.strip():
            conversation_state.append_message({
                "from": "User",
                "value": f"<stdout>\n{cleaned}\n</stdout>",
            })
        terminal_output_buffer.clear()

    def _flush_pending_edit_for_file(target_file: str) -> None:
        before_snapshot = pending_edits_before.get(target_file)
        if before_snapshot is None:
            return
        after_state = file_states.get(target_file, "")
        if before_snapshot.rstrip("\n") == after_state.rstrip("\n"):
            pending_edits_before[target_file] = None
            pending_edit_regions[target_file] = None
            return
        (
            start_before,
            end_before,
            start_after,
            end_after,
            repl_lines,
        ) = _compute_changed_block_lines(before_snapshot, after_state)
        before_total_lines = len(before_snapshot.splitlines())
        if end_before < start_before:
            escaped_lines = [_escape_single_quotes_for_sed(line) for line in repl_lines]
            sed_payload = "\n".join(escaped_lines)
            if start_before <= max(1, before_total_lines):
                sed_cmd = f"sed -i '{start_before}i\\\n{sed_payload}' {target_file}"
            else:
                sed_cmd = f"sed -i '$a\\\n{sed_payload}' {target_file}"
        elif not repl_lines:
            sed_cmd = f"sed -i '{start_before},{end_before}d' {target_file}"
        else:
            escaped_lines = [_escape_single_quotes_for_sed(line) for line in repl_lines]
            sed_payload = "\n".join(escaped_lines)
            sed_cmd = f"sed -i '{start_before},{end_before}c\\\n{sed_payload}' {target_file}"
        total_lines = len(after_state.splitlines())
        center = (start_after + end_after) // 2
        vp = _compute_viewport(total_lines, center, viewport_radius)
        per_file_viewport[target_file] = vp
        vstart, vend = vp
        conversation_state.maybe_capture_file_contents(target_file, before_snapshot)
        chained_cmd = f"{sed_cmd} && cat -n {target_file} | sed -n '{vstart},{vend}p'"
        conversation_state.append_message({
            "from": "Assistant",
            "value": _fenced_block("bash", _clean_text(chained_cmd)),
        })
        viewport_output = _line_numbered_output(after_state, vstart, vend)
        conversation_state.append_message({
            "from": "User",
            "value": f"<stdout>\n{viewport_output}\n</stdout>",
        })
        pending_edits_before[target_file] = None
        pending_edit_regions[target_file] = None

    def _flush_all_pending_edits() -> None:
        for fname in list(pending_edits_before.keys()):
            _flush_pending_edit_for_file(fname)

    for i in range(len(df)):
        row = df.iloc[i]
        file_path: str = row["File"]
        event_type = row["Type"]

        match event_type:
            case "tab":
                _flush_all_pending_edits()
                _flush_terminal_output_buffer()
                text = row["Text"]
                if pd.notna(text):
                    content = str(text).replace("\\n", "\n").replace("\\r", "\r")
                    file_states[file_path] = content
                    cmd = f"cat -n {file_path}"
                    conversation_state.append_message({
                        "from": "Assistant",
                        "value": _fenced_block("bash", _clean_text(cmd)),
                    })
                    output = _line_numbered_output(content)
                    conversation_state.append_message({
                        "from": "User",
                        "value": f"<stdout>\n{output}\n</stdout>",
                    })
                    conversation_state.files_opened_in_conversation.add(file_path)
                else:
                    # File switch without content snapshot: show current viewport only
                    content = file_states.get(file_path, "")
                    total_lines = len(content.splitlines())
                    vp = per_file_viewport.get(file_path)
                    if not vp or vp[1] == 0:
                        vp = _compute_viewport(total_lines, 1, viewport_radius)
                        per_file_viewport[file_path] = vp
                    if vp and vp[1] >= vp[0]:
                        vstart, vend = vp
                        conversation_state.maybe_capture_file_contents(file_path, content)
                        cmd = f"cat -n {file_path} | sed -n '{vstart},{vend}p'"
                        conversation_state.append_message({
                            "from": "Assistant",
                            "value": _fenced_block("bash", _clean_text(cmd)),
                        })
                        viewport_output = _line_numbered_output(content, vstart, vend)
                        conversation_state.append_message({
                            "from": "User",
                            "value": f"<stdout>\n{viewport_output}\n</stdout>",
                        })

            case "content":
                _flush_terminal_output_buffer()
                offset = int(row["RangeOffset"])
                length = int(row["RangeLength"])
                new_text = row["Text"]
                before = file_states.get(file_path, "")
                # Approximate current edit region in line space
                new_text_str = str(new_text) if pd.notna(new_text) else ""
                start_line_current = before[:offset].count("\n") + 1
                deleted_conversation = before[offset:offset + length]
                lines_added = new_text_str.count("\n")
                lines_deleted = deleted_conversation.count("\n")
                region_start = start_line_current
                region_end = start_line_current + max(lines_added, lines_deleted, 0)
                # Flush pending edits if this edit is far from the pending region
                current_region = pending_edit_regions.get(file_path)
                if current_region is not None:
                    rstart, rend = current_region
                    if region_start < (rstart - coalesce_radius) or region_start > (rend + coalesce_radius):
                        _flush_pending_edit_for_file(file_path)
                        current_region = None
                after = _apply_change(before, offset, length, new_text)
                if pending_edits_before.get(file_path) is None:
                    pending_edits_before[file_path] = before
                # Update/initialize region union
                if current_region is None:
                    pending_edit_regions[file_path] = (region_start, max(region_start, region_end))
                else:
                    rstart, rend = current_region
                    pending_edit_regions[file_path] = (min(rstart, region_start), max(rend, region_end))
                file_states[file_path] = after

            case "selection_command" | "selection_mouse" | "selection_keyboard":
                # During an edit burst (pending edits), suppress flush and viewport emissions
                if pending_edits_before.get(file_path) is None:
                    _flush_terminal_output_buffer()
                else:
                    # Skip emitting viewport while edits are pending to avoid per-keystroke sed/cat spam
                    continue
                offset = int(row["RangeOffset"])
                content = file_states.get(file_path, "")
                total_lines = len(content.splitlines())
                target_line = content[:offset].count("\n") + 1
                vp = per_file_viewport.get(file_path)
                should_emit = False
                if not vp or vp[1] == 0:
                    vp = _compute_viewport(total_lines, target_line, viewport_radius)
                    per_file_viewport[file_path] = vp
                    should_emit = True
                else:
                    vstart, vend = vp
                    if target_line < vstart or target_line > vend:
                        vp = _compute_viewport(total_lines, target_line, viewport_radius)
                        per_file_viewport[file_path] = vp
                        should_emit = True
                if should_emit and vp and vp[1] >= vp[0]:
                    vstart, vend = vp
                    conversation_state.maybe_capture_file_contents(file_path, content)
                    cmd = f"cat -n {file_path} | sed -n '{vstart},{vend}p'"
                    conversation_state.append_message({
                        "from": "Assistant",
                        "value": _fenced_block("bash", _clean_text(cmd)),
                    })
                    viewport_output = _line_numbered_output(content, vstart, vend)
                    conversation_state.append_message({
                        "from": "User",
                        "value": f"<stdout>\n{viewport_output}\n</stdout>",
                    })

            case "terminal_command":
                _flush_all_pending_edits()
                _flush_terminal_output_buffer()
                command = row["Text"]
                command_str = str(command).replace("\\n", "\n").replace("\\r", "\r")
                conversation_state.append_message({
                    "from": "Assistant",
                    "value": _fenced_block("bash", _clean_text(command_str)),
                })

            case "terminal_output":
                output = row["Text"]
                raw_output = str(output).replace("\\n", "\n").replace("\\r", "\r")
                terminal_output_buffer.append(raw_output)

            case "terminal_focus":
                _flush_all_pending_edits()
                _flush_terminal_output_buffer()
                # No-op for bash transcript; focus changes don't emit commands/output
                pass

            case "git_branch_checkout":
                _flush_all_pending_edits()
                _flush_terminal_output_buffer()
                branch_info = row["Text"]
                branch_str = str(branch_info).replace("\\n", "\n").replace("\\r", "\r")
                cleaned = _clean_text(branch_str)
                m = re.search(r"to '([^']+)'", cleaned)
                if not m:
                    raise ValueError(f"Could not extract branch name from git checkout message: {cleaned}")
                branch_name = m.group(1).strip()
                # Safe-quote branch if it contains special characters
                if re.search(r"[^A-Za-z0-9._/\\-]", branch_name):
                    branch_name = "'" + branch_name.replace("'", "'\"'\"'") + "'"
                cmd = f"git checkout {branch_name}"
                conversation_state.append_message({
                    "from": "Assistant",
                    "value": _fenced_block("bash", _clean_text(cmd)),
                })

            case _:
                raise ValueError(f"Unknown event type: {event_type}")

    _flush_all_pending_edits()
    _flush_terminal_output_buffer()
    conversation_state.finalize_conversation()
    return conversations, conversation_token_counts



def load_hf_csv(hf_path: str, split: str) -> Dataset:
    loaded = load_dataset(hf_path, split=split)

    assert isinstance(loaded, Dataset), "Expected a Dataset from load_dataset"
    return loaded