#!/usr/bin/env python3
"""
Common utilities for dataset serialization scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict

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
class SerializeConfig:
    output_dir: str
    shard_size: int
    target_chars: int
    overlap_chars: int
    min_session_chars: int
    max_docs: Optional[int]
    long_pause_threshold_ms: int
    csv_root: Optional[str]
    val_ratio: float
    arrayrecord_group_size: Optional[int] = None


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
    # - If there are multiple rewrites, keep the last non-empty chunk
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
    if e < s:
        # FIXME (f.srambical): If this does not happen, remove the condition
        raise ValueError("This should never happen!")
        e = s
    buf: List[str] = []
    for idx in range(s, e + 1):
        buf.append(f"{idx:6}\t{lines[idx - 1]}")
    return "\n".join(buf)


def _compute_viewport(total_lines: int, center_line: int, radius: int) -> Tuple[int, int]:
    if total_lines <= 0:
        return (1, 0)
    start = max(1, center_line - radius)
    end = min(total_lines, center_line + radius)
    if end < start:
        # FIXME (f.srambical): If this does not happen, remove the condition
        raise ValueError("This should never happen!")
    return (start, end)


def _escape_single_quotes_for_sed(text: str) -> str:
    # Close quote, add an escaped single quote, reopen quote: '"'"'
    return text.replace("'", "'\"'\"'")


def _compute_changed_block_lines(before: str, after: str) -> Tuple[int, int, List[str]]:
    """
    Return 1-based start and end line numbers in 'before' that should be replaced,
    and the replacement lines from 'after'.
    For pure deletions, the replacement list may be empty.
    """
    before_lines = before.splitlines()
    after_lines = after.splitlines()
    sm = difflib.SequenceMatcher(a=before_lines, b=after_lines, autojunk=False)
    opcodes = [op for op in sm.get_opcodes() if op[0] != "equal"]
    if not opcodes:
        # FIXME (f.srambical): clean this up
        raise ValueError("No diff opcodes found for content change")
                # No visible change; choose a safe single-line replace at end of file
        start_line = max(1, len(before_lines))
        end_line = start_line
        repl = after_lines[start_line - 1:start_line] if after_lines else [""]
        return (start_line, end_line, repl)

    first = opcodes[0]
    last = opcodes[-1]
    # i1/i2 refer to 'before' indices, j1/j2 to 'after'
    start_line = (first[1] + 1) if (first[1] + 1) > 0 else 1
    end_line = last[2] # no increment since we go from 'exclusive' to 'inclusive' indexing
    replacement_lines = after_lines[first[3]:last[4]]
    return (start_line, end_line, replacement_lines)


def _session_to_transcript(
    df: pd.DataFrame,
    long_pause_threshold_ms: int,
) -> str:

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

    return "\n".join(parts).strip()


def session_to_bash_formatted_transcript(
    df: pd.DataFrame,
    viewport_radius: int = 10,
    normalize_terminal_output: bool = True,
    coalesce_radius: int = 5,
) -> str:
    r"""
    Serialize a session to a bash-like transcript comprised of:
      - Commands (bash fenced blocks): cat -n, sed -i 'S,Ec\...' && cat -n | sed -n 'VSTART,VENDp'
      - Outputs (<stdout>...</stdout>) that reflect the file state after each action
    Tracks per-file state and a per-file viewport. Viewport only shifts when selection moves out of bounds
    or when first initialized.
    """
    file_states: Dict[str, str] = {}
    per_file_viewport: Dict[str, Optional[Tuple[int, int]]] = {}

    parts: List[str] = []
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
            parts.append(f"<stdout>\n{cleaned}\n</stdout>")
        terminal_output_buffer.clear()

    def _flush_pending_edit_for_file(target_file: str) -> None:
        before_snapshot = pending_edits_before.get(target_file)
        if before_snapshot is None:
            return
        after_state = file_states.get(target_file, "")
        try:
            start_line, end_line, repl_lines = _compute_changed_block_lines(before_snapshot, after_state)
        except ValueError:
            pending_edits_before[target_file] = None
            return
        before_total_lines = len(before_snapshot.splitlines())
        if end_line < start_line:
            escaped_lines = [_escape_single_quotes_for_sed(line) for line in repl_lines]
            sed_payload = "\n".join(escaped_lines)
            if start_line <= max(1, before_total_lines):
                sed_cmd = f"sed -i '{start_line}i\\\n{sed_payload}' {target_file}"
            else:
                sed_cmd = f"sed -i '$a\\\n{sed_payload}' {target_file}"
        elif not repl_lines:
            sed_cmd = f"sed -i '{start_line},{end_line}d' {target_file}"
        else:
            escaped_lines = [_escape_single_quotes_for_sed(line) for line in repl_lines]
            sed_payload = "\n".join(escaped_lines)
            sed_cmd = f"sed -i '{start_line},{end_line}c\\\n{sed_payload}' {target_file}"
        total_lines = len(after_state.splitlines())
        center = (start_line + end_line) // 2
        vp = _compute_viewport(total_lines, center, viewport_radius)
        per_file_viewport[target_file] = vp
        vstart, vend = vp
        chained_cmd = f"{sed_cmd} && cat -n {target_file} | sed -n '{vstart},{vend}p'"
        parts.append(_fenced_block(target_file, "bash", _clean_text(chained_cmd)))
        viewport_output = _line_numbered_output(after_state, vstart, vend)
        parts.append(f"<stdout>\n{viewport_output}\n</stdout>")
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
                    # First open with full file capture
                    cmd = f"cat -n {file_path}"
                    parts.append(_fenced_block(file_path, "bash", _clean_text(cmd)))
                    output = _line_numbered_output(content)
                    parts.append(f"<stdout>\n{output}\n</stdout>")
                else:
                    # File switch without content snapshot: show current viewport only
                    content = file_states.get(file_path, "")
                    total_lines = len(content.splitlines())
                    vp = per_file_viewport.get(file_path)
                    if not vp or vp[1] == 0:
                        vp = _compute_viewport(total_lines, 1, viewport_radius)
                        per_file_viewport[file_path] = vp
                    if vp:
                        vstart, vend = vp
                        cmd = f"cat -n {file_path} | sed -n '{vstart},{vend}p'"
                        parts.append(_fenced_block(file_path, "bash", _clean_text(cmd)))
                        viewport_output = _line_numbered_output(content, vstart, vend)
                        parts.append(f"<stdout>\n{viewport_output}\n</stdout>")

            case "content":
                _flush_terminal_output_buffer()
                offset = int(row["RangeOffset"])
                length = int(row["RangeLength"])
                new_text = row["Text"]
                before = file_states.get(file_path, "")
                # Approximate current edit region in line space
                new_text_str = str(new_text) if pd.notna(new_text) else ""
                start_line_current = before[:offset].count("\n") + 1
                deleted_chunk = before[offset:offset + length]
                lines_added = new_text_str.count("\n")
                lines_deleted = deleted_chunk.count("\n")
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
                if should_emit and vp:
                    vstart, vend = vp
                    cmd = f"cat -n {file_path} | sed -n '{vstart},{vend}p'"
                    parts.append(_fenced_block(file_path, "bash", _clean_text(cmd)))
                    viewport_output = _line_numbered_output(content, vstart, vend)
                    parts.append(f"<stdout>\n{viewport_output}\n</stdout>")

            case "terminal_command":
                _flush_all_pending_edits()
                _flush_terminal_output_buffer()
                command = row["Text"]
                command_str = str(command).replace("\\n", "\n").replace("\\r", "\r")
                parts.append(_fenced_block(file_path, "bash", _clean_text(command_str)))

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
                parts.append(_fenced_block(file_path, "bash", _clean_text(cmd)))

            case _:
                _flush_all_pending_edits()
                _flush_terminal_output_buffer()
                raise ValueError(f"Unknown event type: {event_type}")

    _flush_all_pending_edits()
    _flush_terminal_output_buffer()
    return "\n".join(parts).strip()

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


def _chunk_text(text: str, target_chars: int, overlap_chars: int) -> List[str]:
    """Split a long text into overlapping chunks near target length."""
    if target_chars <= 0:
        return [text]
    n = len(text)
    if n <= target_chars:
        return [text]

    chunks: List[str] = []
    start = 0
    # Ensure sane overlap
    overlap = max(0, min(overlap_chars, target_chars // 2))
    while start < n:
        end_target = min(start + target_chars, n)
        if end_target < n:
            end = end_target
        else:
            end = n
        chunk = text[start:end].strip()
        chunks.append(chunk)
        if end == n:
            break
        # advance with overlap
        start = max(0, end - overlap)
        if start >= n:
            break
    return chunks


