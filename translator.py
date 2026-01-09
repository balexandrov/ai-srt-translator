import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from openai import OpenAI


DEBUG_LEVEL = 0


def log(msg: str, level: int = 1) -> None:
    """Level 0 always shows; higher levels depend on --debug."""
    if DEBUG_LEVEL >= level:
        print(msg, flush=True)

# ----------------------------
# SRT parsing that preserves exact separators (blank lines) and line endings
# ----------------------------

@dataclass
class SrtBlock:
    number_line: str            # includes newline
    timestamp_line: str         # includes newline
    text_lines: List[str]       # each includes newline
    separator_after: str        # exact blank lines after this block (may be "")

def _is_blank_line(line: str) -> bool:
    return line.strip("\r\n") == ""

def parse_srt_preserve(raw: str) -> Tuple[List[SrtBlock], str]:
    """
    Returns (blocks, leading_preamble).
    leading_preamble is any blank lines at the start of file, preserved.
    """
    # Keep original line endings exactly
    lines = raw.splitlines(keepends=True)

    i = 0
    preamble_parts: List[str] = []
    while i < len(lines) and _is_blank_line(lines[i]):
        preamble_parts.append(lines[i])
        i += 1
    leading_preamble = "".join(preamble_parts)

    blocks: List[SrtBlock] = []

    while i < len(lines):
        block_lines: List[str] = []
        while i < len(lines) and not _is_blank_line(lines[i]):
            block_lines.append(lines[i])
            i += 1

        sep_parts: List[str] = []
        while i < len(lines) and _is_blank_line(lines[i]):
            sep_parts.append(lines[i])
            i += 1
        sep = "".join(sep_parts)

        if not block_lines:
            # trailing blanks at end (if any) go into preamble? here: attach to last block if exists
            if blocks:
                blocks[-1].separator_after += sep
            else:
                leading_preamble += sep
            continue

        if len(block_lines) < 2:
            raise ValueError(f"Malformed SRT block (expected >=2 lines): {block_lines!r}")

        number_line = block_lines[0]
        timestamp_line = block_lines[1]
        text_lines = block_lines[2:]  # may be empty (rare but possible)

        blocks.append(SrtBlock(
            number_line=number_line,
            timestamp_line=timestamp_line,
            text_lines=text_lines,
            separator_after=sep
        ))

    return blocks, leading_preamble

def rebuild_srt(blocks: List[SrtBlock], leading_preamble: str) -> str:
    out = [leading_preamble]
    for b in blocks:
        out.append(b.number_line)
        out.append(b.timestamp_line)
        out.extend(b.text_lines)
        out.append(b.separator_after)
    return "".join(out)


def derive_lang_code(lang: str) -> str:
    """Best-effort 2-letter code from language name; falls back to first two letters."""
    lang_lower = lang.strip().lower()
    common = {
        "english": "en",
        "bulgarian": "bg",
        "spanish": "es",
        "french": "fr",
        "german": "de",
        "italian": "it",
        "portuguese": "pt",
        "chinese": "zh",
        "japanese": "ja",
        "korean": "ko",
        "russian": "ru",
        "arabic": "ar",
        "hindi": "hi",
        "turkish": "tr",
        "polish": "pl",
        "dutch": "nl",
        "greek": "el",
    }
    if lang_lower in common:
        return common[lang_lower]
    letters = re.findall(r"[a-z]", lang_lower)
    if len(letters) >= 2:
        return (letters[0] + letters[1]).lower()
    if len(lang_lower) >= 2:
        return lang_lower[:2]
    return (lang_lower or "xx").ljust(2, "x")

# ----------------------------
# OpenAI call (Structured Outputs JSON schema)
# ----------------------------

def translate_chunk(
    client: OpenAI,
    model: str,
    chunk_text_lines: List[List[str]],  # per block: list of text lines (without newline endings)
    prior_glossary: List[str],
    source_lang_hint: str,
    target_lang: str,
) -> Tuple[List[List[str]], List[str], dict]:
    """
    Returns translated lines with EXACT same shape and a rolling glossary list.
    Uses JSON schema structured output for robustness. :contentReference[oaicite:1]{index=1}
    """
    expected_blocks = len(chunk_text_lines)

    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "blocks": {
                "type": "array",
                "minItems": expected_blocks,
                "maxItems": expected_blocks,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "lines": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    },
                    "required": ["lines"]
                }
            },
            "glossary": {
                "type": "array",
                "items": {"type": "string"}
            }
        },
        "required": ["blocks", "glossary"]
    }

    prior_glossary_text = "\n".join([f"- {item}" for item in prior_glossary]) if prior_glossary else "(none provided; start a concise list)"

    instructions = f"""
You are a professional human subtitle translator.

Task: translate subtitle text lines to {target_lang}.
Source language can be {source_lang_hint} (may be mixed).

Translation hard rules:
- Translate ONLY the given text lines.
- Do NOT add/remove blocks.
- Keep the SAME number of lines inside each block (do not merge/split lines).
- Keep punctuation intent, tone, slang, ambiguity, hesitation.
- Do not invent meaning.

Glossary/Consistency list rules:
- Produce a concise, unstructured list (~10-30 items guidance only) of names, nicknames, places/organizations, recurring terms, and key style decisions (e.g., formal vs informal address, swear style, fixed translations).
- If nothing new appears in this chunk, repeat the prior glossary unchanged.
- Preserve tone and context cues that matter for consistency.

Prior glossary to repeat/extend:
{prior_glossary_text}

Return ONLY valid JSON matching the provided schema.
""".strip()

    # Provide the chunk as JSON input (safe & unambiguous)
    user_input_obj = {
        "blocks": [{"lines": lines} for lines in chunk_text_lines],
        "prior_glossary": prior_glossary,
    }

    if DEBUG_LEVEL >= 2:
        log(f"Request payload: instructions=\n{instructions}\ninput={json.dumps(user_input_obj, ensure_ascii=False)}", level=2)

    resp = client.responses.create(
        model=model,
        temperature=0,
        instructions=instructions,
        input=json.dumps(user_input_obj, ensure_ascii=False),
        text={
            "format": {
                "type": "json_schema",
                "name": "srt_translation",
                "strict": True,
                "schema": schema
            }
        }
    )

    # Capture usage if provided by the API
    usage_data: dict = {}
    usage_raw = getattr(resp, "usage", None)
    if usage_raw is not None:
        if hasattr(usage_raw, "model_dump"):
            usage_data = usage_raw.model_dump()
        elif isinstance(usage_raw, dict):
            usage_data = usage_raw

    # With structured outputs, output_text should be valid JSON
    data = json.loads(resp.output_text)
    out_blocks = data["blocks"]
    glossary_out = data.get("glossary", [])

    if len(out_blocks) != expected_blocks:
        log(
            f"Block count mismatch: returned {len(out_blocks)} vs expected {expected_blocks}; coercing",
            level=1,
        )
        if len(out_blocks) > expected_blocks:
            out_blocks = out_blocks[:expected_blocks]
        else:
            # Pad missing blocks with empty lines matching the expected shape
            for idx in range(len(out_blocks), expected_blocks):
                out_blocks.append({"lines": ["" for _ in chunk_text_lines[idx]]})
    if not isinstance(glossary_out, list):
        raise ValueError("Model returned non-list glossary")

    translated: List[List[str]] = []
    for idx, (orig_lines, b) in enumerate(zip(chunk_text_lines, out_blocks)):
        lines = b["lines"]
        if len(lines) != len(orig_lines):
            log(
                " | ".join([
                    f"Block {idx}: line count {len(lines)} != expected {len(orig_lines)}",
                    f"source={orig_lines!r}",
                    f"returned={lines!r}",
                    "coercing to expected line count",
                ]),
                level=1,
            )
            if len(lines) < len(orig_lines):
                lines = lines + ["" for _ in range(len(orig_lines) - len(lines))]
            else:
                lines = lines[:len(orig_lines)]
        translated.append(lines)

    return translated, glossary_out, usage_data

# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_srt", help="Path to input .srt")
    ap.add_argument("output_srt", nargs="?", help="Path to output .srt (optional; defaults to input name + .<lang>.srt)")
    ap.add_argument("--model", default="gpt-5.2", help="Model name (e.g. gpt-5.2, gpt-4o-mini)")
    ap.add_argument("--chunk-size", type=int, default=300, help="Subtitle blocks per API call")
    ap.add_argument("--source-lang-hint", default="English", help="Hint for source language(s)")
    ap.add_argument("--target-lang", default="Bulgarian", help="Target language")
    ap.add_argument("--debug", type=int, choices=[0, 1, 2], default=0, help="Debug level: 0=minimal, 1=progress/usage, 2=include request payloads")
    ap.add_argument("--dry-run", action="store_true", help="Parse and validate only; no API calls or output file")
    args = ap.parse_args()

    global DEBUG_LEVEL
    DEBUG_LEVEL = args.debug

    log(
        f"Starting translation; model={args.model}; target={args.target_lang}; chunk_size={args.chunk_size}; dry_run={args.dry_run}; debug={args.debug}",
        level=0,
    )

    input_path = Path(args.input_srt)
    if input_path.suffix.lower() != ".srt":
        raise ValueError("Input file must have .srt extension")

    lang_code = derive_lang_code(args.target_lang)

    if args.output_srt:
        output_path = Path(args.output_srt)
        if output_path.suffix.lower() != ".srt":
            output_path = output_path.with_suffix(".srt")
    else:
        output_path = input_path.with_name(f"{input_path.stem}.{lang_code}.srt")

    log(f"Input file: {input_path}", level=0)
    log(f"Output file: {output_path}", level=0)

    raw = input_path.read_text(encoding="utf-8", errors="strict")
    blocks, preamble = parse_srt_preserve(raw)
    total_blocks = len(blocks)
    log(f"Parsed SRT: {total_blocks} blocks", level=1)

    # Prepare chunks
    all_text_shapes: List[List[str]] = []
    for b in blocks:
        # Strip newline endings for model input; we re-add originals later.
        # Keep empty blocks as empty list.
        clean = [ln.rstrip("\r\n") for ln in b.text_lines]
        all_text_shapes.append(clean)

    if args.dry_run:
        log(f"Dry run: first block lines={len(all_text_shapes[0]) if all_text_shapes else 0}", level=0)
        log("Dry run complete; no output written or API calls made.", level=0)
        return

    client = OpenAI()
    log("OpenAI client initialized", level=1)

    total_input_tokens = 0
    total_output_tokens = 0
    total_tokens = 0
    current_glossary: List[str] = []

    # Translate chunk by chunk
    for start in range(0, len(blocks), args.chunk_size):
        end = min(start + args.chunk_size, len(blocks))
        chunk = all_text_shapes[start:end]

        log(f"Translating blocks {start+1}..{end}", level=1)

        translated_chunk, glossary_out, usage = translate_chunk(
            client=client,
            model=args.model,
            chunk_text_lines=chunk,
            prior_glossary=current_glossary,
            source_lang_hint=args.source_lang_hint,
            target_lang=args.target_lang,
        )

        current_glossary = glossary_out
        if DEBUG_LEVEL >= 1:
            log(f"Glossary items after chunk: {len(current_glossary)}", level=1)

        # Track token usage if present
        chunk_input = usage.get("input_tokens") if isinstance(usage, dict) else None
        chunk_output = usage.get("output_tokens") if isinstance(usage, dict) else None
        chunk_total = usage.get("total_tokens") if isinstance(usage, dict) else None
        if chunk_input is not None:
            total_input_tokens += chunk_input
        if chunk_output is not None:
            total_output_tokens += chunk_output
        if chunk_total is not None:
            total_tokens += chunk_total

        if DEBUG_LEVEL >= 1:
            if chunk_input is not None or chunk_output is not None or chunk_total is not None:
                log(f"Tokens this chunk: input={chunk_input} output={chunk_output} total={chunk_total}", level=1)
            else:
                log("Tokens this chunk: unavailable (usage not returned)", level=1)

        # Write translations back into blocks, preserving each original line ending
        for bi, new_lines in enumerate(translated_chunk, start=start):
            original_line_endings = [re.search(r"(\r?\n)$", ln).group(1) for ln in blocks[bi].text_lines] if blocks[bi].text_lines else []
            rebuilt_lines: List[str] = []
            for j, txt in enumerate(new_lines):
                eol = original_line_endings[j] if j < len(original_line_endings) else "\n"
                rebuilt_lines.append(txt + eol)
            blocks[bi].text_lines = rebuilt_lines

        log(f"Finished blocks {start+1}..{end}", level=1)
        log(f"Progress: {end}/{total_blocks} blocks translated", level=0)

    out = rebuild_srt(blocks, preamble)
    output_path.write_text(out, encoding="utf-8")
    log(f"Done -> {output_path}", level=0)

    if DEBUG_LEVEL >= 1:
        if total_input_tokens or total_output_tokens or total_tokens:
            log(f"Total tokens: input={total_input_tokens or 'n/a'} output={total_output_tokens or 'n/a'} total={total_tokens or 'n/a'}", level=1)
        else:
            log("Total tokens: unavailable (usage not returned)", level=1)

if __name__ == "__main__":
    main()
