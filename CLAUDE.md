# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
python -m pip install -U openai
export OPENAI_API_KEY="sk-..."   # or $env:OPENAI_API_KEY on PowerShell
```

No `requirements.txt` exists; the only external dependency is the `openai` SDK.

## Common commands

```bash
# Validate SRT parsing without API calls
python translator.py input.srt --dry-run

# Translate (output written to input.bg.srt by default)
python translator.py input.srt --target-lang Bulgarian --model gpt-5.5 --chunk-size 300 --debug 1

# Explicit output path
python translator.py input.srt output.srt --target-lang Spanish
```

There are no tests, linter config, or build steps.

## Architecture

Everything lives in the single file [translator.py](translator.py). The pipeline is:

1. **Parse** — `parse_srt_preserve(raw)` splits raw SRT text into `SrtBlock` dataclasses, recording the exact `number_line`, `timestamp_line`, `text_lines`, and `separator_after` (blank lines) verbatim including `\r\n` vs `\n`.
2. **Translate** — `translate_chunk()` calls the **OpenAI Responses API** (`client.responses.create`, not Chat Completions) with `text.format.type = "json_schema"` (strict structured output). The schema enforces `blocks[N].lines` to have the same block count per call. A rolling `glossary` list is passed forward across chunks for name/tone consistency.
3. **Rebuild** — `rebuild_srt(blocks, preamble)` reassembles blocks in place, re-attaching original line endings extracted from `text_lines`.

### Critical invariants

- Block count and per-block line count must never change between input and output. The model is constrained by `minItems`/`maxItems` in the schema; any remaining drift is coerced (truncated or padded with `""`) and logged at `debug >= 1`.
- Original line endings (`\r\n` / `\n`) are extracted from `text_lines` before sending to the model and reattached after — the model only ever sees/returns plain text strings.
- `temperature=0` is sent for deterministic output; models that reject it (e.g. gpt-5.5) get a one-time 400, after which `_create_response` omits it for the rest of the run.

### Debug levels

| Level | What you get |
|-------|-------------|
| 0 | Start message, output path, per-chunk progress, final path |
| 1 | Parsed block count, glossary size, token usage per chunk and total |
| 2 | Full request instructions + JSON payload for every API call |
