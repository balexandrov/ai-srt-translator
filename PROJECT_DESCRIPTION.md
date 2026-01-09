# AI SRT Translator

Single-file tool (`translator.py`) that translates `.srt` subtitles with OpenAI while preserving exact formatting (line endings, blank lines, numbering, timestamps). It works in chunks for large files, maintains a rolling glossary across calls, and can emit minimal or detailed debug logs.

## What it does
- Parses `.srt` while keeping every separator and original line ending intact.
- Translates subtitle text via OpenAI in chunked batches; preserves block and line counts.
- Carries a model-generated glossary/consistency list between chunks to keep tone, names, and recurring terms stable.
- Restores the output `.srt` with the original spacing/format.

## Inputs and outputs
- Input must be `.srt`.
- Output defaults to the same folder as input, named `<input>.XX.srt` (XX derived from `--target-lang`) when not provided.

## CLI usage
```bash
python translator.py input.srt [output.srt] \
  --target-lang Bulgarian \
  --model gpt-5.2 \
  --chunk-size 300 \
  --source-lang-hint English \
  --debug 1
```

### Important options
- `input_srt` (positional): source `.srt` file.
- `output_srt` (positional, optional): destination `.srt`; defaults to `<input>.<langcode>.srt`.
- `--target-lang`: target language (default `Bulgarian`).
- `--source-lang-hint`: hint for source language(s).
- `--model`: OpenAI model name (default `gpt-5.2`).
- `--chunk-size`: subtitle blocks per API call (default 300).
- `--debug`: 0=minimal with progress, 1=progress + usage, 2=include per-request payloads.
- `--dry-run`: parse/validate only; no API calls or file writes.

## Rolling glossary behavior
- Each chunk request includes the prior glossary; the model returns updated glossary lines.
- Glossary is unstructured text lines capturing names, places, recurring terms, and style decisions; guidance size ~10–30 items.
- If no new items appear, the model repeats the prior glossary unchanged.

## Reliability safeguards
- Structured JSON schema enforces block counts; out-of-spec responses are trimmed/padded.
- Line-count mismatches within a block are coerced with a warning.
- Original line endings are re-applied when rebuilding the `.srt`.

## Prerequisites
- Python 3.9+
- `openai` package (`py -m pip install openai`)
- `OPENAI_API_KEY` available to the process (env var, inline in shell, or wrapper script).

## Context menu (Windows, optional)
You can register a shell verb that runs `translator.py` on right-clicked `.srt` files; it works best in the classic context menu (Win11 shows it under “Show more options” unless classic menu is restored). Ensure the command points at your script path and supplies `OPENAI_API_KEY`.
