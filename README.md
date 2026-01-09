# Subtitle Translator

Single-file SRT translator that preserves exact formatting (line endings, blank lines, numbering, timestamps) while translating text via OpenAI. It runs in chunks for large files, maintains a rolling glossary across calls, and supports adjustable debug output.

## Features
- Preserves SRT numbering, timestamps, blank lines, and original line endings.
- Chunked translation for large files; structured JSON output with coercion if counts drift.
- Rolling, model-generated glossary carried across chunks for consistent names/tone.
- Default output naming: if no output is given, writes alongside the input as `<input>.<langcode>.srt`.
- Debug levels: 0 minimal (with progress), 1 progress + token usage, 2 includes per-request payloads.
- Dry-run mode to validate parsing without API calls or file writes.

## Requirements
- Python 3.9+
- `openai` Python SDK
- An OpenAI API key (`OPENAI_API_KEY`)

Install dependencies:
```bash
py -m pip install openai
```

Set your API key (PowerShell example):
```powershell
$env:OPENAI_API_KEY="sk-..."
```

## Usage
Dry run (parse only):
```bash
python translator.py input.srt --dry-run
```

Translate (default output next to input, e.g., `input.bg.srt`):
```bash
python translator.py input.srt --target-lang Bulgarian --model gpt-5.2 --chunk-size 300 --debug 1
```

Specify an explicit output path:
```bash
python translator.py input.srt output.srt --target-lang Spanish
```

### CLI options
- `input_srt` (positional): source `.srt` file (required; must end with `.srt`).
- `output_srt` (positional, optional): destination `.srt`; defaults to `<input>.<langcode>.srt`.
- `--model`: model name (default `gpt-5.2`).
- `--chunk-size`: subtitle blocks per API call (default 300).
- `--source-lang-hint`: hint for source language(s) (default `English`).
- `--target-lang`: target language (default `Bulgarian`).
- `--debug`: 0=minimal+progress, 1=progress+usage, 2=request payloads.
- `--dry-run`: parse/validate only; no API calls or output writes.

## Notes and behavior
- Enforces same block/line counts; trims/pads when the model drifts and logs at debug≥1.
- Rolling glossary: each chunk returns a concise glossary/consistency list; reused on the next chunk.
- Token usage logging depends on the API returning `usage` fields.
- Line endings are preserved; missing endings default to `\n` when rebuilding.

## License
MIT. See `LICENSE`.
