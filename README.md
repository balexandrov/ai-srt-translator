# Subtitle Translator

A simple SRT subtitle translator that preserves exact formatting (line endings, blank lines) while translating text via OpenAI models. Supports glossary guidance, chunked requests, dry-run validation, and token usage logging.

## Features
- Preserves SRT numbering, timestamps, blank lines, and original line endings.
- Chunked translation to handle large files.
- Optional glossary prompt guidance (`source = target` per line).
- Dry-run mode to validate parsing without API calls or file writes.
- Token usage logging per chunk and totals when the API returns usage data.
- Graceful handling when the model returns the wrong line count (pads/truncates with a warning).

## Requirements
- Python 3.9+
- `openai` Python SDK
- An OpenAI API key (`OPENAI_API_KEY`)

Install dependencies:
```bash
py -m pip install openai
```

Set your API key (PowerShell):
```powershell
$env:OPENAI_API_KEY="sk-..."
```

## Usage
Dry run (no API calls, no output file):
```bash
python translator.py input.srt output.srt --dry-run
```

Translate:
```bash
python translator.py input.srt output.srt \
  --model gpt-5.2 \
  --target-lang Bulgarian \
  --chunk-size 300 \
  --glossary glossary.txt
```

### CLI options
- `input_srt` (positional): path to input `.srt`.
- `output_srt` (positional): path to write translated `.srt`.
- `--model`: model name (default `gpt-5.2`).
- `--chunk-size`: subtitle blocks per API call (default 300).
- `--glossary`: path to glossary file.
- `--source-lang-hint`: hint for source language(s) (default `English`).
- `--target-lang`: target language (default `Bulgarian`).
- `--dry-run`: parse/validate only; no API calls or output writes.

### Glossary format
File with one mapping per line:
```
source term = target term
```
Blank lines and lines starting with `#` are ignored.

Example (`glossary.txt`):
```
OFPRA = OFPRA
UFDG = UFDG
phone units = airtime
```

## Notes and behavior
- Keeps the same number of blocks and lines per block; if the model returns fewer/more lines for a block, the script logs a warning and pads/truncates to match the original line count.
- Token usage logging depends on the model/API returning `usage` fields.
- Line endings are preserved from the input file; any missing endings default to `\n` when rebuilding.

## Developing
- Run type checks/linters as desired; no extra tooling is required.
- The code is single-file (`translator.py`) for simplicity.

## License
MIT. See `LICENSE`.
