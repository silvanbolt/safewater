# Image Review Tool

Manually review images and flag ones that are mislabelled. Results are saved to `rejections.txt`.

## How it works

- Shows each image in `top2_binary_confusion_images/` one by one
- Extracts the class label from the filename (e.g. `Borehole_Tubewell__14086.jpg` → `"Borehole Tubewell"`)
- Asks `Is this a "Borehole Tubewell"? [y/n/b/q]` in the terminal
- Any image answered with `n` is written to `rejections.txt` in the project root
- Pressing `b` goes back to the previous image to change your decision
- Pressing `q` quits early but still saves all decisions collected so far

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (replaces pip/venv)

### Install uv

**Mac / Linux**

```bash
curl -Lsf https://astral.sh/uv | sh
```

**Windows**

```powershell
winget install astral-sh.uv
```

## Running

```bash
uv run python review_images.py
```

uv reads `pyproject.toml` and `uv.lock`, creates a virtual environment, and installs the exact dependency versions automatically. No manual `pip install` needed.

## Sharing with others

Zip the following files and send them:

```
valstar/
├── review_images.py
├── pyproject.toml
├── uv.lock
└── top2_binary_confusion_images/
```

Do **not** include the `.venv` folder — it is large and gets recreated automatically on first run.

The recipient only needs Python 3.12+ and uv installed, then runs the same command above.

## Output

`rejections.txt` is saved in the project root (next to `review_images.py`) and contains one filename per line for every image that was answered with `n`.
