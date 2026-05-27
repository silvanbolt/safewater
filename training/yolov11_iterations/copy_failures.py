#!/usr/bin/env python3
"""Copy top-3 failure images into a single folder for easy download."""

import re
import shutil
from pathlib import Path

VAL_DIR = Path("/work/courses/dslab/team7/multi-national/yolo_classification_oversample/val")
FAILURES_TXT = Path("/work/courses/dslab/team7/multi-national/top2_failures.txt")
OUT_DIR = Path("/work/courses/dslab/team7/multi-national/top2_failures_images")

OUT_DIR.mkdir(exist_ok=True)

image_pat = re.compile(r"^Image:\s+(\S+)")
class_pat = re.compile(r"^True class:\s+(\S+)")

current_image = None
current_class = None
copied = 0
missing = 0

with FAILURES_TXT.open() as f:
    for line in f:
        line = line.strip()
        m = image_pat.match(line)
        if m:
            current_image = m.group(1)
            current_class = None
            continue
        m = class_pat.match(line)
        if m and current_image:
            current_class = m.group(1)
            src = VAL_DIR / current_class / current_image
            # Destination: <TrueClass>__<filename> so it's self-describing
            dst = OUT_DIR / f"{current_class}__{current_image}"
            if src.exists():
                shutil.copy2(src, dst)
                copied += 1
            else:
                print(f"MISSING: {src}")
                missing += 1
            current_image = None
            current_class = None

print(f"Done — copied {copied} images, {missing} missing → {OUT_DIR}")
