#!/usr/bin/env python3
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

IMAGE_DIR = Path(__file__).parent / "top2_binary_confusion_images"
REJECTIONS_FILE = Path(__file__).parent / "rejections.txt"

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}


def extract_class_name(filename: str) -> str:
    stem = Path(filename).stem
    match = re.match(r"^(.*?)__?\d+$", stem)
    if match:
        return match.group(1).replace("_", " ")
    return stem.replace("_", " ")


class QuitRequested(Exception):
    pass


def ask_user(class_name: str, can_go_back: bool):
    options = "y/n/b/q" if can_go_back else "y/n/q"
    while True:
        answer = input(f'  Is this a "{class_name}"? [{options}]: ').strip().lower()
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        if answer in ("b", "back"):
            if can_go_back:
                return "back"
            print("  Already at the first image.")
        elif answer in ("q", "quit"):
            raise QuitRequested
        else:
            print(f"  Please enter {options}.")


def main():
    images = sorted(
        f for f in IMAGE_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in EXTENSIONS
    )

    if not images:
        print(f"No images found in {IMAGE_DIR}")
        sys.exit(1)

    # decisions[i] = True (accepted) or False (rejected)
    decisions: dict[int, bool] = {}

    def save():
        with open(REJECTIONS_FILE, "w") as f:
            for idx in sorted(decisions.keys()):
                if not decisions[idx]:
                    f.write(images[idx].name + "\n")

    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.canvas.manager.set_window_title("Image Review")

    i = 0
    try:
        while i < len(images):
            img_path = images[i]
            class_name = extract_class_name(img_path.name)

            img = mpimg.imread(str(img_path))
            ax.clear()
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(f"[{i + 1}/{len(images)}] {img_path.name}", fontsize=10)
            fig.tight_layout()
            plt.pause(0.1)

            print(f"\n[{i + 1}/{len(images)}] {img_path.name}")
            result = ask_user(class_name, can_go_back=(i > 0))

            if result == "back":
                decisions.pop(i - 1, None)
                save()
                i -= 1
                continue

            decisions[i] = result
            if not result:
                print("  Marked as rejection.")
            save()
            i += 1

    except QuitRequested:
        print("Quitting early.")

    plt.close(fig)
    rejection_count = sum(1 for v in decisions.values() if not v)
    print(f"\nDone. {rejection_count} rejection(s) saved to {REJECTIONS_FILE}")


if __name__ == "__main__":
    main()
