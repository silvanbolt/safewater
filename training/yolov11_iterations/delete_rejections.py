from pathlib import Path

VAL = Path("/work/courses/dslab/team7/multi-national/yolo_classification_oversample/val")
REJECTIONS = Path("/work/courses/dslab/team7/multi-national/rejections.txt")

deleted = []
missing = []

for line in REJECTIONS.read_text().splitlines():
    line = line.strip()
    if not line:
        continue

    # Format: ClassName__number.jpg  →  val/ClassName/number.jpg
    class_name, rest = line.split("__", 1)
    stem = Path(rest).stem

    jpg = VAL / class_name / f"{stem}.jpg"
    npy = VAL / class_name / f"{stem}.npy"

    if jpg.exists():
        jpg.unlink()
        deleted.append(str(jpg))
        if npy.exists():
            npy.unlink()
            deleted.append(str(npy))
    else:
        missing.append(line)

print(f"Deleted {len(deleted)} files from {len(deleted) - len([f for f in deleted if f.endswith('.npy')])} images.")
if missing:
    print(f"\nNot found ({len(missing)}):")
    for m in missing:
        print(f"  {m}")
else:
    print("All entries found and deleted successfully.")
