"""Build the Konkle-word asset library from screened Objaverse picks plus procedural fills."""

import hashlib
import json
import multiprocessing
from pathlib import Path
import random
import subprocess
import tempfile


HERE = Path(__file__).parent
VOCABULARY = HERE / "vocabulary.json"
MANIFEST = HERE / "assets.json"
PROCEDURAL = ("balloon", "bill", "button", "tape", "bagel")
TARGET = 20
OBJAVERSE = "https://huggingface.co/datasets/allenai/objaverse/resolve/main/"


def load_vocabulary():
    vocabulary = json.loads(VOCABULARY.read_text())
    if vocabulary["schema_version"] != 1:
        raise ValueError("Unsupported vocabulary")
    return vocabulary


def bill_texture(seed, path):
    from PIL import Image, ImageDraw

    rng = random.Random(seed)
    width, height = 512, 218
    base = [rng.randint(120, 210) for _ in range(3)]
    tint = rng.randrange(3)
    base[tint] = min(235, base[tint] + 40)
    image = Image.new("RGB", (width, height), tuple(base))
    draw = ImageDraw.Draw(image)
    dark = tuple(max(0, c - 110) for c in base)
    mid = tuple(max(0, c - 55) for c in base)
    # Guilloche-like line field, a border, a portrait oval, seals and denomination numerals.
    for k in range(0, width + height, 6):
        draw.line([(k, 0), (k - height, height)], fill=mid, width=1)
    draw.rectangle([8, 8, width - 9, height - 9], outline=dark, width=6)
    draw.rectangle([20, 20, width - 21, height - 21], outline=mid, width=2)
    cx = rng.choice((width // 2, width // 3, 2 * width // 3))
    draw.ellipse([cx - 60, 34, cx + 60, height - 34], fill=tuple(min(255, c + 30) for c in base), outline=dark, width=4)
    draw.ellipse([cx - 28, 60, cx + 28, 130], fill=dark)
    draw.pieslice([cx - 50, 120, cx + 50, 220], 180, 360, fill=dark)
    for x in (70, width - 70):
        draw.ellipse([x - 30, height // 2 - 30, x + 30, height // 2 + 30], outline=dark, width=3)
    value = str(rng.choice((1, 5, 10, 20, 50, 100)))
    for x, y in ((30, 26), (width - 70, 26), (30, height - 56), (width - 70, height - 56)):
        draw.text((x, y), value, fill=dark, font_size=28)
    image.save(path)


def _entries(selected, vocabulary, glbs, textures):
    words = vocabulary["words"]
    entries = {}
    for word in words:
        picks = selected.get(word, [])
        rows = [{"id": item["uid"], "source": "objaverse", "uid": item["uid"], "name": item["name"],
                 "author": item.get("author"), "license": item["license"],
                 "url": OBJAVERSE + item["path"], "path": str(glbs / f"{item['uid']}.glb"),
                 "size_m": words[word]["size_m"]} for item in picks]
        if word in PROCEDURAL:
            for index in range(max(0, TARGET - len(rows))):
                seed = int(hashlib.sha256(f"{word}:{index}".encode()).hexdigest()[:8], 16)
                row = {"id": f"proc_{word}_{index:02d}", "source": "procedural", "seed": seed,
                       "license": "generated", "size_m": words[word]["size_m"]}
                if word == "bill":
                    row["image"] = str(textures / f"{row['id']}.png")
                    bill_texture(seed, row["image"])
                rows.append(row)
        entries[word] = rows
    return entries


def _run_blender(blender, job_path, log_path):
    with open(log_path, "w") as log:
        return subprocess.run([blender, "-b", "--factory-startup", "--python-exit-code", "1", "-P",
                               str(HERE / "library_blender.py"), "--", str(job_path)],
                              stdout=log, stderr=subprocess.STDOUT).returncode


def _assign_splits(rows, fractions):
    rows = sorted(rows, key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest())
    if len(rows) < 4:
        raise ValueError("A word needs at least four instances for train/dev/test splits")
    # Fixed counts by hash order: ~15% dev and test each, at least one, the rest train.
    count = {split: max(1, round(fractions[split] * len(rows))) for split in ("dev", "test")}
    for index, row in enumerate(rows):
        row["split"] = "test" if index < count["test"] else "dev" if index < count["test"] + count["dev"] else "train"
    return rows


def build(selected_path, library, blender, workers=10, overrides=None, reuse=False):
    vocabulary = load_vocabulary()
    selected = json.loads(Path(selected_path).read_text())
    library = Path(library).resolve()
    (library / "textures").mkdir(parents=True, exist_ok=True)
    glbs = library.parent / "glbs"
    overrides = overrides or {}
    entries = _entries(selected, vocabulary, glbs, library / "textures")
    jobs = []
    with tempfile.TemporaryDirectory(prefix="nursery-library-") as temporary:
        for word, rows in entries.items():
            for row in rows:
                if row["id"] in overrides:
                    row.update(overrides[row["id"]])
            rows = [row for row in rows if not row.get("excluded")]
            job = Path(temporary) / f"{word}.json"
            job.write_text(json.dumps({"word": word, "output": str(library), "assets": rows}))
            jobs.append((blender, job, library / f"{word}.log"))
        jobs = [job for job in jobs if not (library / f"{job[1].stem}.blend").exists()] if reuse else jobs
        with multiprocessing.get_context("spawn").Pool(workers) as pool:
            codes = pool.starmap(_run_blender, jobs)
    failed = [job[1].stem for job, code in zip(jobs, codes) if code]
    if failed:
        raise RuntimeError(f"Blender library build failed for: {failed}")
    manifest = {"schema_version": 1, "target_per_word": TARGET, "words": {}}
    for word, rows in entries.items():
        results = {r["id"]: r for r in json.loads((library / f"{word}.json").read_text())}
        kept = []
        for row in rows:
            result = results.get(row["id"])
            if row.get("excluded"):
                continue
            if result is None or result["status"] != "ok":
                continue
            kept.append({key: row[key] for key in ("id", "source", "uid", "name", "author", "license", "url", "seed")
                         if key in row} | {"faces": result["faces"], "textured": result["textured"], "extents_m": [round(x, 4) for x in result["extents_m"]],
                                           "source_scale": result["source_scale"], **({"rotation": row["rotation"]} if "rotation" in row else {})})
        manifest["words"][word] = _assign_splits(kept, vocabulary["instance_split"])
    MANIFEST.write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def contact_sheet(library, output):
    from PIL import Image, ImageDraw

    library = Path(library)
    manifest = json.loads(MANIFEST.read_text())
    size = 96
    words = list(manifest["words"])
    columns = 24
    sheet = Image.new("RGB", (columns * size + 90, len(words) * size), "white")
    draw = ImageDraw.Draw(sheet)
    for row, word in enumerate(words):
        draw.text((2, row * size + 40), word, fill="black")
        for column, item in enumerate(manifest["words"][word][:columns]):
            path = library / "thumbs" / f"{item['id']}.png"
            if path.exists():
                image = Image.open(path).convert("RGB").resize((size, size))
                sheet.paste(image, (90 + column * size, row * size))
            color = {"train": "black", "dev": "blue", "test": "red"}[item["split"]]
            draw.text((92 + column * size, row * size + 2), item["split"][0], fill=color)
    sheet.save(output, quality=85)
    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected", required=True)
    parser.add_argument("--library", required=True)
    parser.add_argument("--blender", default="/opt/homebrew/bin/blender")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--overrides", help="JSON of per-id fixes: rotation [x_deg, y_deg] or excluded")
    parser.add_argument("--reuse", action="store_true", help="Keep existing per-word libraries")
    args = parser.parse_args()
    overrides = json.loads(Path(args.overrides).read_text()) if args.overrides else None
    build(args.selected, args.library, args.blender, args.workers, overrides, args.reuse)
    print(contact_sheet(args.library, Path(args.library) / "contact_sheet.jpg"))
