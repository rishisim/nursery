"""Write contact sheets with recorded hand-contact cues overlaid."""

import json
from pathlib import Path
import subprocess


def preview(episodes, output):
    import pyarrow.parquet as pq
    from PIL import Image, ImageDraw

    episodes, output = Path(episodes).resolve(), Path(output).absolute()
    parent = output.parent
    while not parent.exists():
        parent = parent.parent
    if subprocess.run(["git", "-C", str(parent), "check-ignore", "--quiet", str(output)],
                      check=False).returncode:
        raise ValueError("Preview must be inside an ignored repository directory")
    if output.exists():
        raise ValueError("Preview output already exists")
    rows = pq.read_table(episodes / "frames.parquet").to_pylist()
    subjects = sorted({row["video_id"] for row in rows})
    selected = []
    for subject in subjects:
        frames = [row for row in rows if row["video_id"] == subject]
        step = max(1, len(frames) // 8)
        selected.extend(frames[::step][:8])
    if not selected:
        raise ValueError("No frames to preview")
    sheet = Image.new("RGB", (4 * 320, ((len(selected) + 3) // 4) * 280), "white")
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(selected):
        image = Image.open(episodes / row["image"]).convert("RGB")
        image.thumbnail((320, 240))
        x, y = index % 4 * 320, index // 4 * 280
        sheet.paste(image, (x, y))
        cue = f"L={int(row['contact_left'])} R={int(row['contact_right'])} {row['held_category'] or '-'}"
        visible = ", ".join(obj["category"] for obj in json.loads(row["objects"]) if obj["visible_px"] > 0)
        draw.text((x + 4, y + 240), f"{row['video_id']} t={row['frame_idx']} {cue}", fill="black")
        draw.text((x + 4, y + 256), visible, fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
    print(output)
    return output
