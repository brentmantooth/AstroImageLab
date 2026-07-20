"""Extract base64-embedded images from an HTML report into a sibling folder.

Rewrites every `data:image/...;base64,...` URI in the HTML with a relative
path to an extracted image file. Given `<stem>.html`, writes
`<stem>_Extracted.html` and `<stem>_images/figure_NNN.<ext>` next to it.

Run standalone from the repo root:
    python tools/extract_images.py path/to/report.html

Also available from the GUI's Tools menu ("Extract Images from HTML…") via
`extract_images()`.
"""

from __future__ import annotations

import base64
import re
import sys
from pathlib import Path

_PATTERN = re.compile(
    r'data:image/(?P<ext>png|jpeg|jpg|gif|webp);base64,(?P<data>[A-Za-z0-9+/=\n\r]+)'
)


def extract_images(input_path: Path) -> tuple[Path, Path, int]:
    """Extract embedded base64 images from `input_path` into a sibling folder.

    Returns (output_html_path, image_dir, image_count).
    """
    input_path = Path(input_path)
    output_html = input_path.with_name(f"{input_path.stem}_Extracted{input_path.suffix}")
    image_dir = input_path.with_name(f"{input_path.stem}_images")
    image_dir.mkdir(exist_ok=True)

    html = input_path.read_text(encoding="utf-8")

    count = 0

    def replace(match: re.Match) -> str:
        nonlocal count
        ext = match.group("ext")
        if ext == "jpeg":
            ext = "jpg"

        count += 1
        filename = f"figure_{count:03d}.{ext}"
        path = image_dir / filename

        image_bytes = base64.b64decode(match.group("data"))
        path.write_bytes(image_bytes)

        return f"{image_dir.name}/{filename}"

    html = _PATTERN.sub(replace, html)
    output_html.write_text(html, encoding="utf-8")

    return output_html, image_dir, count


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("input.html")
    out_path, img_dir, n = extract_images(target)
    print(f"Extracted {n} images to {img_dir}")
    print(f"Wrote {out_path}")
