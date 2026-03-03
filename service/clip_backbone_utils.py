from __future__ import annotations

import re


def infer_clip_input_resolution(clip_type: str | None) -> int:
    """Infer CLIP image input resolution (square side length).

    This repo mainly uses OpenAI CLIP backbone names (e.g. RN50x64).
    Some backbones are resolution-sensitive (notably RN50x64 -> 448).

    Rules:
    - If clip_type contains an explicit suffix like "@336px", use that.
    - Otherwise, use known defaults, and fall back to 224.
    """

    if not clip_type:
        return 224

    name = str(clip_type).strip()

    m = re.search(r"@(\d+)px\b", name)
    if m:
        return int(m.group(1))

    defaults: dict[str, int] = {
        "RN50x64": 448,
        # Most OpenAI CLIP backbones default to 224
        "RN50": 224,
        "RN101": 224,
        "RN50x4": 224,
        "RN50x16": 224,
        "ViT-B/32": 224,
        "ViT-B/16": 224,
        "ViT-L/14": 224,
    }

    return defaults.get(name, 224)
