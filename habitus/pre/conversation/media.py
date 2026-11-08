"""只有媒体、没有文字的工具结果：认形状、给描述与原载荷的大小和摘要值（不调用模型）。

形状与插件出站剥媒体（``plugins/memory-plugin-shared/lib/media.mjs``）认的一致，两侧由共享 fixture
``media-placeholder.json`` 守：适配器用它给消息算身份（经不经插件同一个 ``message_id``），保留策略用它把结果存成
OMITTED（经不经插件同样的描述、大小与摘要值）。
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

#: 纯媒体段的 type 值（与插件出站剥媒体认的形状一致）。
MEDIA_TYPES = frozenset({"image", "input_image", "image_url", "document", "file", "input_file", "audio", "input_audio"})


@dataclass(frozen=True)
class MediaOnlyDescription:
    """只有媒体、没有文字的工具结果：描述文字与原载荷的字节数、sha256。

    原载荷指转写里那段载荷文本本身（base64 文本或整条 ``data:`` URL）的 UTF-8 字节，与插件占位的
    ``omitted_bytes`` / ``omitted_sha256`` 同一口径；多段时字节数相加，sha256 是各段 sha256（十六进制）按顺序
    以换行连接后再取的 sha256。
    """

    text: str
    original_size_bytes: int
    original_sha256: str


def describe_media_only(content: object) -> MediaOnlyDescription | None:
    """内容是只有媒体、没有文字的段列表时给出描述与原载荷大小、摘要，否则 None。

    插件已把载荷换成占位（带 ``omitted_bytes`` 与 ``omitted_sha256``）时按占位记，与不经插件逐值相同。
    """

    if isinstance(content, str | bytes) or not isinstance(content, Sequence) or not content:
        return None
    parts: list[str] = []
    sizes: list[int] = []
    digests: list[str] = []
    for item in content:
        if not isinstance(item, Mapping) or isinstance(item.get("text"), str):
            return None
        media = _media_facts(item)
        if media is None:
            return None
        kind, media_type, size, digest = media
        detail = "，".join(value for value in (media_type, f"载荷 {size:,} 字节") if value)
        parts.append(f"{kind}（{detail}）")
        sizes.append(size)
        digests.append(digest)
    combined = digests[0] if len(digests) == 1 else hashlib.sha256("\n".join(digests).encode("ascii")).hexdigest()
    return MediaOnlyDescription(
        f"工具结果只有媒体、没有文字，原始载荷未保存：{'；'.join(parts)}。",
        sum(sizes),
        combined,
    )


def _media_facts(item: Mapping[str, Any]) -> tuple[str, str | None, int, str] | None:
    # 没有 type 的段只凭载荷形状（base64 source、image_url 的 data URL、带 mime 的 data URL）认作媒体，
    # 其余不当媒体；描述里记成 file，与插件占位缺 type 时写的一致。
    declared = _text_or_none(item.get("type"))
    kind = declared or "file"
    source = item.get("source")
    if isinstance(source, Mapping) and source.get("type") == "base64" and isinstance(data := source.get("data"), str):
        return kind, _text_or_none(source.get("media_type")), *_payload(data)
    image_url = item.get("image_url")
    if isinstance(image_url, Mapping):
        image_url = image_url.get("url")
    if isinstance(image_url, str) and image_url.startswith("data:"):
        return kind, _data_url_media_type(image_url), *_payload(image_url)
    url = item.get("url")
    if (
        isinstance(url, str)
        and url.startswith("data:")
        and (declared in MEDIA_TYPES or isinstance(item.get("mime"), str))
    ):
        return kind, _text_or_none(item.get("mime")) or _data_url_media_type(url), *_payload(url)
    if declared not in MEDIA_TYPES:
        return None
    media_type = next(
        (text for text in (_text_or_none(item.get(key)) for key in ("mimeType", "media_type", "mime")) if text),
        None,
    )
    if isinstance(data := item.get("data"), str):
        return kind, media_type, *_payload(data)
    file = item.get("file")
    if isinstance(file, Mapping) and isinstance(encoded := file.get("base64"), str):
        return kind, _text_or_none(file.get("type")), *_payload(encoded)
    omitted = item.get("omitted_bytes")
    digest = item.get("omitted_sha256")
    if (
        isinstance(omitted, int)
        and not isinstance(omitted, bool)
        and omitted >= 0
        and isinstance(digest, str)
        and re.fullmatch(r"[0-9a-f]{64}", digest)
    ):
        return kind, media_type, omitted, digest
    return None


def _payload(value: str) -> tuple[int, str]:
    encoded = value.encode("utf-8")
    return len(encoded), hashlib.sha256(encoded).hexdigest()


def _text_or_none(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _data_url_media_type(value: str) -> str | None:
    match = re.match(r"data:([^;,]+)[;,]", value)
    return match.group(1) if match else None


__all__ = [
    "MEDIA_TYPES",
    "MediaOnlyDescription",
    "describe_media_only",
]
