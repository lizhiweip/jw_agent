"""Convert Word legal documents into one-article-per-line UTF-8 text files."""

from __future__ import annotations

import argparse
import json
import re
from datetime import date, datetime
from pathlib import Path

import docx2txt


ARTICLE_RE = re.compile(
    r"(?m)^[ \t\u3000]*(第[ \t\u3000]*[零〇一二三四五六七八九十百千万两\d]+"
    r"[ \t\u3000]*条)[ \t\u3000]*"
)
STRUCTURE_LINE_RE = re.compile(
    r"^第[零〇一二三四五六七八九十百千万两\d]+[编章节][ \t\u3000]?.*$"
)
METADATA_RE = re.compile(
    r"制定机关|公布日期|施行日期|生效日期|时效性|效力位阶|法规类别|"
    r"主席令|委员会令|公告|通过|修订|修正|发布|公布|施行"
)


def clean_title(path: Path) -> str:
    title = re.sub(r"^\d+(?:\.\d+)?[-.、]?", "", path.stem).strip()
    return title or path.stem


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    text = re.sub(r"[ \t\u3000]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def compact_lines(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    while lines and STRUCTURE_LINE_RE.fullmatch(lines[-1]):
        lines.pop()
    return " ".join(lines)


def extract_articles(text: str, law_name: str) -> list[str]:
    matches = list(ARTICLE_RE.finditer(text))
    articles: list[str] = []

    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        article_number = re.sub(r"[ \t\u3000]", "", match.group(1))
        body = compact_lines(text[match.end() : end])
        if body:
            articles.append(f"《{law_name}》{article_number} {body}")

    return articles


def extract_metadata(text: str, law_name: str) -> list[str]:
    first_article = ARTICLE_RE.search(text)
    preamble = text[: first_article.start()] if first_article else text
    effective_match = re.search(
        r"(?:施行日期|生效日期)\s*[:：]\s*(\d{4})[./年-](\d{1,2})[./月-](\d{1,2})",
        preamble,
    )
    effective_date = None
    if effective_match:
        effective_date = datetime.strptime(
            "-".join(effective_match.groups()),
            "%Y-%m-%d",
        ).date()
    selected: list[str] = []

    for line in (item.strip() for item in preamble.splitlines()):
        if not line or STRUCTURE_LINE_RE.fullmatch(line):
            continue
        if METADATA_RE.search(line):
            if (
                line.startswith("时效性")
                and "尚未施行" in line
                and effective_date
                and effective_date <= date.today()
            ):
                line = f"时效性: 已施行（施行日期 {effective_date.isoformat()}）"
            selected.append(line)

    if not selected:
        return []

    metadata = "；".join(dict.fromkeys(selected))
    return [f"《{law_name}》【法规元数据】{metadata[:2000]}"]


def extract_fallback_chunks(text: str, law_name: str, max_chars: int = 1000) -> list[str]:
    paragraphs = [line.strip() for line in text.splitlines() if line.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_length = 0

    for paragraph in paragraphs:
        if current and current_length + len(paragraph) + 1 > max_chars:
            chunks.append(f"《{law_name}》 " + " ".join(current))
            current = []
            current_length = 0
        current.append(paragraph)
        current_length += len(paragraph) + 1

    if current:
        chunks.append(f"《{law_name}》 " + " ".join(current))
    return chunks


def convert_directory(source_dir: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"source": str(source_dir), "files": [], "total_chunks": 0}

    source_files = sorted(
        path
        for path in source_dir.glob("*.docx")
        if not path.name.startswith("~$")
    )

    for source_path in source_files:
        law_name = clean_title(source_path)
        text = clean_text(docx2txt.process(str(source_path)) or "")
        metadata_chunks = extract_metadata(text, law_name)
        chunks = extract_articles(text, law_name)
        mode = "articles"
        if len(chunks) < 2:
            chunks = extract_fallback_chunks(text, law_name)
            mode = "paragraph_fallback"
        else:
            chunks = metadata_chunks + chunks

        output_path = output_dir / f"{source_path.stem}.txt"
        output_path.write_text("\n".join(chunks) + "\n", encoding="utf-8")
        manifest["files"].append(
            {
                "source": source_path.name,
                "output": output_path.name,
                "mode": mode,
                "chunks": len(chunks),
            }
        )
        manifest["total_chunks"] += len(chunks)

    (output_dir / "_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    manifest = convert_directory(args.source_dir, args.output_dir)
    print(
        f"Converted {len(manifest['files'])} files into "
        f"{manifest['total_chunks']} chunks."
    )


if __name__ == "__main__":
    main()
