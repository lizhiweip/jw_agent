"""Build the independent case-handling guidance FAISS and BM25 indexes."""

import json
import os
from pathlib import Path

from dotenv import load_dotenv

from model_utils import LegalRAGapi


load_dotenv()

records_dir = Path(
    os.getenv("GUIDANCE_RECORDS_PATH", "knowledge_base_guidance/records")
)
db_path = os.getenv("VECTOR_GUIDANCE_DB_PATH", "law_faiss_guidance")

if not records_dir.exists():
    raise SystemExit(
        f"规范文件记录目录不存在：{records_dir}\n"
        "请先运行 prepare_guidance_documents.py。"
    )

if os.path.exists(db_path):
    raise SystemExit(
        f"规范库索引已存在：{db_path}\n"
        "如需重建，请先备份并删除该目录。"
    )

records = []
for path in sorted(records_dir.glob("guidance_*.json")):
    record = json.loads(path.read_text(encoding="utf-8"))
    content = str(record.get("content") or "").strip()
    pid = str(record.get("pid") or "").strip()
    if not content or not pid:
        continue

    header = [f"【办案规范】{record.get('title', '')}"]
    if record.get("document_number"):
        header.append(f"【文号】{record['document_number']}")
    heading_path = [
        str(item).strip()
        for item in record.get("heading_path", [])
        if str(item).strip()
    ]
    if heading_path:
        header.append(f"【层级】{' > '.join(heading_path)}")
    if record.get("section_title"):
        header.append(f"【条款】{record['section_title']}")
    page_start = record.get("page_start")
    page_end = record.get("page_end")
    if page_start:
        page_text = (
            f"第{page_start}页"
            if not page_end or page_end == page_start
            else f"第{page_start}-{page_end}页"
        )
        header.append(f"【页码】{page_text}")
    header.append(content)
    records.append(("\n".join(header), pid))

if not records:
    raise SystemExit("没有找到可索引的规范文件记录。")

print(f"准备构建办案规范库：{len(records)} 个文本块")
rag = LegalRAGapi(
    db_path=db_path,
    source_dir=str(records_dir),
    vector_weight=0.3,
    bm25_weight=0.7,
)
rag.add_documents(
    [item[0] for item in records],
    pids=[item[1] for item in records],
    save_to_disk=True,
)
print(f"办案规范库构建完成：{db_path}")
