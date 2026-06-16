"""Build the independent纪检监察业务问答 FAISS and BM25 indexes."""

import json
import os
from pathlib import Path

from dotenv import load_dotenv

from model_utils import LegalRAGapi


load_dotenv()

records_dir = Path(os.getenv("QA_RECORDS_PATH", "knowledge_base_qa/records"))
db_path = os.getenv("VECTOR_QA_DB_PATH", "law_faiss_qa")

if not records_dir.exists():
    raise SystemExit(f"业务问答记录目录不存在：{records_dir}")
if os.path.exists(db_path):
    raise SystemExit(f"业务问答索引已存在：{db_path}，请先备份或删除后重建。")

texts: list[str] = []
pids: list[str] = []
for path in sorted(records_dir.glob("qa_*.json")):
    record = json.loads(path.read_text(encoding="utf-8"))
    pid = str(record.get("pid") or "").strip()
    question = str(record.get("question") or "").strip()
    answer = str(record.get("answer") or "").strip()
    if not pid or not question or not answer:
        continue
    # Keep a question-focused vector so long reference answers do not dilute
    # similarity, while a second answer-focused vector remains searchable by
    # legal terms and handling conclusions. Both resolve to the same record.
    texts.extend(
        (
            "\n".join(
                (
                    f"【业务问答题目】{record.get('title', '')}",
                    f"【类型】{record.get('category', '')}",
                    f"【题目】{question}",
                )
            ),
            "\n".join(
                (
                    f"【业务问答参考答案】{record.get('title', '')}",
                    f"【类型】{record.get('category', '')}",
                    f"【参考答案】{answer}",
                )
            ),
        )
    )
    pids.extend((pid, pid))

if not texts:
    raise SystemExit("没有找到可索引的业务问答记录。")

print(f"准备构建业务问答库：{len(pids) // 2} 组，{len(texts)} 个检索向量")
rag = LegalRAGapi(
    db_path=db_path,
    source_dir=str(records_dir),
    vector_weight=0.4,
    bm25_weight=0.6,
)
rag.add_documents(texts, pids=pids, save_to_disk=True)
print(f"业务问答库构建完成：{db_path}")
