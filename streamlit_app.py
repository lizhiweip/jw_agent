# 2
import os
import sys
import traceback
import json
import re
import shutil
import tempfile
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
import threading
import queue

# Keep background/service logs from failing on Chinese text or status icons.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="backslashreplace")

import streamlit as st
from dotenv import load_dotenv
from model_utils import  LegalRAGapi
# --- database setup ---
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.engine import make_url
from sqlalchemy.orm import relationship, sessionmaker, declarative_base
from agent_graph import create_legal_agent

from langchain_core.messages import HumanMessage, AIMessage # 用于构建历史对话对象


# load environment variables
load_dotenv()

# ======================
# 页面配置 (必须是第一个 st 命令)
# ======================
st.set_page_config(
    page_title="纪检监察业务助手",
    page_icon="⚙",
    layout="wide",
    initial_sidebar_state="expanded"
)


def inject_legacy_browser_scroll_fix():
    """兼容 Chrome 105 等旧浏览器的 Streamlit 页面滚动异常。"""
    st.markdown(
        """
        <style>
        html, body {
            height: auto !important;
            min-height: 100% !important;
            overflow-y: auto !important;
            overflow-x: hidden !important;
        }

        #root,
        .stApp,
        [data-testid="stApp"],
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        [data-testid="stMainBlockContainer"],
        section.main,
        section[data-testid="stMain"] {
            height: auto !important;
            min-height: 100vh !important;
            max-height: none !important;
            overflow: visible !important;
            overflow-y: visible !important;
        }

        [data-testid="stVerticalBlock"],
        [data-testid="stChatMessage"],
        [data-testid="stExpander"],
        [data-testid="stStatusWidget"] {
            height: auto !important;
            max-height: none !important;
            overflow: visible !important;
        }

        /* Chrome 105 对新版 Streamlit 的 chat_input 容器定位不稳定。
           这里使用基础 fixed 布局，避免输入框插到回答中间。 */
        [data-testid="stBottomBlockContainer"] {
            position: fixed !important;
            left: 21rem !important;
            right: 0 !important;
            bottom: 0 !important;
            z-index: 999 !important;
            background: #ffffff !important;
            border-top: 1px solid #e8edf3 !important;
            padding: 0.55rem 2.2rem 0.9rem 2.2rem !important;
            box-shadow: 0 -6px 18px rgba(15, 23, 42, 0.06) !important;
        }

        [data-testid="stChatInput"] {
            position: static !important;
            width: calc(100% - 5.5rem) !important;
            max-width: calc(100% - 5.5rem) !important;
            margin: 0 !important;
            overflow: visible !important;
        }

        [data-testid="stChatInput"] > div {
            background: #f3f6fa !important;
            border-color: #ff6b6b !important;
            border-radius: 10px !important;
            display: flex !important;
            flex-direction: row !important;
            align-items: center !important;
            overflow: visible !important;
        }

        [data-testid="stChatInput"] [data-baseweb="textarea"],
        [data-testid="stChatInput"] [data-baseweb="base-input"],
        [data-testid="stChatInputTextArea"],
        [data-testid="stChatInput"] textarea {
            min-height: 42px !important;
            max-height: 150px !important;
            border-radius: 10px !important;
            background: #f3f6fa !important;
            border-color: #f3f6fa !important;
            box-shadow: none !important;
        }

        [data-testid="stChatInputSubmitButton"],
        [data-testid="stChatInput"] [data-testid="stChatInputSubmitButton"],
        [data-testid="stChatInput"] button,
        [data-testid="stChatInput"] button[kind],
        [data-testid="stChatInput"] button[type="button"],
        [data-testid="stChatInput"] button[aria-label],
        [data-testid="stChatInput"] [role="button"] {
            position: static !important;
            transform: none !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            visibility: visible !important;
            opacity: 1 !important;
            color: #075985 !important;
            background: #dff2ff !important;
            border: 1px solid #9ed8ff !important;
            width: 4.6rem !important;
            min-width: 4.6rem !important;
            height: 2.25rem !important;
            min-height: 2.25rem !important;
            padding: 0 !important;
            margin: 0 !important;
            border-radius: 0.55rem !important;
            font-size: 0.9rem !important;
            overflow: hidden !important;
            box-shadow: none !important;
            z-index: 3 !important;
            pointer-events: auto !important;
            flex: 0 0 auto !important;
        }

        [data-testid="stChatInputSubmitButton"] svg,
        [data-testid="stChatInput"] svg {
            display: block !important;
            width: 1.1rem !important;
            height: 1.1rem !important;
            color: #075985 !important;
        }

        [data-testid="stChatInputSubmitButton"]:disabled,
        [data-testid="stChatInput"] button:disabled {
            cursor: not-allowed !important;
            opacity: 0.58 !important;
            background: #e6f1fb !important;
            color: #64748b !important;
        }

        .block-container {
            padding-bottom: 9.5rem !important;
            max-width: 100% !important;
        }

        /* 紧凑化引用按钮和反馈按钮，避免宽屏/旧浏览器下间距被拉得很散。 */
        div.stButton {
            margin-top: 0.12rem !important;
            margin-bottom: 0.12rem !important;
        }

        div.stButton > button {
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            text-align: center !important;
            min-height: 2.05rem !important;
            padding: 0.25rem 0.7rem !important;
            border-radius: 0.42rem !important;
            line-height: 1.2 !important;
        }

        div.stButton > button p,
        div.stButton > button span,
        div.stButton > button div[data-testid="stMarkdownContainer"] {
            width: 100% !important;
            margin: 0 !important;
            text-align: center !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            line-height: 1.2 !important;
        }

        .feedback-row div.stButton > button {
            width: 2.35rem !important;
            min-width: 2.35rem !important;
            max-width: 2.35rem !important;
            padding-left: 0 !important;
            padding-right: 0 !important;
        }

        .stChatMessage {
            margin-bottom: 0.8rem !important;
        }

        [data-testid="stStatusWidget"] {
            margin-bottom: 0.4rem !important;
        }

        @media (max-width: 900px) {
            [data-testid="stBottomBlockContainer"] {
                left: 0 !important;
                padding-left: 1rem !important;
                padding-right: 1rem !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_legacy_browser_scroll_fix()

# ======================
# 知识库初始化
# ======================
KNOWLEDGE_BASE_CASE_FOLDER = os.path.join(
    os.getcwd(), 'knowledge_base_case', os.getenv('CASE_KB_SUBDIR', 'candidate_folder')
)
os.makedirs(KNOWLEDGE_BASE_CASE_FOLDER, exist_ok=True)
KNOWLEDGE_BASE_LP_FOLDER=os.path.join(os.getcwd(), 'knowledge_base_lp')
os.makedirs(KNOWLEDGE_BASE_LP_FOLDER, exist_ok=True)
KNOWLEDGE_BASE_GUIDANCE_FOLDER = os.path.join(
    os.getcwd(), "knowledge_base_guidance", "records"
)
os.makedirs(KNOWLEDGE_BASE_GUIDANCE_FOLDER, exist_ok=True)
KNOWLEDGE_BASE_QA_FOLDER = os.path.join(
    os.getcwd(), "knowledge_base_qa", "records"
)
os.makedirs(KNOWLEDGE_BASE_QA_FOLDER, exist_ok=True)
MAX_CONTENT_LENGTH = 10 * 1024 * 1024

case_db_path = os.getenv("VECTOR_CASE_DB_PATH", os.path.join(os.getcwd(), "law_faiss_case")) 
lp_db_path = os.getenv("VECTOR_LP_DB_PATH", os.path.join(os.getcwd(), "law_faiss_lp")) 
guidance_db_path = os.getenv(
    "VECTOR_GUIDANCE_DB_PATH",
    os.path.join(os.getcwd(), "law_faiss_guidance"),
)
qa_db_path = os.getenv(
    "VECTOR_QA_DB_PATH",
    os.path.join(os.getcwd(), "law_faiss_qa"),
)
PRIVATE_KB_ROOT = Path(os.getenv("PRIVATE_KB_ROOT", os.path.join(os.getcwd(), "private_knowledge_bases")))
PRIVATE_INDEX_ROOT = Path(os.getenv("PRIVATE_INDEX_ROOT", os.path.join(os.getcwd(), "private_law_faiss")))
UPLOAD_TMP_ROOT = Path(os.getenv("UPLOAD_TMP_ROOT", os.path.join(os.getcwd(), "data", "uploads")))
for _path in (PRIVATE_KB_ROOT, PRIVATE_INDEX_ROOT, UPLOAD_TMP_ROOT):
    _path.mkdir(parents=True, exist_ok=True)


KNOWLEDGE_TYPE_CONFIG = {
    "法律法规": {
        "key": "lp",
        "public_source": Path(KNOWLEDGE_BASE_LP_FOLDER),
        "public_index": Path(lp_db_path),
        "private_source": "knowledge_base_lp",
        "private_index": "law_faiss_lp",
    },
    "案例": {
        "key": "case",
        "public_source": Path(KNOWLEDGE_BASE_CASE_FOLDER),
        "public_index": Path(case_db_path),
        "private_source": "knowledge_base_case",
        "private_index": "law_faiss_case",
    },
    "办案规范": {
        "key": "guidance",
        "public_source": Path(os.getcwd()) / "knowledge_base_guidance" / "records",
        "public_index": Path(guidance_db_path),
        "private_source": "knowledge_base_guidance/records",
        "private_index": "law_faiss_guidance",
    },
    "业务题库": {
        "key": "qa",
        "public_source": Path(os.getcwd()) / "knowledge_base_qa" / "records",
        "public_index": Path(qa_db_path),
        "private_source": "knowledge_base_qa/records",
        "private_index": "law_faiss_qa",
    },
}


def _safe_space_name(name: str) -> str:
    name = (name or "").strip()
    if not name:
        return ""
    cleaned = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", name)
    return cleaned.strip("._-")[:50]


def _safe_index_space_name(name: str) -> str:
    """FAISS on Windows can fail to write index files under non-ASCII paths."""
    safe_name = _safe_space_name(name)
    ascii_part = re.sub(r"[^A-Za-z0-9_.-]+", "_", safe_name).strip("._-")
    digest = uuid.uuid5(uuid.NAMESPACE_URL, safe_name).hex[:12]
    if ascii_part:
        return f"{ascii_part[:36]}_{digest}"
    return f"space_{digest}"


def _private_source_dir(space: str, knowledge_type: str) -> Path:
    cfg = KNOWLEDGE_TYPE_CONFIG[knowledge_type]
    return PRIVATE_KB_ROOT / _safe_space_name(space) / cfg["private_source"]


def _private_index_dir(space: str, knowledge_type: str) -> Path:
    cfg = KNOWLEDGE_TYPE_CONFIG[knowledge_type]
    return PRIVATE_INDEX_ROOT / _safe_index_space_name(space) / cfg["private_index"]


def list_private_spaces() -> list[str]:
    if not PRIVATE_KB_ROOT.exists():
        return []
    return sorted(
        p.name for p in PRIVATE_KB_ROOT.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )


class CombinedRAG:
    """把公共知识库和私有知识库包装成一个检索对象，供现有 Agent 无感使用。"""

    def __init__(self, public_rag, private_rag=None):
        self.public_rag = public_rag
        self.private_rag = private_rag
        self.source_dir = getattr(public_rag, "source_dir", None)
        self.embedding_model = getattr(public_rag, "embedding_model", None)
        self.vector_db = getattr(public_rag, "vector_db", None)
        self.db_path = getattr(public_rag, "db_path", None)

    def retrieve_documents(self, query: str, top_k: int = 10):
        results = []
        if self.public_rag is not None:
            results.extend(self.public_rag.retrieve_documents(query, top_k=top_k))
        if self.private_rag is not None:
            results.extend(self.private_rag.retrieve_documents(query, top_k=top_k))
        return sorted(results, key=lambda item: item[1], reverse=True)[:top_k]

    def get_case_record(self, pid: str):
        for rag in (self.public_rag, self.private_rag):
            if rag is None:
                continue
            record = rag.get_case_record(pid)
            if record:
                return record
        return None

    def get_guidance_record(self, pid: str):
        for rag in (self.public_rag, self.private_rag):
            if rag is None:
                continue
            record = rag.get_guidance_record(pid)
            if record:
                return record
        return None

    def get_qa_record(self, pid: str):
        for rag in (self.public_rag, self.private_rag):
            if rag is None:
                continue
            record = rag.get_qa_record(pid)
            if record:
                return record
        return None

    def get_document_count(self):
        total = 0
        for rag in (self.public_rag, self.private_rag):
            if rag is not None:
                total += rag.get_document_count()
        return total

    def get_bm25_document_count(self):
        total = 0
        for rag in (self.public_rag, self.private_rag):
            if rag is not None:
                total += rag.get_bm25_document_count()
        return total




def initialize_vector_database(case_rag_model: LegalRAGapi, lp_rag_model: LegalRAGapi):
    """初始化向量数据库，自动处理knowledge_base文件夹"""

    # 检查案例向量数据库是否已存在
    if not os.path.exists(case_db_path):
        print("案例向量数据库不存在，开始构建...")

        # 首先处理knowledge_base_case文件夹
        if os.path.exists(KNOWLEDGE_BASE_CASE_FOLDER) and os.listdir(KNOWLEDGE_BASE_CASE_FOLDER):
            print(f"正在处理knowledge_base_case文件夹: {KNOWLEDGE_BASE_CASE_FOLDER}")
            case_rag_model.add_folder_documents(KNOWLEDGE_BASE_CASE_FOLDER)
            print("knowledge_base_case文件夹中的文档已添加到向量数据库")
        else:
            print(f"knowledge_base_case文件夹为空或不存在: {KNOWLEDGE_BASE_CASE_FOLDER}")

        print(f"案例向量数据库构建完成，总文档数: {case_rag_model.get_document_count()}")
        print(f"BM25索引构建完成，总文档数: {case_rag_model.get_bm25_document_count()}")
    
    else:
        print("案例向量数据库已存在，跳过初始化构建")

    # 检查法条向量数据库是否已存在
    if not os.path.exists(lp_db_path):
        print("法条向量数据库不存在，开始构建...")

        # 首先处理knowledge_base_lp文件夹
        if os.path.exists(KNOWLEDGE_BASE_LP_FOLDER) and os.listdir(KNOWLEDGE_BASE_LP_FOLDER):
            print(f"正在处理knowledge_base_lp文件夹: {KNOWLEDGE_BASE_LP_FOLDER}")
            lp_rag_model.add_folder_documents(KNOWLEDGE_BASE_LP_FOLDER)
            print("knowledge_base_lp文件夹中的文档已添加到向量数据库")
        else:
            print(f"knowledge_base_lp文件夹为空或不存在: {KNOWLEDGE_BASE_LP_FOLDER}")

        print(f"法条向量数据库构建完成，总文档数: {lp_rag_model.get_document_count()}")
        print(f"BM25索引构建完成，总文档数: {lp_rag_model.get_bm25_document_count()}")
    else:
        print("法条向量数据库已存在，跳过初始化构建")


def _save_uploaded_files(uploaded_files, target_dir: Path) -> list[Path]:
    target_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    for uploaded_file in uploaded_files or []:
        safe_name = Path(uploaded_file.name).name
        stamp = datetime.now().strftime("%Y%m%d%H%M%S")
        dest = target_dir / f"{stamp}_{uuid.uuid4().hex[:8]}_{safe_name}"
        with open(dest, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_paths.append(dest)
    return saved_paths


def _ocr_python_executable() -> Path | None:
    candidates = [
        Path(os.getcwd()) / ".ocr_env" / "python.exe",
        Path(os.getcwd()) / ".ocr_env" / "Scripts" / "python.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _append_guidance_record_texts(record_files: list[Path], texts: list[str], pids: list[str]):
    for record_path in record_files:
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"⚠️ 办案规范记录读取失败：{record_path}，{e}")
            continue
        content = str(record.get("content") or "").strip()
        pid = str(record.get("pid") or record_path.stem).strip()
        if not content or not pid:
            continue
        texts.append(
            "\n".join(
                (
                    f"【办案规范】{record.get('title', record_path.stem)}",
                    f"【来源文件】{record.get('source_file', '')}",
                    content,
                )
            )
        )
        pids.append(pid)


def _build_guidance_records_with_ocr(file_path: Path, records_dir: Path) -> list[Path]:
    ocr_python = _ocr_python_executable()
    if not ocr_python:
        raise RuntimeError(
            "该 PDF 可能是扫描版，当前主环境无法直接抽取文字；未找到 .ocr_env/python.exe，无法自动 OCR。"
        )

    output_root = records_dir.parent
    before = {path.resolve() for path in records_dir.glob("guidance_*.json")}
    cmd = [
        str(ocr_python),
        str(Path(os.getcwd()) / "prepare_guidance_documents.py"),
        "--source",
        str(file_path.parent),
        "--output",
        str(output_root),
        "--contains",
        file_path.name,
        "--force-ocr",
    ]
    print("🔎 扫描版 PDF 自动 OCR：", " ".join(cmd))
    result = subprocess.run(
        cmd,
        cwd=os.getcwd(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError(
            "OCR 处理失败：\n"
            + (result.stdout or "")
            + "\n"
            + (result.stderr or "")
        )
    after_files = sorted(records_dir.glob("guidance_*.json"), key=lambda p: p.stat().st_mtime)
    new_files = [path for path in after_files if path.resolve() not in before]
    if not new_files:
        raise RuntimeError(
            "OCR 已运行但未生成可入库记录。请检查 PDF 是否为空、是否加密，或查看 prepare_guidance_documents.py 输出。"
        )
    return new_files


def _build_guidance_records_from_files(files: list[Path], records_dir: Path, rag: LegalRAGapi):
    records_dir.mkdir(parents=True, exist_ok=True)
    texts: list[str] = []
    pids: list[str] = []
    for file_path in files:
        try:
            structured_chunks = rag.document_processor.process_document(str(file_path))
        except Exception as e:
            if file_path.suffix.lower() == ".pdf":
                print(f"⚠️ PDF 直接抽取失败，尝试 OCR：{e}")
                ocr_records = _build_guidance_records_with_ocr(file_path, records_dir)
                _append_guidance_record_texts(ocr_records, texts, pids)
                continue
            raise
        if not structured_chunks:
            continue
        document_id = uuid.uuid5(uuid.NAMESPACE_URL, str(file_path)).hex[:12]
        for idx, chunk in enumerate(structured_chunks, 1):
            content = str(chunk.get("full_text") or chunk.get("content") or "").strip()
            if not content:
                continue
            sub_chunks = [content]
            if len(content) > 1200:
                sub_chunks = rag.general_splitter.split_text(content)
            for sub_idx, text in enumerate(sub_chunks, 1):
                pid = f"guidance_upload_{document_id}_{idx:04d}_{sub_idx:02d}"
                record = {
                    "pid": pid,
                    "document_id": document_id,
                    "knowledge_type": "办案规范",
                    "title": file_path.stem,
                    "document_number": "",
                    "confidentiality": "",
                    "source_file": file_path.name,
                    "page_start": chunk.get("page", chunk.get("page_start")),
                    "page_end": chunk.get("page", chunk.get("page_end")),
                    "chunk_index": idx,
                    "section_title": chunk.get("title", ""),
                    "heading_path": [],
                    "content": text,
                }
                (records_dir / f"{pid}.json").write_text(
                    json.dumps(record, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                texts.append(
                    "\n".join(
                        (
                            f"【办案规范】{record['title']}",
                            f"【来源文件】{record['source_file']}",
                            text,
                        )
                    )
                )
                pids.append(pid)
    if texts:
        rag.add_documents(texts, pids=pids, save_to_disk=True)
    return len(texts)


def _extract_plain_text(path: Path, rag: LegalRAGapi) -> str:
    if path.suffix.lower() == ".json":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return json.dumps(data, ensure_ascii=False)
        except Exception:
            return path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix.lower() == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")
    chunks = rag.document_processor.process_document(str(path))
    return "\n".join(
        str(chunk.get("full_text") or chunk.get("content") or "").strip()
        for chunk in chunks
        if str(chunk.get("full_text") or chunk.get("content") or "").strip()
    )


def _extract_json_object(text: str) -> dict | None:
    text = (text or "").strip()
    if not text:
        return None
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text).strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _extract_json_array(text: str) -> list | None:
    text = (text or "").strip()
    if not text:
        return None
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text).strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else None
    except Exception:
        pass
    match = re.search(r"\[.*\]", text, re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        return data if isinstance(data, list) else None
    except Exception:
        return None


def _clean_single_line(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").replace("\u3000", " ")).strip()


def _original_upload_stem(path: Path) -> str:
    stem = path.stem
    return re.sub(r"^\d{14}_[0-9a-f]{8}_", "", stem)


def _infer_statute_title(text: str, filename: str) -> str:
    text = text or ""
    fallback_bracket_title = None
    for line in text.splitlines()[:30]:
        line = _clean_single_line(line)
        if not line:
            continue
        bracket_title = re.fullmatch(r"《([^》]{2,80})》", line)
        if bracket_title:
            fallback_bracket_title = bracket_title.group(1).strip()
            continue
        if re.match(r"^(第\s*[零〇一二三四五六七八九十百千万两\d\s]+\s*[章节条编]|目录|总则|附则|[一二三四五六七八九十百千万]+、)", line):
            continue
        if len(line) <= 80 and not re.search(r"[。；;，,]", line) and not re.search(r"第\s*[零〇一二三四五六七八九十百千万两\d\s]+\s*条", line):
            return re.sub(r"(全文|原文|正式版|修订版)$", "", line).strip(" ：:") or Path(filename).stem

    if fallback_bracket_title:
        return fallback_bracket_title
    bracket_title = re.search(r"《([^》]{2,80})》", text)
    if bracket_title:
        return bracket_title.group(1).strip()
    return _original_upload_stem(Path(filename))


def _title_looks_bad(title: str) -> bool:
    title = title or ""
    return (
        not title
        or len(title) > 60
        or "《" in title
        or "》" in title
        or bool(re.search(r"第\s*[零〇一二三四五六七八九十百千万两\d\s]+\s*条", title))
        or bool(re.search(r"[。；;，,]", title))
    )


def _candidate_title_from_line(line: str) -> str:
    line = _clean_single_line(line)
    line = re.sub(r"^《([^》]+)》$", r"\1", line)
    line = re.sub(r"[（(].{0,30}?[）)]$", "", line).strip()
    if (
        4 <= len(line) <= 40
        and re.search(r"(条例|法律|法|规定|办法|规则|细则|决定|解释|通知|意见|监察法)", line)
        and not re.search(r"第\s*[零〇一二三四五六七八九十百千万两\d\s]+\s*条", line)
        and not re.search(r"[。；;，,]", line)
    ):
        return line
    return ""


def _repair_statute_title(title: str, source_text: str, filename: str, lines: list[str]) -> str:
    title = (title or "").strip("《》 ")
    candidates: list[str] = []
    for raw_line in (source_text or "").splitlines()[:80]:
        candidate = _candidate_title_from_line(raw_line)
        if candidate:
            candidates.append(candidate)

    for line in lines[:5]:
        body = re.sub(r"^《[^》]+》\s*第\s*[零〇一二三四五六七八九十百千万两\d\s]+\s*条\s*", "", line)
        head = re.split(r"[，,。；;：:]", body, maxsplit=1)[0].strip()
        candidate = _candidate_title_from_line(head)
        if candidate:
            candidates.append(candidate)

    if candidates:
        if _title_looks_bad(title):
            return candidates[0]
        for candidate in candidates:
            if candidate != title and candidate not in title and title not in candidate:
                first_body = lines[0] if lines else source_text[:500]
                first_body = re.sub(
                    r"^《[^》]+》\s*第\s*[零〇一二三四五六七八九十百千万两\d\s]+\s*条\s*",
                    "",
                    first_body,
                )
                if candidate in first_body and title not in first_body:
                    return candidate

    if not _title_looks_bad(title):
        return title

    return _original_upload_stem(Path(filename))


def _replace_statute_title(lines: list[str], title: str) -> list[str]:
    return [
        re.sub(r"^《.*》(?=第\s*[零〇一二三四五六七八九十百千万两\d\s]+\s*条|第[一二三四五六七八九十百千万]+项|全文)", f"《{title}》", line, count=1)
        for line in lines
    ]


def _split_statute_articles(text: str, title: str) -> list[str]:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    article_num_token = r"[零〇一二三四五六七八九十百千万两\d](?:\s*[零〇一二三四五六七八九十百千万两\d])*"
    article_token = rf"第\s*{article_num_token}\s*条"

    def build_lines(matches: list[re.Match], label_getter) -> list[str]:
        built: list[str] = []
        for idx, match in enumerate(matches):
            label = re.sub(r"\s+", "", label_getter(match))
            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            body = _clean_single_line(text[start:end])
            body = re.sub(rf"^({article_token}|[一二三四五六七八九十百千万]+、)\s*", "", body).strip()
            body = re.sub(r"(?:\s*《[^》]{2,80}》\s*)+$", "", body).strip()
            if body:
                built.append(f"《{title}》{label} {body}")
        return built

    article_pattern = re.compile(rf"(?m)^\s*({article_token})\s*")
    matches = list(article_pattern.finditer(text))
    if len(matches) < 2:
        article_pattern = re.compile(rf"(?:^|[\n。；;])\s*({article_token})\s*")
        matches = list(article_pattern.finditer(text))
    if len(matches) < 2:
        article_pattern = re.compile(rf"({article_token})\s*")
        loose_matches = list(article_pattern.finditer(text))
        if len(loose_matches) >= 2:
            matches = loose_matches

    lines = build_lines(matches, lambda m: m.group(1)) if matches else []
    if len(lines) >= 2:
        return lines

    item_pattern = re.compile(r"(?m)^\s*([一二三四五六七八九十百千万]+)、\s*")
    item_matches = list(item_pattern.finditer(text))
    if len(item_matches) < 2:
        item_pattern = re.compile(r"(?:^|[\n。；;])\s*([一二三四五六七八九十百千万]+)、\s*")
        item_matches = list(item_pattern.finditer(text))
    if len(item_matches) >= 2:
        return build_lines(item_matches, lambda m: f"第{m.group(1)}项")

    return lines


def _format_statute_with_llm(source_text: str, filename: str) -> list[str]:
    api_key = st.session_state.get("current_api_key") or os.getenv("DEEPSEEK_API_KEY")
    base_url = st.session_state.get("current_base_url") or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    model_name = st.session_state.get("current_model_name") or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    if not api_key:
        return []

    try:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            api_key=api_key,
            base_url=base_url,
            model=model_name,
            temperature=0,
            streaming=False,
        )
        prompt = f"""
你是法规文本结构化助手。请把用户上传的一部法规原文整理成“一行一条”的字符串数组。
要求：
1. 只输出 JSON 数组，不要输出 Markdown，不要解释。
2. 每个元素格式必须是： 《法规标题》第X条 原文正文
3. 不要总结、改写、增删原文；只做标题补齐、条文合并、换行清理。
4. 如果无法识别条文，返回空数组 []。
5. 上传内容通常只是一部法规或一个版本，不要编造其他版本。

文件名：{filename}
原文：
{source_text[:16000]}
"""
        response = llm.invoke(prompt)
        data = _extract_json_array(getattr(response, "content", ""))
        if not data:
            return []
        return [_clean_single_line(str(item)) for item in data if _clean_single_line(str(item))]
    except Exception as e:
        print(f"⚠️ 法规文本 LLM 重构失败：{e}")
        return []


def _build_statute_records_from_files(files: list[Path], source_dir: Path, rag: LegalRAGapi) -> int:
    source_dir.mkdir(parents=True, exist_ok=True)
    total_lines = 0
    for path in files:
        source_text = _extract_plain_text(path, rag)
        if not source_text:
            continue

        normalized_stem = _original_upload_stem(path)
        title = normalized_stem or _infer_statute_title(source_text, path.name)
        lines = _split_statute_articles(source_text, title)
        if len(lines) < 2:
            llm_lines = _format_statute_with_llm(source_text, path.name)
            if llm_lines:
                lines = _replace_statute_title(llm_lines, title)

        if not lines:
            fallback = _clean_single_line(source_text)
            if fallback:
                lines = [f"《{title}》全文 {fallback}"]

        if not lines:
            continue

        lines = _replace_statute_title(lines, title)

        normalized_path = source_dir / f"{normalized_stem}.txt"
        if normalized_path.exists():
            normalized_path = source_dir / f"{normalized_stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.txt"
        normalized_path.write_text("\n".join(lines), encoding="utf-8")
        Path(rag.db_path).mkdir(parents=True, exist_ok=True)
        rag.add_file_documents(str(normalized_path), save_to_disk=True)
        total_lines += len(lines)

    return total_lines


def _read_docx_preview(path: Path, max_chars: int) -> str:
    try:
        from zipfile import ZipFile
        import xml.etree.ElementTree as ET

        with ZipFile(path) as zf:
            root = ET.fromstring(zf.read("word/document.xml"))
        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        paragraphs = []
        for para in root.findall(".//w:p", ns):
            text = "".join(t.text or "" for t in para.findall(".//w:t", ns)).strip()
            if text:
                paragraphs.append(text)
        return "\n".join(paragraphs)[:max_chars] or "未抽取到可预览文本。"
    except Exception as e:
        return f"DOCX 预览失败：{e}"


def _read_pdf_preview(path: Path, max_chars: int) -> str:
    try:
        try:
            from pypdf import PdfReader
        except Exception:
            from PyPDF2 import PdfReader

        reader = PdfReader(str(path))
        pages = []
        for page in reader.pages[:8]:
            text = page.extract_text() or ""
            if text.strip():
                pages.append(text.strip())
        return "\n\n".join(pages)[:max_chars] or "未抽取到可预览文本，可能是扫描版 PDF。"
    except Exception as e:
        return f"PDF 预览失败：{e}"


def _xlsx_column_index(cell_ref: str) -> int:
    letters = re.sub(r"[^A-Z]", "", (cell_ref or "").upper())
    index = 0
    for char in letters:
        index = index * 26 + (ord(char) - ord("A") + 1)
    return max(index - 1, 0)


def _read_xlsx_rows(path: Path, max_rows_per_sheet: int | None = None) -> list[tuple[str, list[list[str]]]]:
    from zipfile import ZipFile
    import xml.etree.ElementTree as ET

    ns_main = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    ns_rel = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
    ns_pkg_rel = "{http://schemas.openxmlformats.org/package/2006/relationships}"

    def read_xml(zf: ZipFile, name: str):
        return ET.fromstring(zf.read(name))

    with ZipFile(path) as zf:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = read_xml(zf, "xl/sharedStrings.xml")
            for si in root.findall(f"{ns_main}si"):
                text = "".join(t.text or "" for t in si.findall(f".//{ns_main}t"))
                shared_strings.append(text)

        rels = {}
        if "xl/_rels/workbook.xml.rels" in zf.namelist():
            rel_root = read_xml(zf, "xl/_rels/workbook.xml.rels")
            for rel in rel_root:
                rel_id = rel.attrib.get("Id")
                target = rel.attrib.get("Target", "")
                if rel_id and target:
                    clean_target = target.lstrip("/")
                    rels[rel_id] = clean_target if clean_target.startswith("xl/") else "xl/" + clean_target

        workbook = read_xml(zf, "xl/workbook.xml")
        sheets = []
        for sheet in workbook.findall(f".//{ns_main}sheet"):
            name = sheet.attrib.get("name", "Sheet")
            rel_id = sheet.attrib.get(f"{ns_rel}id")
            target = rels.get(rel_id or "")
            if target:
                sheets.append((name, target))

        result: list[tuple[str, list[list[str]]]] = []
        for sheet_name, sheet_path in sheets:
            if sheet_path not in zf.namelist():
                continue
            root = read_xml(zf, sheet_path)
            rows: list[list[str]] = []
            for row_idx, row_node in enumerate(root.findall(f".//{ns_main}sheetData/{ns_main}row"), 1):
                if max_rows_per_sheet is not None and len(rows) >= max_rows_per_sheet:
                    break
                row_values: list[str] = []
                for cell in row_node.findall(f"{ns_main}c"):
                    cell_ref = cell.attrib.get("r", "")
                    col_idx = _xlsx_column_index(cell_ref)
                    while len(row_values) <= col_idx:
                        row_values.append("")
                    cell_type = cell.attrib.get("t")
                    value_node = cell.find(f"{ns_main}v")
                    inline_node = cell.find(f"{ns_main}is/{ns_main}t")
                    value = ""
                    if cell_type == "inlineStr" and inline_node is not None:
                        value = inline_node.text or ""
                    elif value_node is not None:
                        raw_value = value_node.text or ""
                        if cell_type == "s":
                            try:
                                value = shared_strings[int(raw_value)]
                            except Exception:
                                value = raw_value
                        elif cell_type == "b":
                            value = "是" if raw_value == "1" else "否"
                        else:
                            value = raw_value
                    row_values[col_idx] = str(value).strip()
                if any(row_values):
                    rows.append(row_values)
            result.append((sheet_name, rows))
    return result


def _read_xlsx_preview(path: Path, max_chars: int) -> str:
    try:
        parts = []
        for sheet_name, rows in _read_xlsx_rows(path, max_rows_per_sheet=30)[:3]:
            parts.append(f"【{sheet_name}】")
            for row in rows:
                if any(value.strip() for value in row):
                    parts.append("\t".join(row))
        return "\n".join(parts)[:max_chars] or "未抽取到可预览内容。"
    except Exception as e:
        return f"XLSX 预览失败：{e}"


def _read_kb_file_preview(path: Path, max_chars: int = 12000) -> str:
    suffix = path.suffix.lower()
    try:
        if suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            return json.dumps(data, ensure_ascii=False, indent=2)[:max_chars]
        if suffix in {".txt", ".md", ".csv"}:
            return path.read_text(encoding="utf-8", errors="ignore")[:max_chars]
        if suffix == ".docx":
            return _read_docx_preview(path, max_chars)
        if suffix == ".pdf":
            return _read_pdf_preview(path, max_chars)
        if suffix == ".xlsx":
            return _read_xlsx_preview(path, max_chars)
        return f"该文件类型暂不直接预览：{path.name}\n路径：{path}"
    except Exception as e:
        return f"读取失败：{e}"


def _collect_kb_files(source_dir: Path) -> list[tuple[str, Path]]:
    files: list[tuple[str, Path]] = []
    if source_dir.exists():
        files.extend(("入库文件", path) for path in source_dir.rglob("*") if path.is_file())
    upload_dir = source_dir.parent / "uploads"
    if upload_dir.exists():
        files.extend(("原始上传", path) for path in upload_dir.rglob("*") if path.is_file())
    return sorted(files, key=lambda item: item[1].stat().st_mtime, reverse=True)


def render_knowledge_view_page():
    st.title("查看知识库")
    st.caption("查看公共知识库和私有知识库中的入库文件、原始上传文件及内容预览。")

    top_left, top_right = st.columns([1, 5])
    with top_left:
        if st.button("返回对话", key="back_to_chat_from_kb_view", use_container_width=True):
            st.session_state.current_page = "chat"
            st.rerun()
    with top_right:
        if st.button("去上传知识库", key="go_to_kb_upload_from_view", use_container_width=True):
            st.session_state.current_page = "knowledge_upload"
            st.rerun()

    st.divider()
    c1, c2, c3 = st.columns([1.2, 1.6, 1.6])
    with c1:
        view_scope = st.radio(
        "查看范围",
        options=["公共知识库", "私有知识库"],
        horizontal=True,
        key="kb_view_scope",
        )
    view_private_space = ""
    with c2:
        if view_scope == "私有知识库":
            spaces = list_private_spaces()
            if not spaces:
                st.info("当前还没有私有知识库。")
                return
            view_private_space = st.selectbox(
                "选择私有知识库",
                options=spaces,
                key="kb_view_private_space",
            )
        else:
            st.selectbox("选择私有知识库", options=["不使用"], disabled=True, key="kb_view_private_disabled")
    with c3:
        view_type = st.selectbox(
            "知识库类型",
            options=list(KNOWLEDGE_TYPE_CONFIG.keys()),
            key="kb_view_type",
        )
    try:
        source_dir, index_dir = _get_target_paths(view_scope, view_private_space, view_type)
    except Exception as e:
        st.warning(str(e))
        return

    files = _collect_kb_files(source_dir)
    source_count = sum(1 for group, _ in files if group == "入库文件")
    upload_count = sum(1 for group, _ in files if group == "原始上传")

    m1, m2, m3 = st.columns(3)
    m1.metric("入库文件", source_count)
    m2.metric("原始上传", upload_count)
    m3.metric("索引目录", "已创建" if index_dir.exists() else "未创建")

    with st.expander("目录信息", expanded=False):
        st.code(f"源文件目录：{source_dir}\n索引目录：{index_dir}")

    if not files:
        st.info("该知识库暂无文件。")
        return

    left, right = st.columns([1.15, 2.2], gap="large")
    with left:
        group_filter = st.radio(
            "文件类别",
            options=["全部", "入库文件", "原始上传"],
            horizontal=True,
            key="kb_view_group_filter",
        )
        keyword = st.text_input("按文件名搜索", key="kb_view_keyword", placeholder="输入关键词筛选")
        visible_files = [
            (group, path)
            for group, path in files
            if (group_filter == "全部" or group == group_filter)
            and (not keyword.strip() or keyword.strip().lower() in path.name.lower())
        ]
        if not visible_files:
            st.warning("没有匹配的文件。")
            return
        file_labels = [
            f"{group} | {path.name} | {path.stat().st_size / 1024:.1f} KB"
            for group, path in visible_files
        ]
        selected_label = st.selectbox(
            "文件列表",
            options=file_labels,
            key=f"kb_file_preview_{view_scope}_{view_private_space}_{view_type}_{group_filter}",
        )
        selected_group, selected_path = visible_files[file_labels.index(selected_label)]

    with right:
        header_col, download_col = st.columns([3, 1])
        with header_col:
            st.markdown(f"### {selected_path.name}")
            st.caption(f"{selected_group} / {selected_path.suffix.lower() or '无扩展名'} / {selected_path.stat().st_size / 1024:.1f} KB")
        with download_col:
            st.download_button(
                "下载文件",
                data=selected_path.read_bytes(),
                file_name=selected_path.name,
                mime="application/octet-stream",
                use_container_width=True,
            )

        preview = _read_kb_file_preview(selected_path)
        if selected_path.suffix.lower() == ".json":
            st.code(preview, language="json")
        else:
            st.text_area("内容预览", preview, height=520, disabled=True)


def _structure_case_with_llm(source_text: str, filename: str) -> dict | None:
    api_key = st.session_state.get("current_api_key") or os.getenv("DEEPSEEK_API_KEY")
    base_url = st.session_state.get("current_base_url") or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    model_name = st.session_state.get("current_model_name") or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    if not api_key:
        return None

    try:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            api_key=api_key,
            base_url=base_url,
            model=model_name,
            temperature=0,
            streaming=False,
        )
        prompt = f"""
你是纪检监察案例材料结构化助手。请把用户上传的案例/案情材料整理为严格 JSON。
只输出 JSON，不要输出 Markdown，不要解释。

字段要求：
- pid: 留空，系统会生成
- fact: 案件事实或线索事实，尽量完整但不要编造
- reason: 违反内容、定性理由、问题性质、审理意见；没有就留空
- result: 处理结果、处分结果、移送情况；没有就留空
- charge: 数组，涉及罪名、违纪类型或问题类型；没有就 []
- article: 数组，涉及条文编号或法规条目；没有就 []
- qw: 原文全文或可核验摘录

禁止编造原文没有的信息。不能确定的字段留空或 []。

文件名：{filename}
原文：
{source_text[:12000]}
"""
        response = llm.invoke(prompt)
        content = getattr(response, "content", "")
        data = _extract_json_object(content)
        return data if isinstance(data, dict) else None
    except Exception as e:
        print(f"⚠️ 案例结构化 LLM 调用失败：{e}")
        return None


def _normalize_case_record(record: dict | None, source_text: str, filename: str) -> dict:
    record = record if isinstance(record, dict) else {}
    pid = str(record.get("pid") or f"case_upload_{uuid.uuid4().hex[:12]}").strip()
    fact = str(record.get("fact") or "").strip()
    reason = str(record.get("reason") or "").strip()
    result = str(record.get("result") or "").strip()
    qw = str(record.get("qw") or "").strip()
    if not fact:
        fact = source_text[:6000]
    if not qw:
        qw = source_text

    charge = record.get("charge") or []
    if isinstance(charge, str):
        charge = [item.strip() for item in re.split(r"[、,，;\s]+", charge) if item.strip()]
    if not isinstance(charge, list):
        charge = []

    article = record.get("article") or []
    if isinstance(article, str):
        article = [item.strip() for item in re.split(r"[、,，;\s]+", article) if item.strip()]
    if not isinstance(article, list):
        article = []

    return {
        "pid": pid,
        "source_file": filename,
        "fact": fact,
        "reason": reason,
        "result": result,
        "charge": charge,
        "article": article,
        "qw": qw,
    }


def _case_field_from_row(row: dict, aliases: list[str]) -> str:
    alias_set = {alias.lower() for alias in aliases}
    for key, value in row.items():
        normalized_key = str(key or "").strip().lower()
        if normalized_key in alias_set or any(alias in normalized_key for alias in alias_set):
            text = str(value or "").strip()
            if text:
                return text
    return ""


def _split_case_list_field(value: str) -> list[str]:
    return [item.strip() for item in re.split(r"[、,，;；\n\r\t]+", value or "") if item.strip()]


def _row_to_case_record(row: dict, source_file: str, row_number: int) -> dict:
    fact = _case_field_from_row(
        row,
        ["fact", "案件事实", "事实", "案情", "线索事实", "主要事实", "问题事实", "违纪事实", "违法事实", "案例内容", "内容"],
    )
    reason = _case_field_from_row(
        row,
        ["reason", "理由", "判决理由", "处理理由", "定性依据", "定性理由", "违反内容", "问题性质", "审理意见", "分析"],
    )
    result = _case_field_from_row(
        row,
        ["result", "结果", "处理结果", "处分结果", "判决结果", "处置结果", "移送情况", "结论"],
    )
    charge = _case_field_from_row(
        row,
        ["charge", "罪名", "违纪类型", "问题类型", "违法类型", "涉及问题", "案由"],
    )
    article = _case_field_from_row(
        row,
        ["article", "条文", "法条", "法规依据", "依据", "适用条款", "适用法规"],
    )
    pid = _case_field_from_row(row, ["pid", "id", "编号", "案号", "序号"])
    title = _case_field_from_row(row, ["标题", "名称", "案例名称", "案件名称"])

    row_text_parts = []
    for key, value in row.items():
        text = str(value or "").strip()
        if text:
            row_text_parts.append(f"{key}: {text}")
    row_text = "\n".join(row_text_parts)

    if not fact:
        fact = row_text
    if title and title not in fact:
        fact = f"{title}\n{fact}".strip()

    return _normalize_case_record(
        {
            "pid": pid or f"case_upload_{Path(source_file).stem}_{row_number}_{uuid.uuid4().hex[:8]}",
            "fact": fact,
            "reason": reason,
            "result": result,
            "charge": _split_case_list_field(charge),
            "article": _split_case_list_field(article),
            "qw": row_text,
        },
        row_text,
        source_file,
    )


def _case_records_from_excel(path: Path) -> list[dict]:
    records: list[dict] = []
    for sheet_name, sheet_rows in _read_xlsx_rows(path):
        headers = None
        for row_index, cells in enumerate(sheet_rows, 1):
            cells = [str(value or "").strip() for value in cells]
            if not any(cells):
                continue
            if headers is None:
                headers = [
                    cell if cell else f"列{idx}"
                    for idx, cell in enumerate(cells, 1)
                ]
                continue
            row = {
                headers[idx] if idx < len(headers) else f"列{idx + 1}": cells[idx]
                for idx in range(max(len(headers), len(cells)))
                if idx < len(cells) and cells[idx]
            }
            if not row:
                continue
            row["工作表"] = sheet_name
            records.append(_row_to_case_record(row, path.name, row_index))
    return records


def _build_qa_records_from_files(files: list[Path], records_dir: Path, rag: LegalRAGapi):
    records_dir.mkdir(parents=True, exist_ok=True)
    files_by_stem = {path.stem: path for path in files}
    used: set[Path] = set()
    texts: list[str] = []
    pids: list[str] = []

    pair_pattern = re.compile(r"^(?P<number>\d+)(?P<answer>\.1)?$")
    numbers = sorted(
        {
            int(match.group("number"))
            for path in files
            for match in [pair_pattern.match(path.stem)]
            if match
        }
    )
    for number in numbers:
        question_path = files_by_stem.get(str(number))
        answer_path = files_by_stem.get(f"{number}.1")
        if not question_path or not answer_path:
            continue
        question = _extract_plain_text(question_path, rag)
        answer = _extract_plain_text(answer_path, rag)
        if not question or not answer:
            continue
        pid = f"qa_upload_{uuid.uuid4().hex[:12]}"
        record = {
            "pid": pid,
            "knowledge_type": "业务问答",
            "number": number,
            "title": f"上传业务问答第{number}题",
            "category": "纪检监察业务题",
            "question": question,
            "answer": answer,
            "question_file": question_path.name,
            "answer_file": answer_path.name,
        }
        (records_dir / f"{pid}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        texts.extend(
            (
                f"【业务问答题目】{record['title']}\n【题目】{question}",
                f"【业务问答参考答案】{record['title']}\n【参考答案】{answer}",
            )
        )
        pids.extend((pid, pid))
        used.update({question_path, answer_path})

    for path in files:
        if path in used:
            continue
        if path.suffix.lower() == ".json":
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                data = None
            records = data if isinstance(data, list) else [data] if isinstance(data, dict) else []
            for record in records:
                question = str(record.get("question") or record.get("题目") or "").strip()
                answer = str(record.get("answer") or record.get("答案") or "").strip()
                if not question or not answer:
                    continue
                pid = str(record.get("pid") or f"qa_upload_{uuid.uuid4().hex[:12]}")
                normalized = {
                    "pid": pid,
                    "knowledge_type": "业务问答",
                    "title": record.get("title", path.stem),
                    "category": record.get("category", "纪检监察业务题"),
                    "question": question,
                    "answer": answer,
                    "question_file": path.name,
                    "answer_file": path.name,
                }
                (records_dir / f"{pid}.json").write_text(
                    json.dumps(normalized, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                texts.extend(
                    (
                        f"【业务问答题目】{normalized['title']}\n【题目】{question}",
                        f"【业务问答参考答案】{normalized['title']}\n【参考答案】{answer}",
                    )
                )
                pids.extend((pid, pid))

    if texts:
        rag.add_documents(texts, pids=pids, save_to_disk=True)
    return len(texts)


def _build_case_records_from_files(files: list[Path], source_dir: Path, rag: LegalRAGapi) -> int:
    json_files: list[Path] = []
    for path in files:
        records_to_write: list[dict] = []
        source_text = ""
        if path.suffix.lower() in {".xlsx", ".xlsm"}:
            records_to_write.extend(_case_records_from_excel(path))
        else:
            source_text = _extract_plain_text(path, rag)
            if not source_text:
                continue
            try:
                data = json.loads(source_text) if path.suffix.lower() == ".json" else None
            except Exception:
                data = None

            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        records_to_write.append(_normalize_case_record(item, source_text, path.name))
            elif isinstance(data, dict):
                records_to_write.append(_normalize_case_record(data, source_text, path.name))
            else:
                structured = _structure_case_with_llm(source_text, path.name)
                records_to_write.append(_normalize_case_record(structured, source_text, path.name))

        for record in records_to_write:
            dest = source_dir / f"{record['pid']}.json"
            dest.write_text(
                json.dumps(record, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            json_files.append(dest)

    for json_path in json_files:
        rag.add_file_documents(str(json_path), save_to_disk=True)
    return rag.get_document_count()


def _get_target_paths(scope: str, private_space: str, knowledge_type: str) -> tuple[Path, Path]:
    cfg = KNOWLEDGE_TYPE_CONFIG[knowledge_type]
    if scope == "私有知识库":
        safe_space = _safe_space_name(private_space)
        if not safe_space:
            raise ValueError("请选择或输入私有知识库名称")
        return _private_source_dir(safe_space, knowledge_type), _private_index_dir(safe_space, knowledge_type)
    return cfg["public_source"], cfg["public_index"]


def ingest_uploaded_files(scope: str, private_space: str, knowledge_type: str, uploaded_files) -> dict:
    source_dir, index_dir = _get_target_paths(scope, private_space, knowledge_type)
    source_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)
    type_key = KNOWLEDGE_TYPE_CONFIG[knowledge_type]["key"]
    raw_dir = source_dir.parent / "uploads"
    saved_files = _save_uploaded_files(uploaded_files, raw_dir)
    if not saved_files:
        return {"files": 0, "chunks": 0, "index": str(index_dir)}

    base_embedding = None
    try:
        base_agent = load_global_agent(st.session_state.get("private_kb_scope", ""))
        base_embedding = getattr(getattr(base_agent, "case_rag", None), "embedding_model", None)
    except Exception:
        base_embedding = None

    rag = LegalRAGapi(
        db_path=str(index_dir),
        source_dir=str(source_dir),
        vector_weight=0.3 if knowledge_type != "业务题库" else 0.4,
        bm25_weight=0.7 if knowledge_type != "业务题库" else 0.6,
        embedding_model=base_embedding,
    )

    chunks = 0
    if type_key in ("lp", "case"):
        if type_key == "case":
            chunks = _build_case_records_from_files(saved_files, source_dir, rag)
        else:
            chunks = _build_statute_records_from_files(saved_files, source_dir, rag)
    elif type_key == "guidance":
        chunks = _build_guidance_records_from_files(saved_files, source_dir, rag)
    else:
        chunks = _build_qa_records_from_files(saved_files, source_dir, rag)

    load_global_agent.clear()
    return {"files": len(saved_files), "chunks": chunks, "index": str(index_dir)}

# ======================
# 数据库引擎（带线程安全）
# ======================
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///user.db')

# SQLite creates the database file, but it does not create missing parent
# directories. Ensure relative paths such as ./data/user.db work on first run.
database_url = make_url(DATABASE_URL)
if database_url.get_backend_name() == "sqlite" and database_url.database:
    database_path = database_url.database
    if database_path != ":memory:":
        if not os.path.isabs(database_path):
            database_path = os.path.abspath(database_path)
        database_parent = os.path.dirname(database_path)
        if database_parent:
            os.makedirs(database_parent, exist_ok=True)

engine = create_engine(
    DATABASE_URL, 
    echo=False, 
    future=True, 
    connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
Base = declarative_base()  # 创建一个基类（Base Class），所有 ORM 模型类都继承自它，从而自动获得与数据库表映射的能力。

# ======================
# 数据模型
# ======================
class User(Base):
    __tablename__ = 'user'
    id = Column(Integer, primary_key=True, index=True)
    phone = Column(String(20), unique=True, nullable=False)
    username = Column(String(50), nullable=False)
    password_hash = Column(String(128), nullable=False)
    role = Column(String(20), nullable=False, default='user')
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    chats = relationship('Chat', backref='user', cascade='all, delete-orphan')

class Chat(Base):
    __tablename__ = 'chat'
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('user.id'), nullable=False)
    title = Column(String(100), nullable=False, default='新对话')
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    messages = relationship('Message', backref='chat', cascade='all, delete-orphan')

class Message(Base):
    __tablename__ = 'message'
    id = Column(Integer, primary_key=True, index=True)
    chat_id = Column(Integer, ForeignKey('chat.id'), nullable=False)
    role = Column(String(10), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    feedback = relationship(
        'ResponseFeedback',
        back_populates='message',
        uselist=False,
        cascade='all, delete-orphan',
    )


class ResponseFeedback(Base):
    __tablename__ = 'response_feedback'
    id = Column(Integer, primary_key=True, index=True)
    message_id = Column(Integer, ForeignKey('message.id'), nullable=False, unique=True)
    rating = Column(String(10), nullable=False)
    comment = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
    message = relationship('Message', back_populates='feedback')

# ======================
# 创建数据库表
# ======================
Base.metadata.create_all(bind=engine)


def _save_response_feedback(message_id: int, rating: str, comment: str = ""):
    """创建或更新一条 AI 回答反馈。"""
    with get_db() as db:
        feedback = db.query(ResponseFeedback).filter_by(message_id=message_id).first()
        if feedback is None:
            feedback = ResponseFeedback(
                message_id=message_id,
                rating=rating,
                comment=comment.strip() or None,
            )
            db.add(feedback)
        else:
            feedback.rating = rating
            feedback.comment = comment.strip() or None
            feedback.updated_at = datetime.now(timezone.utc)
        db.commit()


def _render_response_feedback(
    message_id: int,
    saved_rating: str | None = None,
    saved_comment: str = "",
):
    """在 AI 回答下方渲染好、中、差评价组件。"""
    rating_labels = {
        "good": "好",
        "medium": "中",
        "bad": "差",
        "好": "好",
        "中": "中",
        "差": "差",
    }
    rating_codes = {"好": "good", "中": "medium", "差": "bad"}
    saved_rating_label = rating_labels.get(saved_rating)
    saved_rating_code = rating_codes.get(saved_rating, saved_rating)
    rating_key = f"feedback_rating_{message_id}"
    if saved_rating_label and st.session_state.get(rating_key) is None:
        st.session_state[rating_key] = saved_rating_label

    st.markdown('<div class="feedback-row">', unsafe_allow_html=True)
    feedback_cols = st.columns([0.45, 0.45, 0.45, 10], gap="small")
    with feedback_cols[0]:
        good_clicked = st.button("好", key=f"{rating_key}_good", use_container_width=True)
    with feedback_cols[1]:
        medium_clicked = st.button("中", key=f"{rating_key}_medium", use_container_width=True)
    with feedback_cols[2]:
        bad_clicked = st.button("差", key=f"{rating_key}_bad", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    rating = st.session_state.get(rating_key)
    if good_clicked:
        rating = "好"
        st.session_state[rating_key] = rating
    elif medium_clicked:
        rating = "中"
        st.session_state[rating_key] = rating
    elif bad_clicked:
        rating = "差"
        st.session_state[rating_key] = rating

    if rating == "好":
        if saved_rating_code != "good" or saved_comment:
            _save_response_feedback(message_id, "good")
            st.toast("感谢反馈，已保存。", icon="✅")
            st.rerun()
        return

    if rating in ("中", "差"):
        edit_key = f"feedback_edit_{message_id}"
        rating_code = rating_codes[rating]
        has_saved_comment = saved_rating_code == rating_code and bool(saved_comment)
        is_editing = st.session_state.get(edit_key, False)

        if has_saved_comment and not is_editing:
            if st.button(
                "修改意见",
                key=f"feedback_edit_button_{message_id}_{rating}",
                type="tertiary",
            ):
                st.session_state[edit_key] = True
                st.rerun()
            return

        comment_default = saved_comment if saved_rating_code == rating_code else ""
        comment = st.text_area(
            "请输入您的意见",
            value=comment_default,
            placeholder="请说明回答中需要改进的地方……",
            max_chars=1000,
            key=f"feedback_comment_{message_id}_{rating}",
            height=90,
        )
        if st.button(
            "提交意见",
            key=f"feedback_submit_{message_id}_{rating}",
            disabled=not bool(comment and comment.strip()),
            type="secondary",
        ):
            _save_response_feedback(message_id, rating_code, comment)
            st.session_state[edit_key] = False
            st.toast("感谢反馈，您的意见已保存。", icon="✅")
            st.rerun()


def _render_chat_input_form(form_key: str, disabled: bool = False) -> str | None:
    """底部输入框。使用 Streamlit 原生 chat_input 保持底部定位。"""
    return st.chat_input("请输入问题...", key=form_key, disabled=disabled)


# ======================
# 从 DB 加载历史对话
# ======================
def load_chat_history_from_db(chat_id: int, limit: int = 10):
    """
    从数据库加载最近 limit 轮对话，转换为 LangChain Message 对象。
    limit=10 意味着加载最近 10 条消息（即 5 轮问答）。
    """
    with get_db() as db:
        # 按时间正序排列，取最后 limit 条
        msgs = db.query(Message).filter_by(chat_id=chat_id).order_by(Message.created_at.asc()).all()
        
        # 截取最近 N 条，防止一次性加载太多导致 Token 溢出
        # 如果消息总数超过 limit，只取最后 limit 条
        recent_msgs = msgs[-limit:] if len(msgs) > limit else msgs
        
        history = []
        for m in recent_msgs:
            if m.role == "user":
                history.append(HumanMessage(content=m.content))
            elif m.role == "assistant":
                history.append(AIMessage(content=m.content))
            # 忽略 system 或其他角色的消息，除非你有特殊需求
            
        return history





# ======================
# RAG 模型初始化
# ======================
def _load_private_rag(space: str, knowledge_type: str, embedding_model, vector_weight: float, bm25_weight: float):
    safe_space = _safe_space_name(space)
    if not safe_space:
        return None
    index_dir = _private_index_dir(safe_space, knowledge_type)
    source_dir = _private_source_dir(safe_space, knowledge_type)
    if not index_dir.exists():
        return None
    print(f"📂 准备加载私有{knowledge_type}库：{index_dir}")
    return LegalRAGapi(
        db_path=str(index_dir),
        source_dir=str(source_dir),
        vector_weight=vector_weight,
        bm25_weight=bm25_weight,
        embedding_model=embedding_model,
    )


@st.cache_resource(show_spinner="🧠 正在加载纪检监察知识库与智能体引擎（仅首次启动需要）...")
def load_global_agent(private_space: str = ""):
    """加载 RAG 系统并编译 Agent 图"""
    try:
        print(f"📂 准备加载案例库：{case_db_path}")
        case_rag = LegalRAGapi(
            db_path=case_db_path,
            source_dir=KNOWLEDGE_BASE_CASE_FOLDER,
            vector_weight=0.3,
            bm25_weight=0.7
        )
        print(f"📂 准备加载法条库：{lp_db_path}")
        lp_rag = LegalRAGapi(
            db_path=lp_db_path,
            vector_weight=0.35,
            bm25_weight=0.65,
            embedding_model=case_rag.embedding_model,
        )
        guidance_rag = None
        if os.path.exists(guidance_db_path):
            print(f"📂 准备加载办案规范库：{guidance_db_path}")
            guidance_rag = LegalRAGapi(
                db_path=guidance_db_path,
                source_dir=KNOWLEDGE_BASE_GUIDANCE_FOLDER,
                vector_weight=0.3,
                bm25_weight=0.7,
                embedding_model=case_rag.embedding_model,
            )
        else:
            print(
                f"ℹ️ 办案规范库尚未构建：{guidance_db_path}，"
                "当前先使用案例库和法规库。"
            )
        qa_rag = None
        if os.path.exists(qa_db_path):
            print(f"📂 准备加载业务问答库：{qa_db_path}")
            qa_rag = LegalRAGapi(
                db_path=qa_db_path,
                source_dir=KNOWLEDGE_BASE_QA_FOLDER,
                vector_weight=0.4,
                bm25_weight=0.6,
                embedding_model=case_rag.embedding_model,
            )
        else:
            print(f"ℹ️ 业务问答库尚未构建：{qa_db_path}")
        initialize_vector_database(case_rag, lp_rag)

        private_space = _safe_space_name(private_space)
        if private_space:
            private_case_rag = _load_private_rag(private_space, "案例", case_rag.embedding_model, 0.3, 0.7)
            private_lp_rag = _load_private_rag(private_space, "法律法规", case_rag.embedding_model, 0.35, 0.65)
            private_guidance_rag = _load_private_rag(private_space, "办案规范", case_rag.embedding_model, 0.3, 0.7)
            private_qa_rag = _load_private_rag(private_space, "业务题库", case_rag.embedding_model, 0.4, 0.6)
            case_rag = CombinedRAG(case_rag, private_case_rag)
            lp_rag = CombinedRAG(lp_rag, private_lp_rag)
            if guidance_rag is not None or private_guidance_rag is not None:
                guidance_rag = CombinedRAG(guidance_rag, private_guidance_rag)
            if qa_rag is not None or private_qa_rag is not None:
                qa_rag = CombinedRAG(qa_rag, private_qa_rag)
    except Exception as e:
        st.error(f"❌ 加载向量库失败：{e}")
        st.stop()
        return None
    #  创建 Agent，传入两个实例
    agent_graph = create_legal_agent(
        case_rag=case_rag,
        lp_rag=lp_rag,
        guidance_rag=guidance_rag,
        qa_rag=qa_rag,
    )
    # 暴露 case_rag 引用，供引用溯源的前端渲染使用
    agent_graph.case_rag = case_rag
    agent_graph.guidance_rag = guidance_rag
    agent_graph.qa_rag = qa_rag
    return agent_graph

# ======================
# 辅助函数
# ======================
def get_db():
    """获取数据库会话"""
    return SessionLocal()

def get_current_user_id():
    """获取当前用户 ID，缓存到 session_state 避免重复查询"""
    if "user_id" in st.session_state:
        return st.session_state.user_id
    with get_db() as db:
        user = db.query(User).filter_by(phone="demo_user").first()
        if not user:
            from werkzeug.security import generate_password_hash
            demo_password = os.getenv("DEMO_USER_PASSWORD")
            if not demo_password:
                raise RuntimeError(
                    "DEMO_USER_PASSWORD 未设置。请在 .env 中为演示用户配置一个强密码后再启动应用。"
                )
            user = User(
                phone="demo_user",
                username="DemoUser",
                role="user",
                password_hash=generate_password_hash(demo_password)
            )
            db.add(user)
            db.commit()
        st.session_state.user_id = user.id
        st.session_state.username = user.username
        return user.id

def get_username():
    """获取当前用户名（用于显示）"""
    if "username" in st.session_state:
        return st.session_state.username
    get_current_user_id()
    return st.session_state.get("username", "未知用户")


DEFAULT_CHAT_TITLES = {"新对话", "新对话 ", ""}


def _build_chat_title(question: str, max_length: int = 30) -> str:
    """用首条用户问题生成简洁、稳定的历史对话标题。"""
    import re

    title = re.sub(r"\s+", " ", question or "").strip()
    title = title.lstrip("#*-—:： ").strip()
    if not title:
        return "未命名对话"
    if len(title) > max_length:
        return title[:max_length].rstrip("，。！？；：,.!?;: ") + "…"
    return title


def _backfill_default_chat_titles(db, user_id: int) -> bool:
    """为已有默认标题对话补上第一条用户问题，不覆盖手动标题。"""
    chats = db.query(Chat).filter_by(user_id=user_id).all()
    changed = False
    for chat in chats:
        if (chat.title or "").strip() not in DEFAULT_CHAT_TITLES:
            continue
        first_user_message = (
            db.query(Message)
            .filter_by(chat_id=chat.id, role="user")
            .order_by(Message.created_at.asc(), Message.id.asc())
            .first()
        )
        if first_user_message:
            chat.title = _build_chat_title(first_user_message.content)
            changed = True
    if changed:
        db.commit()
    return changed


# ======================
# 引用溯源：内嵌按钮 + 侧边栏面板（纯 Streamlit 原生组件）
# ======================

def _normalize_reference_markdown(text: str) -> str:
    """清理模型在引用标记周围生成的 Markdown，避免拆分按钮后出现孤立符号。"""
    import re

    # 引用按钮会替代 [ref_N]，因此包裹它的加粗、斜体和代码标记必须一并移除。
    normalized = re.sub(
        r'(?:\*\*|__|`|\*)(\[ref_\d+\])(?:\*\*|__|`|\*)',
        r'\1',
        text,
    )

    # 模型偶尔会把引用写成独立列表项，如 "- [ref_0]：说明"。
    # 按钮本身是块级组件，去掉这个列表前缀可避免页面出现空项目符号。
    normalized = re.sub(
        r'(?m)^[ \t]*[-+*]\s*(\[ref_\d+\])\s*[：:]\s*',
        r'\1\n\n',
        normalized,
    )
    return normalized


def _render_case_reference_details(case_rag, pid: str):
    """在引用位置展示案例中的事实、违反内容和对应依据。"""
    if case_rag is None:
        st.warning("案例库尚未加载，暂时无法查看引用详情。")
        return

    record = case_rag.get_case_record(pid)
    if not record:
        st.warning(f"无法加载案例详情（案例编号：{pid}）。")
        return

    st.markdown(f"#### 案例详情 `{pid}`")

    reason = str(record.get("reason") or "").strip()
    result = str(record.get("result") or "").strip()
    fact = str(record.get("fact") or "").strip()
    charges = record.get("charge") or []
    articles = record.get("article") or []

    if reason:
        st.markdown("**违反内容/问题性质**")
        st.write(reason)
    if result:
        st.markdown("**对应依据/处理结果**")
        st.write(result)
    if fact:
        st.markdown("**案件事实**")
        st.write(fact)
    if charges:
        st.markdown("**涉及罪名**")
        st.write("、".join(str(item) for item in charges))
    if articles:
        article_text = "、".join(f"第{item}条" for item in articles)
        st.markdown("**引用法条**")
        st.write(article_text)

    source = record.get("source")
    if isinstance(source, dict):
        source_parts = [
            str(source.get(key)).strip()
            for key in ("file", "sheet", "row")
            if source.get(key) not in (None, "")
        ]
        if source_parts:
            st.caption("数据来源：" + " / ".join(source_parts))


def _render_statute_reference_details(reference: dict):
    """展示法规检索结果中的法规名称、条号和原文。"""
    content = str(reference.get("content") or "").strip()
    if not content:
        st.warning("该法规引用没有可展示的条文内容。")
        return

    st.markdown("#### 法规依据原文")
    st.markdown(content)
    st.caption("来源：本地党纪法规知识库")


def _render_guidance_reference_details(reference: dict):
    """展示办案规范文件名称、文号、页码和原文。"""
    title = str(reference.get("title") or "办案规范").strip()
    document_number = str(reference.get("document_number") or "").strip()
    content = str(reference.get("content") or "").strip()
    page_start = reference.get("page_start")
    page_end = reference.get("page_end")

    st.markdown(f"#### 办案规范原文：{title}")
    if document_number:
        st.markdown(f"**文号**：{document_number}")
    heading_path = [
        str(item).strip()
        for item in reference.get("heading_path", [])
        if str(item).strip()
    ]
    if heading_path:
        st.markdown(f"**所属章节**：{' > '.join(heading_path)}")
    if reference.get("section_title"):
        st.markdown(f"**条款范围**：{reference['section_title']}")
    if page_start:
        page_text = (
            f"第{page_start}页"
            if not page_end or page_end == page_start
            else f"第{page_start}-{page_end}页"
        )
        st.markdown(f"**页码**：{page_text}")
    if reference.get("confidentiality"):
        st.warning("该文件标注为内部资料，请按权限使用并注意保密。")
    if content:
        st.markdown(content)
    else:
        st.warning("该规范引用没有可展示的正文。")
    if reference.get("source_file"):
        st.caption(f"来源文件：{reference['source_file']}")


def _render_qa_reference_details(reference: dict):
    """展示业务题目及其配套参考答案。"""
    title = str(reference.get("title") or "纪检监察业务问答").strip()
    category = str(reference.get("category") or "").strip()
    question = str(reference.get("question") or "").strip()
    answer = str(reference.get("answer") or "").strip()

    st.markdown(f"#### 参考问答：{title}")
    if category:
        st.markdown(f"**类型**：{category}")
    st.info("该内容是业务学习参考答案，具体办理仍应以现行法规和正式制度文件为准。")
    if question:
        st.markdown("**题目**")
        st.write(question)
    if answer:
        st.markdown("**参考答案**")
        st.write(answer)
    source_files = [
        str(reference.get(key) or "").strip()
        for key in ("question_file", "answer_file")
        if str(reference.get(key) or "").strip()
    ]
    if source_files:
        st.caption("来源文件：" + " / ".join(source_files))


def _render_response_with_inline_refs(
    text: str,
    ref_map: dict,
    case_rag,
    reference_scope: str = "",
):
    """将响应文本按 [ref_N] 拆分，markdown 文本与 st.button 交替渲染。

    按钮直接嵌入报告原文位置，点击后在原位置展开案例详情。
    按钮 key 必须跨 Streamlit rerun 保持稳定，否则点击事件会丢失。
    """
    import hashlib
    import re

    response_id = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    stable_scope = f"{reference_scope}_{response_id}"
    selected_key = f"_selected_reference_{stable_scope}"
    btn_idx = 0  # 递增索引，防止同一回答中同一 pid 被多次引用时的 key 冲突

    text = _normalize_reference_markdown(text)
    parts = re.split(r'(\[ref_\d+\])', text)

    for part in parts:
        if part in ref_map:
            reference = ref_map[part]
            is_statute = (
                isinstance(reference, dict)
                and reference.get("type") == "statute"
            )
            is_guidance = (
                isinstance(reference, dict)
                and reference.get("type") == "guidance"
            )
            is_qa = (
                isinstance(reference, dict)
                and reference.get("type") == "qa"
            )
            reference_value = (
                reference.get("content", "")
                if is_statute
                else (
                    reference.get("pid", "")
                    if is_qa
                    else str(reference)
                )
            )
            reference_id = hashlib.sha1(
                reference_value.encode("utf-8")
            ).hexdigest()[:10]
            button_label = (
                f"查看法规原文 {part}"
                if is_statute
                else (
                    f"查看办案规范 {part}"
                    if is_guidance
                    else (
                        f"查看参考问答 {part}"
                        if is_qa
                        else f"查看案例依据 {part}"
                    )
                )
            )
            button_help = (
                "点击查看该法规条文原文"
                if is_statute
                else (
                    "点击查看该办案规范的文件信息、页码和原文"
                    if is_guidance
                    else (
                        "点击查看相似业务题目及配套参考答案"
                        if is_qa
                        else f"点击查看案例 {reference} 的违反内容、依据和案件事实"
                    )
                )
            )
            if st.button(
                button_label,
                key=f"inref_{stable_scope}_{btn_idx}_{reference_id}",
                help=button_help,
                type="secondary",
            ):
                if st.session_state.get(selected_key) == part:
                    st.session_state.pop(selected_key, None)
                else:
                    st.session_state[selected_key] = part
            if st.session_state.get(selected_key) == part:
                with st.container(border=True):
                    if is_statute:
                        _render_statute_reference_details(reference)
                    elif is_guidance:
                        _render_guidance_reference_details(reference)
                    elif is_qa:
                        _render_qa_reference_details(reference)
                    else:
                        _render_case_reference_details(case_rag, str(reference))
            btn_idx += 1
        elif part.strip():
            st.markdown(part)


def _render_sidebar_doc(case_rag):
    """兼容旧会话中已保存的侧边栏案例引用状态。

    新引用直接在回答内展开；旧状态仍可在侧边栏查看并关闭。
    """
    pid = st.session_state.get('_sidebar_pid')
    if not pid or case_rag is None:
        return

    with st.sidebar:
        st.divider()
        title_col, close_col = st.columns([5, 1])
        with title_col:
            st.subheader(f"📋 案例 {pid}")
        with close_col:
            if st.button("✕", key="close_sidepanel", help="关闭侧边文档面板"):
                st.session_state.pop('_sidebar_pid', None)
                st.rerun()

        _render_case_reference_details(case_rag, pid)


def render_knowledge_upload_page():
    st.title("上传知识库")
    st.caption("上传文件后会自动写入对应知识库并增量更新向量索引。扫描版 PDF 建议先用 OCR 脚本处理后再入库。")

    top_left, top_right = st.columns([1, 5])
    with top_left:
        if st.button("返回对话", key="back_to_chat_from_kb", use_container_width=True):
            st.session_state.current_page = "chat"
            st.rerun()
    with top_right:
        if st.button("查看知识库", key="go_to_kb_view_from_upload", use_container_width=True):
            st.session_state.current_page = "knowledge_view"
            st.rerun()

    st.divider()
    scope = st.radio(
        "上传到",
        options=["公共知识库", "私有知识库"],
        horizontal=True,
        key="upload_scope",
    )

    private_space = ""
    if scope == "私有知识库":
        spaces = list_private_spaces()
        options = ["新建私有知识库"] + spaces
        selected = st.selectbox("选择私有知识库", options=options, key="upload_private_space_select")
        if selected == "新建私有知识库":
            private_space = st.text_input(
                "私有知识库名称",
                placeholder="例如：第一监督室知识库",
                key="upload_private_space_new",
            )
        else:
            private_space = selected
        if private_space:
            st.info(f"本次上传将写入私有知识库：{_safe_space_name(private_space)}")

    knowledge_type = st.selectbox(
        "知识库类型",
        options=list(KNOWLEDGE_TYPE_CONFIG.keys()),
        key="upload_knowledge_type",
    )

    help_text = {
        "法律法规": "支持上传单部法规原文 Word/PDF/TXT。系统会自动按“第X条”拆成一行一条，格式特别乱时会尝试调用当前模型重构；建议不要把多部法规混在同一个文件里。",
        "案例": "支持 Word/PDF/TXT/JSON/XLSX。Excel 会按“首行为表头、每行一个案例”入库，并自动识别案件事实、处理结果、依据等常见列名。",
        "办案规范": "支持 PDF、DOC/DOCX、TXT、XLSX。扫描版 PDF 如抽取效果差，请先运行 OCR 预处理。",
        "业务题库": "支持编号 DOCX 配对：1.docx 为题目，1.1.docx 为答案；也支持包含 question/answer 的 JSON。",
    }[knowledge_type]
    st.info(help_text)

    allowed_types = {
        "法律法规": ["txt", "doc", "docx", "pdf"],
        "案例": ["txt", "doc", "docx", "pdf", "json", "xlsx", "xlsm"],
        "办案规范": ["pdf", "doc", "docx", "txt", "xlsx"],
        "业务题库": ["docx", "json", "txt", "doc", "pdf"],
    }[knowledge_type]
    uploaded_files = st.file_uploader(
        "选择文件",
        type=allowed_types,
        accept_multiple_files=True,
        key=f"kb_upload_{knowledge_type}",
    )

    if uploaded_files:
        st.write("待上传文件：")
        for item in uploaded_files:
            st.caption(f"- {item.name} ({item.size / 1024:.1f} KB)")

    disabled = not uploaded_files or (scope == "私有知识库" and not _safe_space_name(private_space))
    if st.button("上传并更新知识库", type="primary", disabled=disabled, key="upload_ingest_btn"):
        try:
            with st.spinner("正在保存文件、分片并更新索引，请稍候..."):
                result = ingest_uploaded_files(scope, private_space, knowledge_type, uploaded_files)
            st.success(
                f"入库完成：文件 {result['files']} 个，新增/处理文本块 {result['chunks']} 个。"
            )
            st.caption(f"索引目录：{result['index']}")
            if scope == "私有知识库":
                st.session_state.private_kb_scope = _safe_space_name(private_space)
            st.info("索引已刷新，新对话或刷新页面后会使用最新知识库。")
        except Exception as e:
            st.error(f"入库失败：{e}")
            with st.expander("查看错误详情"):
                st.code(traceback.format_exc())


# ======================
# 主页面
# ======================
def chat_page():
    # 获取用户信息
    user_id = get_current_user_id()
    username = get_username()
    
    # ======================
    # 📍 侧边栏逻辑 (Sidebar)
    # ======================
    with st.sidebar:
        st.title("纪检监察业务助手")
        st.caption(f"👤 当前用户：{username}")
        col_upload, col_view = st.columns(2)
        with col_upload:
            if st.button("上传知识库", key="open_kb_upload", use_container_width=True):
                st.session_state.current_page = "knowledge_upload"
                st.rerun()
        with col_view:
            if st.button("查看知识库", key="open_kb_view", use_container_width=True):
                st.session_state.current_page = "knowledge_view"
                st.rerun()
        if st.button("返回对话", key="open_chat_page", use_container_width=True):
            st.session_state.current_page = "chat"
            st.rerun()
        st.divider()

        st.subheader("知识库范围")
        private_spaces = list_private_spaces()
        active_options = ["仅公共知识库"] + private_spaces
        current_scope = st.session_state.get("private_kb_scope", "")
        default_scope_label = current_scope if current_scope in private_spaces else "仅公共知识库"
        selected_scope_label = st.selectbox(
            "私有知识库",
            options=active_options,
            index=active_options.index(default_scope_label),
            key="active_private_kb_select",
            help="选择后，检索时会同时使用公共知识库和该私有知识库。",
        )
        selected_private_scope = "" if selected_scope_label == "仅公共知识库" else selected_scope_label
        if st.session_state.get("private_kb_scope", "") != selected_private_scope:
            st.session_state.private_kb_scope = selected_private_scope
            load_global_agent.clear()
            st.rerun()
        if selected_private_scope:
            st.caption(f"当前检索：公共 + {selected_private_scope}")
        else:
            st.caption("当前检索：仅公共知识库")
        st.divider()
        
        # 1. 新建对话功能区
        st.subheader("🆕 新建对话")
        new_title = st.text_input("对话标题", value="新对话", key="new_chat_title")
        if st.button("创建新对话", key="create_chat_btn", use_container_width=True):
            if new_title and new_title.strip():
                with get_db() as db:
                    new_chat = Chat(user_id=user_id, title=new_title.strip())
                    db.add(new_chat)
                    db.commit()

                    init_msg = Message(
                        chat_id=new_chat.id,
                        role="assistant",
                        content="您好！我是类案检索智能助手，请问有什么可以帮您的吗？"
                    )
                    db.add(init_msg)
                    db.commit()
                    st.session_state._auto_select_chat_id = new_chat.id
                st.rerun()
            else:
                st.error("请输入对话标题")
        
        st.divider()
        
        # 2. 选择历史对话区
        st.subheader("💬 历史对话")
        with get_db() as db:
            _backfill_default_chat_titles(db, user_id)
            chats = db.query(Chat).filter_by(user_id=user_id).order_by(Chat.updated_at.desc()).all()
            chat_options = {f"#{c.id} | {c.title}": c.id for c in chats}

        # 自动选中新建的对话：直接写入 session_state，确保 selectbox 读取到正确值
        auto_id = st.session_state.pop("_auto_select_chat_id", None)
        if auto_id:
            for label, cid in chat_options.items():
                if cid == auto_id:
                    st.session_state.chat_selector = label
                    break
        else:
            # 标题自动更新后，旧的 selectbox 文本已不在 options 中。
            # 从 "#ID | 旧标题" 恢复对话 ID，再映射到新的标题文本。
            previous_label = st.session_state.get("chat_selector")
            if previous_label and previous_label not in chat_options:
                import re

                id_match = re.match(r"#(\d+)\s*\|", previous_label)
                previous_chat_id = int(id_match.group(1)) if id_match else None
                replacement_label = next(
                    (
                        label
                        for label, cid in chat_options.items()
                        if cid == previous_chat_id
                    ),
                    None,
                )
                if replacement_label:
                    st.session_state.chat_selector = replacement_label
                elif previous_label != "(请新建或选择对话)":
                    st.session_state.chat_selector = "(请新建或选择对话)"

        selected_label = st.selectbox(
            "选择对话",
            options=["(请新建或选择对话)"] + list(chat_options.keys()),
            key="chat_selector"
        )

        # 解析选中的对话 ID
        current_chat_id = None
        current_chat_title = None
        if selected_label != "(请新建或选择对话)" and selected_label in chat_options:
            current_chat_id = chat_options[selected_label]
            current_chat_title = selected_label.split(" | ", 1)[1] if " | " in selected_label else selected_label

            # 删除按钮
            if st.button("🗑️ 删除当前对话", key=f"del_{current_chat_id}", type="secondary", use_container_width=True):
                with get_db() as db:
                    chat_to_del = db.get(Chat, current_chat_id)
                    if chat_to_del:
                        db.delete(chat_to_del)
                        db.commit()
                st.rerun()
            
        st.divider()

        st.subheader("🤖 模型设置")

        # 定义可用模型列表 (名称 -> 配置字典)
        MODEL_CONFIGS = {
            "DeepSeek-V3": {
                "id": os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
                "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
                "api_key_env": "DEEPSEEK_API_KEY"
            },
            "通义千问3.5-Plus": {
                "id": "qwen3.5-plus", 
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", 
                "api_key_env": "DASHSCOPE_API_KEY"
            },
            "智谱 GLM-4.7": {
                "id": "glm-4.7", 
                "base_url": "https://open.bigmodel.cn/api/paas/v4", 
                "api_key_env": "ZHIPUAI_API_KEY"
            }
        }
        
        # 下拉框选择
        selected_model_name = st.selectbox(
            "选择生成模型",
            options=list(MODEL_CONFIGS.keys()),
            index=0, # 默认选第一个
            key="model_selector"
        )
        
        # 获取当前选中模型的配置
        current_model_config = MODEL_CONFIGS[selected_model_name]
        
        # 从环境变量获取对应的 API Key
        api_key = os.getenv(current_model_config["api_key_env"])
        
        if not api_key:
            st.error(f"❌ 未找到 {current_model_config['api_key_env']} 环境变量，请检查 .env 文件")
            st.stop()
        
                # 存储当前配置到 Session State
        st.session_state.current_api_key = api_key
        st.session_state.current_base_url = current_model_config["base_url"]
        st.session_state.current_model_name = current_model_config["id"]

        st.info(f"当前使用：**{selected_model_name}**")
        st.caption(f"模型ID: `{current_model_config['id']}`")
        



    if st.session_state.get("current_page") == "knowledge_upload":
        render_knowledge_upload_page()
        return
    if st.session_state.get("current_page") == "knowledge_view":
        render_knowledge_view_page()
        return

    # 1. 获取全局 Agent
    agent_graph = load_global_agent(st.session_state.get("private_kb_scope", ""))
    

    # ======================
    # 📍 主聊天区域 (Main Area)
    # ======================
    if current_chat_id:
        st.subheader(f"💬 {current_chat_title}")

        # 显示历史消息（限制显示最近50条，避免长对话渲染卡顿）
        with get_db() as db:
            msgs = db.query(Message).filter_by(chat_id=current_chat_id).order_by(Message.created_at.asc()).all()
            msg_data = [
                (
                    m.id,
                    m.role,
                    m.content,
                    m.feedback.rating if m.feedback else None,
                    m.feedback.comment if m.feedback and m.feedback.comment else "",
                )
                for m in msgs
            ]
        if len(msg_data) > 50:
            st.caption(f"📜 仅显示最近 50 条消息（共 {len(msg_data)} 条）")
            msg_data = msg_data[-50:]

        # 检查是否当前对话的最后一条 bot 回复有内嵌引用数据
        inline_key = f'_inline_{current_chat_id}'
        inline_data = st.session_state.get(inline_key)

        for i, (message_id, role, content, saved_rating, saved_comment) in enumerate(msg_data):
            is_last = (i == len(msg_data) - 1)
            # 最后一条 assistant 消息若有引用数据，用内嵌按钮渲染
            if role == "assistant" and is_last and inline_data:
                with st.chat_message("assistant"):
                    _render_response_with_inline_refs(
                        inline_data['text'], inline_data['ref_map'],
                        getattr(agent_graph, 'case_rag', None),
                        reference_scope=f"chat_{current_chat_id}",
                    )
                    _render_response_feedback(
                        message_id,
                        saved_rating=saved_rating,
                        saved_comment=saved_comment,
                    )
            else:
                with st.chat_message("user" if role == "user" else "assistant"):
                    st.markdown(content)
                    if role == "assistant":
                        _render_response_feedback(
                            message_id,
                            saved_rating=saved_rating,
                            saved_comment=saved_comment,
                        )

        # 用户输入。使用原生 chat_input 保持底部定位，按钮样式由 CSS 兼容旧 Chrome。
        user_input = _render_chat_input_form(f"chat_input_{current_chat_id}")
        if user_input and user_input.strip():
            if len(user_input) > 15000:
                st.error("❌ 问题太长，请限制在 15000 字符以内")
            else:
                try:

                    # 从 Session State 获取当前用户选择的模型配置
                    config_key = st.session_state.get("current_api_key")
                    config_url = st.session_state.get("current_base_url")
                    config_model = st.session_state.get("current_model_name")

                    # 立即显示用户消息
                    with st.chat_message("user"):
                        st.write(user_input.strip())

                    # 保存用户消息
                    with get_db() as db:
                        msg = Message(chat_id=current_chat_id, role="user", content=user_input.strip())
                        db.add(msg)
                        chat_obj = db.get(Chat, current_chat_id)
                        if (
                            chat_obj
                            and (chat_obj.title or "").strip() in DEFAULT_CHAT_TITLES
                        ):
                            chat_obj.title = _build_chat_title(user_input)
                        db.commit()

                    conversation_id = f"chat_{current_chat_id}"

                    def agent_response_generator(query, conv_id, chat_id, api_key, base_url, model_name, status_container=None, result_meta=None):
                        """
                        使用 queue + 线程实现真正的流式输出。
                        LangGraph 在后台线程中运行，generate_node 的每个 chunk 通过 queue 传递到前台。
                        状态更新也通过 queue 传递，避免从子线程直接操作 Streamlit UI。
                        result_meta: 可变容器，用于从子线程回传 ref_map 等元数据到主线程。
                        """
                        chat_history = load_chat_history_from_db(chat_id, limit=10)
                        print(f"ℹ️ 已加载 {len(chat_history)} 条历史消息进入上下文")

                        chunk_queue = queue.Queue()
                        _SENTINEL = object()

                        # 状态消息用特殊前缀标记，与正常文本 chunk 区分
                        _STATUS_PREFIX = "__STATUS__:"

                        def stream_callback(text):
                            chunk_queue.put(text)

                        initial_state = {
                            "query": query,
                            "conversation_id": conv_id,
                            "intent": "",
                            "retrieved_contexts": [],
                            "draft_answer": "",
                            "reflection_feedback": "",
                            "final_answer": "",
                            "generation_error": "",
                            "retry_count": 0,
                            "api_key": api_key,
                            "base_url": base_url,
                            "model_name": model_name,
                            "chat_history": chat_history,
                            "reformulated_query": "",
                            "ref_map": {}
                        }

                        def run_graph():
                            try:
                                agent_graph.nodes_ref.stream_callback = stream_callback

                                last_intent = None
                                last_context_count = -1

                                for event in agent_graph.stream(initial_state, stream_mode="values"):
                                    current_intent = event.get('intent')
                                    if current_intent and current_intent != last_intent:
                                        intent_map = {
                                            "statute": "党纪法规查询",
                                            "complex": "线索研判",
                                            "business_qa": "业务问答",
                                            "normal": "正常问答",
                                        }
                                        label = intent_map.get(current_intent, "分析中")
                                        chunk_queue.put(f"{_STATUS_PREFIX}🧠 **意图识别**: {label}")
                                        last_intent = current_intent

                                    current_contexts = event.get('retrieved_contexts')
                                    if current_contexts and len(current_contexts) != last_context_count:
                                        chunk_queue.put(
                                            f"{_STATUS_PREFIX}🔍 **纪检监察检索**: "
                                            f"已找到 {len(current_contexts)} 条相关依据"
                                        )
                                        last_context_count = len(current_contexts)

                                    current_feedback = event.get('reflection_feedback')
                                    if current_feedback and current_feedback != "":
                                        chunk_queue.put(f"{_STATUS_PREFIX}⚠️ **自我修正**: {current_feedback}")

                                    # 捕获引用溯源映射表，传回主线程
                                    current_ref_map = event.get('ref_map')
                                    if current_ref_map and result_meta is not None:
                                        result_meta['ref_map'] = current_ref_map
                            except Exception as e:
                                print(f"❌ Agent 执行异常：{e}")
                                import traceback
                                traceback.print_exc()
                                chunk_queue.put(f"❌ 发生错误：{str(e)}")
                            finally:
                                agent_graph.nodes_ref.stream_callback = None
                                chunk_queue.put(_SENTINEL)

                        thread = threading.Thread(target=run_graph, daemon=True)
                        thread.start()

                        while True:
                            item = chunk_queue.get()
                            if item is _SENTINEL:
                                break
                            # 状态消息写入 status_container，不 yield 给 write_stream
                            if isinstance(item, str) and item.startswith(_STATUS_PREFIX):
                                if status_container:
                                    status_container.write(item[len(_STATUS_PREFIX):])
                            else:
                                yield item

                    # ======================
                    # 📍 UI 渲染：流式输出 + 内嵌引用按钮
                    # ======================
                    result_meta = {}  # 子线程回传元数据的可变容器

                    with st.chat_message("assistant"):
                        with st.status(f"🤖 {selected_model_name} 智能体正在思考...", expanded=True) as status:

                            # 阶段 1：流式输出（显示打字机效果）
                            full_response = ""
                            stream_placeholder = st.empty()
                            gen = agent_response_generator(
                                user_input,
                                conversation_id,
                                current_chat_id,
                                config_key,
                                config_url,
                                config_model,
                                status_container=status,
                                result_meta=result_meta
                            )
                            for chunk in gen:
                                if chunk.startswith("\x00REPLACE\x00"):
                                    full_response = chunk[len("\x00REPLACE\x00"):]
                                    status.write("⚠️ 流式连接中断，已自动重试并恢复完整回答。")
                                else:
                                    full_response += chunk
                                stream_placeholder.markdown(full_response + " ▌")

                            # 阶段 2：流式完成后，清除状态框内的占位符。
                            # 最终回答必须渲染在 status 外部，否则 status 收起时正文也会被折叠。
                            stream_placeholder.empty()

                            if full_response and "❌" not in str(full_response):
                                status.update(label="回答生成完毕", state="complete", expanded=False)
                            elif full_response and "❌" in str(full_response):
                                status.update(label="生成失败", state="error", expanded=False)

                        # 阶段 3：在 status 外部渲染最终回答，避免回答内容随状态框折叠。
                        ref_map = result_meta.get('ref_map', {})
                        case_rag = getattr(agent_graph, 'case_rag', None)
                        if ref_map and full_response and "❌" not in str(full_response):
                            # 持久化到 session_state，确保按钮在后续 rerun 中仍然渲染
                            st.session_state[f'_inline_{current_chat_id}'] = {
                                'text': full_response,
                                'ref_map': ref_map
                            }
                            _render_response_with_inline_refs(
                                full_response,
                                ref_map,
                                case_rag,
                                reference_scope=f"chat_{current_chat_id}",
                            )
                        else:
                            # 无引用时清除旧数据
                            st.session_state.pop(f'_inline_{current_chat_id}', None)
                            st.markdown(full_response)

                    # 保存 AI 回复 (保持原有逻辑)
                    if full_response and "❌" not in str(full_response):
                        with get_db() as db:
                            bot_msg = Message(chat_id=current_chat_id, role="assistant", content=full_response)
                            db.add(bot_msg)
                            chat_obj = db.get(Chat, current_chat_id)
                            if chat_obj:
                                chat_obj.updated_at = datetime.now(timezone.utc)
                            db.commit()
                            db.refresh(bot_msg)
                            bot_message_id = bot_msg.id
                        _render_response_feedback(bot_message_id)
                    else:
                         with get_db() as db:
                            bot_msg = Message(chat_id=current_chat_id, role="system", content="⚠️ 模型未返回有效内容。")
                            db.add(bot_msg)
                            db.commit()

                except Exception as e:
                    st.error(f"❌ 系统异常：{str(e)}")
                    import traceback
                    with st.expander("查看错误详情"):
                        st.code(traceback.format_exc())
                    
                    with get_db() as db:
                        error_msg = Message(chat_id=current_chat_id, role="system", content=f"⚠️ 系统异常：{str(e)}")
                        db.add(error_msg)
                        db.commit()
        elif user_input:
            st.warning("⚠️ 请输入有效内容")

        # 侧边栏文档面板：响应内嵌引用按钮点击
        _render_sidebar_doc(getattr(agent_graph, 'case_rag', None))
    else:
        # 未选择对话时的欢迎界面
        st.title("欢迎使用纪检监察业务助手")
        st.markdown("""
        ### 功能介绍
        - **业务问答**：识别培训题和案例分析题，优先调用本地配套参考答案。
        - **线索研判**：结合相似违纪违法案例，梳理可能涉及的纪法问题。
        - **党纪法规查询**：检索党内法规、监察法规和监督执纪执法制度。
        - **证据与程序提示**：辅助识别事实缺口、证据方向和办理程序要点。
        - 💬 **智能对话**：支持多轮对话，自动记录上下文。
        - 📁 **对话管理**：在左侧侧边栏创建、切换或删除对话。
        
        👈 **请在左侧侧边栏选择一个对话或创建新对话开始使用！**
        """)
        _render_chat_input_form("chat_input_disabled", disabled=True)

def main():
    """应用入口，包裹全局 try/except 捕获所有未处理异常并写入 crash.log。"""
    try:
        _main()
    except Exception:
        crash_msg = ''.join(traceback.format_exc())
        with open("crash.log", 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"CRASH TIME: {datetime.now(timezone.utc).isoformat()}\n")
            f.write(f"{'='*60}\n")
            f.write(crash_msg)
            f.write(f"{'='*60}\n")
        traceback.print_exc()
        raise

def _main():
    chat_page()

if __name__ == "__main__":
    main()
