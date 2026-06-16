# 2
import os
import sys
import traceback
from datetime import datetime, timezone
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
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

    rating = st.segmented_control(
        "请评价本次回答",
        options=["好", "中", "差"],
        key=rating_key,
        label_visibility="collapsed",
        width="content",
    )

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
@st.cache_resource(show_spinner="🧠 正在加载纪检监察知识库与智能体引擎（仅首次启动需要）...")
def load_global_agent():
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
                "id": "deepseek-chat", 
                "base_url": "https://api.deepseek.com/v1", 
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
        



    # 1. 获取全局 Agent
    agent_graph = load_global_agent()
    

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

        # 用户输入
        user_input = st.chat_input("请输入问题...")
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

                            # 阶段 2：流式完成后，清除占位符，渲染内嵌引用按钮
                            stream_placeholder.empty()
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

                            if full_response and "❌" not in str(full_response):
                                status.update(label="✅ 回答生成完毕", state="complete", expanded=True)
                            elif full_response and "❌" in str(full_response):
                                status.update(label="❌ 生成失败", state="error",expanded=True)

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
        st.chat_input("请输入问题...", disabled=True)

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
