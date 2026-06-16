"""独立脚本：构建案例库向量索引和BM25索引，避免通过Streamlit构建时的重复执行问题"""
import os
from dotenv import load_dotenv
load_dotenv()
import faiss


from model_utils import LegalRAGapi

case_db_path = os.getenv("VECTOR_CASE_DB_PATH", "law_faiss_case_pattern_pid")
knowledge_folder = os.path.join(
    os.getcwd(), 'knowledge_base_case', os.getenv('CASE_KB_SUBDIR', 'candidate_55192')
)

if os.path.exists(case_db_path):
    print(f"案例向量数据库已存在：{case_db_path}，跳过构建。如需重建请先删除该目录。")
else:
    print(f"开始构建案例库索引，数据目录：{knowledge_folder}")
    rag = LegalRAGapi(db_path=case_db_path, vector_weight=0.6, bm25_weight=0.4)
    rag.add_folder_documents(knowledge_folder)
    print("构建完成。")
