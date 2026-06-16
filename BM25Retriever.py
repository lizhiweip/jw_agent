import os
import math
import pickle
from typing import List, Tuple, Optional
from collections import Counter

import numpy as np
import jieba


class NumpyBM25:
    """numpy 向量化 BM25，替代 rank_bm25.BM25Okapi"""

    def __init__(self, tokenized_docs: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(tokenized_docs)

        # 文档长度
        doc_lens = np.array([len(d) for d in tokenized_docs], dtype=np.float32)
        self.avgdl = doc_lens.mean() if self.corpus_size > 0 else 1.0
        # 预计算长度归一化因子: 1 - b + b * dl / avgdl
        self.len_norm = 1.0 - b + b * doc_lens / self.avgdl  # shape: (N,)

        # 构建词汇表和倒排索引
        self.vocab = {}  # word -> index
        df = Counter()  # document frequency
        self.tf_rows = []  # CSR-like: list of (doc_indices, tf_values) per vocab word

        # 第一遍：收集词汇和 DF
        doc_tfs = []
        for doc in tokenized_docs:
            tf = Counter(doc)
            doc_tfs.append(tf)
            for w in tf:
                if w not in self.vocab:
                    self.vocab[w] = len(self.vocab)
                df[w] += 1

        # 预计算 IDF: log((N - df + 0.5) / (df + 0.5) + 1)
        vocab_size = len(self.vocab)
        self.idf = np.zeros(vocab_size, dtype=np.float32)
        for w, idx in self.vocab.items():
            n = df[w]
            self.idf[idx] = math.log((self.corpus_size - n + 0.5) / (n + 0.5) + 1.0)

        # 第二遍：构建稀疏 TF 存储 (per-word inverted lists with TF)
        # 用 list of arrays 而非 scipy sparse，避免额外依赖
        inv_docs = [[] for _ in range(vocab_size)]
        inv_tfs = [[] for _ in range(vocab_size)]
        for doc_id, tf in enumerate(doc_tfs):
            for w, count in tf.items():
                idx = self.vocab[w]
                inv_docs[idx].append(doc_id)
                inv_tfs[idx].append(count)

        self.inv_doc_ids = [np.array(d, dtype=np.int32) for d in inv_docs]
        self.inv_tf = [np.array(t, dtype=np.float32) for t in inv_tfs]

    def get_scores(self, tokenized_query: List[str]) -> np.ndarray:
        scores = np.zeros(self.corpus_size, dtype=np.float32)
        for token in tokenized_query:
            idx = self.vocab.get(token)
            if idx is None:
                continue
            doc_ids = self.inv_doc_ids[idx]
            tf = self.inv_tf[idx]
            # BM25 score per doc: idf * (tf * (k1+1)) / (tf + k1 * len_norm)
            norm = tf + self.k1 * self.len_norm[doc_ids]
            scores[doc_ids] += self.idf[idx] * (tf * (self.k1 + 1.0)) / norm
        return scores


class BM25Retriever:
    """BM25检索器（numpy向量化版，支持批量添加，支持PID追踪）"""

    def __init__(self, bm25_index_path: str = "bm25_index.pkl", rebuild_threshold: int = 50):
        self.bm25_index_path = bm25_index_path
        self.bm25 = None
        self.documents = []
        self.tokenized_docs = []
        self.pids = []
        self.pending_documents = []
        self.pending_pids = []
        self.rebuild_threshold = rebuild_threshold

    def chinese_tokenize(self, text: str) -> List[str]:
        return list(jieba.cut(text))

    def build_index(self, documents: List[str], pids: List[Optional[str]] = None):
        if not documents:
            return

        self.documents = documents
        self.pids = pids if pids else [None] * len(documents)
        print(f"正在构建BM25索引，文档数量: {len(documents)}")

        self.tokenized_docs = [self.chinese_tokenize(doc) for doc in documents]
        self.bm25 = NumpyBM25(self.tokenized_docs)
        print("BM25索引构建完成")

    def add_documents(self, new_documents: List[str], pids: List[Optional[str]] = None, force_rebuild: bool = False):
        if not new_documents:
            return

        if pids is None:
            pids = [None] * len(new_documents)

        self.pending_documents.extend(new_documents)
        self.pending_pids.extend(pids)
        print(f"已缓存 {len(new_documents)} 个文档，待处理文档总数: {len(self.pending_documents)}")

        should_rebuild = (force_rebuild or
                          len(self.pending_documents) >= self.rebuild_threshold or
                          self.bm25 is None)

        if should_rebuild:
            self._rebuild_with_pending()

    def _rebuild_with_pending(self):
        if not self.pending_documents and self.bm25 is not None:
            return

        all_documents = self.documents + self.pending_documents
        all_pids = self.pids + self.pending_pids

        if not all_documents:
            return

        print(f"正在重建BM25索引，总文档数: {len(all_documents)}")
        self.build_index(all_documents, pids=all_pids)

        pending_count = len(self.pending_documents)
        self.pending_documents = []
        self.pending_pids = []
        print(f"索引重建完成，新增 {pending_count} 个文档")

    def force_rebuild(self):
        self._rebuild_with_pending()

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float, Optional[str]]]:
        if self.pending_documents:
            print("检测到有待处理文档，正在重建索引...")
            self._rebuild_with_pending()

        if self.bm25 is None or not self.documents:
            return []

        tokenized_query = self.chinese_tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        k = min(top_k, len(scores))
        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        return [(self.documents[i], float(scores[i]), self.pids[i]) for i in top_indices]

    def save_index(self):
        if self.pending_documents:
            self._rebuild_with_pending()

        if self.bm25 is not None:
            with open(self.bm25_index_path, 'wb') as f:
                pickle.dump({
                    'bm25': self.bm25,
                    'documents': self.documents,
                    'tokenized_docs': self.tokenized_docs,
                    'pids': self.pids
                }, f)
            print(f"BM25索引已保存到: {self.bm25_index_path}")

    def load_index(self) -> bool:
        if not os.path.exists(self.bm25_index_path):
            return False

        try:
            with open(self.bm25_index_path, 'rb') as f:
                data = pickle.load(f)
                self.documents = data['documents']
                self.tokenized_docs = data['tokenized_docs']
                self.pids = data.get('pids', [None] * len(self.documents))
                self.pending_documents = []
                self.pending_pids = []

                # 兼容旧索引：如果存的是 rank_bm25 的 BM25Okapi，重建为 NumpyBM25
                old_bm25 = data.get('bm25')
                if isinstance(old_bm25, NumpyBM25):
                    self.bm25 = old_bm25
                else:
                    print("检测到旧版BM25索引，正在转换为numpy向量化版本...")
                    self.bm25 = NumpyBM25(self.tokenized_docs)
                    # 自动保存转换后的索引
                    self.save_index()
                    print("旧索引已自动转换并保存")

            print(f"BM25索引已从 {self.bm25_index_path} 加载，文档数: {len(self.documents)}")
            return True
        except Exception as e:
            print(f"加载BM25索引失败: {e}")
            return False

    def get_document_count(self) -> int:
        return len(self.documents) + len(self.pending_documents)

    def get_indexed_count(self) -> int:
        return len(self.documents)

    def get_pending_count(self) -> int:
        return len(self.pending_documents)

    def clear_index(self):
        self.bm25 = None
        self.documents = []
        self.tokenized_docs = []
        self.pids = []
        self.pending_documents = []
        self.pending_pids = []
        print("BM25索引已清空")
