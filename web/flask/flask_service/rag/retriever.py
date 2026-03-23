"""
EcommerceRulesRetriever: 连接 pgvector，对电商平台规则知识库进行相似度检索。
"""
from __future__ import annotations

import os
from typing import List, Optional

# 延迟导入，避免无 RAG 依赖时 app 启动失败
_retriever_instance: Optional["EcommerceRulesRetriever"] = None

PLATFORM_KEY_MAP = {
    "淘宝": "taobao",
    "闲鱼": "xianyu",
    "京东": "jd",
    "拼多多": "pdd",
    "taobao": "taobao",
    "xianyu": "xianyu",
    "jd": "jd",
    "pdd": "pdd",
}


class EcommerceRulesRetriever:
    """Singleton——应用生命周期内只加载一次 embedding 模型和 pgvector 连接。"""

    def __init__(self, connection_string: str, collection_name: str = "ecommerce_rules"):
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_postgres.vectorstores import PGVector

        device = "cuda" if _cuda_available() else "cpu"
        self._embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )
        self._store = PGVector(
            embeddings=self._embeddings,
            collection_name=collection_name,
            connection=connection_string,
        )

    def retrieve(
        self,
        query: str,
        platform: Optional[str] = None,
        k: int = 3,
    ) -> List[dict]:
        """
        检索与 query 最相关的规则条款。

        Args:
            query:    用户问题 / 对话文本
            platform: 可选平台名称（中文或英文 key），传入时只检索该平台文档
            k:        返回条数

        Returns:
            List of {"content": str, "platform": str, "doc_name": str, "score": float}
        """
        filter_dict: Optional[dict] = None
        if platform:
            platform_key = PLATFORM_KEY_MAP.get(platform)
            if platform_key:
                filter_dict = {"platform": platform_key}

        try:
            if filter_dict:
                results = self._store.similarity_search_with_relevance_scores(
                    query, k=k, filter=filter_dict
                )
            else:
                results = self._store.similarity_search_with_relevance_scores(query, k=k)
        except Exception:
            return []

        docs = []
        for doc, score in results:
            docs.append({
                "content": doc.page_content,
                "platform": doc.metadata.get("platform_cn", doc.metadata.get("platform", "")),
                "doc_name": doc.metadata.get("doc_name", ""),
                "score": round(float(score), 4),
            })
        return docs


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def get_retriever() -> Optional[EcommerceRulesRetriever]:
    """获取全局单例，若 RAG 依赖未安装则返回 None。"""
    global _retriever_instance
    if _retriever_instance is not None:
        return _retriever_instance
    try:
        conn = os.environ.get(
            "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/laft"
        )
        _retriever_instance = EcommerceRulesRetriever(connection_string=conn)
        return _retriever_instance
    except Exception as exc:
        print(f"[RAG] 初始化检索器失败，RAG 功能不可用: {exc}")
        return None
