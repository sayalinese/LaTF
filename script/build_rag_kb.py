"""
RAG 知识库构建脚本 (一次性离线运行)
======================================
从 knowledge_base/raw/ 下各平台子目录中读取 PDF 规则文档，
经切块、BGE-m3 Embedding 后写入 PostgreSQL pgvector。

使用方式：
  python script/build_rag_kb.py
  python script/build_rag_kb.py --rebuild   # 清空重建（增量更新时不加此参数）
  python script/build_rag_kb.py --platform 淘宝  # 只处理指定平台

依赖安装：
  pip install langchain langchain-community langchain-postgres langchain-huggingface
  pip install pymupdf sentence-transformers pgvector
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv

# 路径设置
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / '.env')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 平台名称映射：文件夹名 -> 数据库中存储的 platform 标签
PLATFORM_MAP = {
    "淘宝":  "taobao",
    "闲鱼":  "xianyu",
    "京东":  "jd",
    "拼多多": "pdd",
}

# BGE-m3 的输出维度
EMBEDDING_DIM = 1024


def load_platform_docs(raw_dir: Path, target_platform: str = None) -> list:
    """
    遍历 raw/ 下的各平台文件夹，加载所有 PDF 文件。
    每个 chunk 附带元数据：platform, doc_name, source_file。
    """
    from langchain_community.document_loaders import PyMuPDFLoader

    docs = []
    for folder_name, platform_key in PLATFORM_MAP.items():
        if target_platform and folder_name != target_platform:
            continue

        folder_path = raw_dir / folder_name
        if not folder_path.exists():
            logger.warning(f"[{folder_name}] 目录不存在，跳过: {folder_path}")
            continue

        pdf_files = list(folder_path.glob("*.pdf"))
        if not pdf_files:
            logger.warning(f"[{folder_name}] 目录下没有 PDF 文件，跳过")
            continue

        for pdf_file in pdf_files:
            logger.info(f"[{folder_name}] 正在加载: {pdf_file.name}")
            try:
                loader = PyMuPDFLoader(str(pdf_file))
                pages = loader.load()

                # 注入平台元数据，供检索时过滤
                for page in pages:
                    page.metadata["platform"]    = platform_key
                    page.metadata["platform_cn"] = folder_name
                    page.metadata["doc_name"]    = pdf_file.stem
                    page.metadata["source_file"] = str(pdf_file.relative_to(PROJECT_ROOT))

                docs.extend(pages)
                logger.info(f"[{folder_name}] {pdf_file.name}: {len(pages)} 页")
            except Exception as e:
                logger.error(f"[{folder_name}] 加载失败 {pdf_file.name}: {e}")

    return docs


def split_docs(docs: list) -> list:
    """
    使用中文友好的分隔符进行递归切分。
    chunk_size=600 适合条款粒度；overlap=80 防止单条规则被切断。
    """
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=80,
        separators=["第", "条", "款", "\n\n", "\n", "；", "。", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(docs)
    logger.info(f"切分完成：{len(docs)} 页 → {len(chunks)} 个 chunk")
    return chunks


def build_embeddings(device: str = "cuda"):
    """
    初始化 BGE-m3 Embedding 模型（支持中英文混合检索）。
    """
    from langchain_huggingface import HuggingFaceEmbeddings

    logger.info(f"加载 BGE-m3 Embedding 模型 (device={device})...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 16},
    )
    logger.info("Embedding 模型加载完成")
    return embeddings


def write_to_pgvector(chunks: list, embeddings, connection_string: str, rebuild: bool):
    """
    将 chunks 写入 PostgreSQL pgvector。
    rebuild=True 时先清空集合再重建（全量更新）。
    rebuild=False 时追加写入（增量更新，适合只增加了新文件时）。
    """
    from langchain_postgres.vectorstores import PGVector

    logger.info(f"连接 PostgreSQL，collection: ecommerce_rules (rebuild={rebuild})")

    vectorstore = PGVector.from_documents(
        documents=chunks,
        embedding=embeddings,
        connection=connection_string,
        collection_name="ecommerce_rules",
        pre_delete_collection=rebuild,  # True=清空重建，False=追加
    )
    logger.info(f"✅ 写入完成！共 {len(chunks)} 个 chunk 已存入 pgvector。")
    return vectorstore


def verify_build(connection_string: str, embeddings):
    """
    简单验证：用一个测试问题查询，看能不能检索到结果。
    """
    from langchain_postgres.vectorstores import PGVector

    logger.info("验证向量库：运行测试检索...")
    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name="ecommerce_rules",
        connection=connection_string,
    )
    test_query = "买家发现商品图片是AI生成的虚假图片，如何申请退款？"
    results = vectorstore.similarity_search(test_query, k=3)

    if results:
        logger.info(f"✅ 验证通过：找到 {len(results)} 条相关条款")
        for i, doc in enumerate(results, 1):
            platform = doc.metadata.get("platform_cn", "未知平台")
            doc_name = doc.metadata.get("doc_name", "未知文档")
            logger.info(f"  [{i}] ({platform}) 《{doc_name}》: {doc.page_content[:80]}...")
    else:
        logger.warning("⚠️ 验证：未检索到任何结果，请检查 PDF 文件是否已放入 knowledge_base/raw/")


def main():
    parser = argparse.ArgumentParser(description="构建电商条款 RAG 向量知识库")
    parser.add_argument("--rebuild",   action="store_true", help="清空并重建整个知识库（默认：追加模式）")
    parser.add_argument("--platform",  type=str, default=None, help="只处理指定平台，如 '淘宝'、'闲鱼'")
    parser.add_argument("--device",    type=str, default="cuda", help="Embedding 运行设备 (cuda/cpu)")
    parser.add_argument("--verify",    action="store_true", default=True, help="构建后运行验证检索")
    args = parser.parse_args()

    connection_string = os.getenv("DATABASE_URL")
    if not connection_string:
        logger.error("未找到 DATABASE_URL 环境变量，请检查 .env 文件")
        sys.exit(1)

    raw_dir = PROJECT_ROOT / "knowledge_base" / "raw"
    logger.info(f"原始文档目录: {raw_dir}")

    # Step 1: 加载 PDF
    docs = load_platform_docs(raw_dir, target_platform=args.platform)
    if not docs:
        logger.error("没有加载到任何文档，请把 PDF 文件放入 knowledge_base/raw/<平台名>/ 目录下")
        sys.exit(1)
    logger.info(f"共加载 {len(docs)} 页文档")

    # Step 2: 切分
    chunks = split_docs(docs)

    # Step 3: 初始化 Embedding
    embeddings = build_embeddings(device=args.device)

    # Step 4: 写入 pgvector
    write_to_pgvector(chunks, embeddings, connection_string, rebuild=args.rebuild)

    # Step 5: 验证
    if args.verify:
        verify_build(connection_string, embeddings)

    logger.info("="*50)
    logger.info("知识库构建完成！")
    logger.info("下次运行 Flask 服务时，RAG 检索器会自动连接此知识库。")


if __name__ == "__main__":
    main()
