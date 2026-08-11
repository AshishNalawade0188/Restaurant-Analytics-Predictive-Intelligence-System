"""
Zomato RAG Chatbot — Refactored CSV-driven Pipeline
===================================================
Refactored to provide natural, humane, and conversational responses:
  - Custom System Prompt: Enforces warm, assistant-style synthesis rather
    than metadata dumping or quoting raw token arrays.
  - Temperature Tuning: Set to 0.4 for a natural conversational balance.
  - Narrative Document Builder: Constructs human-readable narrative sentences
    prior to vector embedding.
"""

import sys
import math
import os
from pathlib import Path
from ast import literal_eval

import pandas as pd
import chromadb
from dotenv import load_dotenv
from llama_index.core import Document, VectorStoreIndex, Settings, StorageContext
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.core.node_parser import SentenceSplitter

# Load environment variables (e.g., GROQ_API_KEY)
load_dotenv()

# ---------------------------------------------------------------------------
# CONFIG — edit these for your setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent

CSV_PATH = r"C:\Users\admin\Desktop\Zomato_ML_Notes\restaurant_reviews_enriched_imputed.csv"
SAMPLE_SIZE = 1000          # Set to None for the full dataset
REBUILD_INDEX = False        # Set to False to reuse existing Chroma storage
CHROMA_PATH = str(SCRIPT_DIR / "chroma_storage")
COLLECTION_NAME = "zomato_rag"

# --- Embedding throughput + chunking config ---
EMBED_MAX_SEQ_TOKENS = 512          
CHUNK_SIZE = 400                    
CHUNK_OVERLAP = 60                  
EMBED_BATCH_SIZE = 128              

# Preprocessed CSV Column Mapping
COLUMN_MAP = {
    "name": "name",
    "location": "location",
    "rest_type": "rest_type",
    "cuisines": "cuisines",
    "dish_liked": "dish_liked",
    "cost_for_two": "approx_cost(for two people)",
    "rate": "rate",
    "votes": "votes",
    "online_order": "online_order",
    "book_table": "book_table",
    "menu_item": "menu_item",
    "reviews_list": "reviews_list",
    "listed_in_type": "listed_in(type)",
    "is_rate_imputed": "is_rate_imputed",
    "clean_review": "clean_review",
    "avg_sentiment_score": "avg_sentiment_score",
    "dominant_sentiment": "dominant_sentiment",
    "positive_ratio": "positive_ratio",
    "keywords": "keywords",
    "dish_keywords": "dish_keywords",
    "review_quality_flag": "review_quality_flag",
    "nlp_source": "nlp_source",
}

NLP_COLUMN_MAP = {
    "positive_review_count": "positive_review_count",
    "negative_review_count": "negative_review_count",
    "neutral_review_count": "neutral_review_count",
    "total_reviews_parsed": "total_reviews_parsed",
    "review_keywords": "review_keywords",
    "keywords_dish_enriched": "keywords_dish_enriched",
    "avg_review_length": "avg_review_length",
}

# ---------------------------------------------------------------------------
# SYSTEM PROMPT DEFINITION
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a friendly, knowledgeable, and objective food and restaurant discovery assistant for Zomato.\n"
    "Your main objective is to provide helpful, natural, and conversational responses based strictly on the retrieved context.\n\n"
    "Core Guidelines:\n"
    "1. Speak naturally like a local dining guide. Never list raw metadata keys or output raw field labels "
    "(such as 'sentiment_score', 'positive_ratio', 'nlp_source', or 'keywords:').\n"
    "2. Handle negative reviews with empathy and nuance. Instead of regurgitating raw token lists (e.g., 'found hair, kill, food thanks'), "
    "summarize core issues naturally (e.g., mentioning specific hygiene concerns or food quality feedback reported by customers).\n"
    "3. Seamlessly weave restaurant statistics (ratings, average costs, location) into flowing sentences.\n"
    "4. If certain details are missing from the context, state it naturally without sounding overly rigid or robotic."
)

# ---------------------------------------------------------------------------
# STEP 0: Model settings
# ---------------------------------------------------------------------------
USE_REAL_MODELS = True

if USE_REAL_MODELS:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.llms.openai_like import OpenAILike

    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Ensure it is defined in your environment or .env file."
        )

    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5",
        embed_batch_size=EMBED_BATCH_SIZE,
    )
    Settings.llm = OpenAILike(
        model="llama-3.1-8b-instant",
        api_base="https://api.groq.com/openai/v1",
        api_key=GROQ_API_KEY,
        is_chat_model=True,
        context_window=131072,
        max_tokens=512,
    )
else:
    from llama_index.core.embeddings import MockEmbedding
    from llama_index.core.llms import MockLLM
    Settings.embed_model = MockEmbedding(embed_dim=384)
    Settings.llm = MockLLM()

Settings.chunk_size = CHUNK_SIZE
Settings.chunk_overlap = CHUNK_OVERLAP
SPLITTER = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

print(f"Embedding model: {type(Settings.embed_model).__name__} (batch_size={EMBED_BATCH_SIZE})")
print(f"LLM: {type(Settings.llm).__name__}")
print(f"Chunking: {CHUNK_SIZE} tokens, {CHUNK_OVERLAP} overlap\n")

# ---------------------------------------------------------------------------
# STEP 1: Load CSV
# ---------------------------------------------------------------------------
def load_dataframe(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    missing = [expected for expected, actual in COLUMN_MAP.items() if actual not in df.columns]
    if missing:
        print(f"WARNING: Expected fields missing from CSV: {missing}")

    if SAMPLE_SIZE and len(df) > SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)
        print(f"Sampled down to {len(df)} rows for this run.\n")

    name_col = COLUMN_MAP["name"]
    before = len(df)
    df = df.dropna(subset=[name_col]).reset_index(drop=True)
    if before != len(df):
        print(f"Dropped {before - len(df)} rows with missing name.\n")

    return df


def col(row: pd.Series, key: str, default="", nlp=False):
    source_map = NLP_COLUMN_MAP if nlp else COLUMN_MAP
    actual_col = source_map.get(key)
    if actual_col is None or actual_col not in row.index:
        return default
    val = row[actual_col]
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return default
    return val


def sanitize_metadata(meta: dict) -> dict:
    clean = {}
    for k, v in meta.items():
        if isinstance(v, float) and math.isnan(v):
            clean[k] = None
        else:
            clean[k] = v
    return clean

# ---------------------------------------------------------------------------
# STEP 2: Parsing utilities
# ---------------------------------------------------------------------------
def parse_reviews(raw_reviews_list) -> str:
    if not raw_reviews_list or not isinstance(raw_reviews_list, str):
        return ""
    try:
        parsed = literal_eval(raw_reviews_list)
        texts = [str(r[1]).strip() for r in parsed if isinstance(r, (tuple, list)) and len(r) >= 2]
        return " ".join(texts)
    except (ValueError, SyntaxError):
        return ""


def parse_menu_items(raw_menu) -> list:
    if not raw_menu or not isinstance(raw_menu, str):
        return []
    try:
        parsed = literal_eval(raw_menu)
        return [str(m).strip() for m in parsed] if isinstance(parsed, list) else []
    except (ValueError, SyntaxError):
        return []

# ---------------------------------------------------------------------------
# STEP 3: Narrative Document builder (Refactored)
# ---------------------------------------------------------------------------
def build_restaurant_document(row: pd.Series) -> Document:
    name = col(row, "name")
    rate_raw = col(row, "rate", default=None)
    rate_is_imputed = bool(col(row, "is_rate_imputed", default=False))

    clean_review = col(row, "clean_review")
    review_text = clean_review if clean_review else parse_reviews(col(row, "reviews_list"))

    menu_items = parse_menu_items(col(row, "menu_item"))
    menu_available = len(menu_items) > 0
    menu_clause = f"Notable menu highlights include {', '.join(menu_items)}." if menu_available else ""

    # Synthesize NLP stats into smooth context sentences
    sentiment_label = col(row, 'dominant_sentiment')
    sentiment_sentence = f"Overall customer feedback is predominantly {sentiment_label}." if sentiment_label else ""

    raw_keywords = col(row, 'keywords')
    keyword_clause = f"Common topics and key phrases in customer feedback include: {raw_keywords}." if raw_keywords else ""

    # Build human-style narrative text chunk for embeddings
    text_content = (
        f"{name} is a {col(row, 'rest_type', 'restaurant')} located in {col(row, 'location', 'an unspecified location')}, "
        f"offering {col(row, 'cuisines', 'a variety of cuisines')}. "
        f"Popular or frequently ordered dishes include: {col(row, 'dish_liked') or 'unspecified dishes'}. "
        f"The approximate cost for two people is ₹{col(row, 'cost_for_two', 'N/A')}. "
        f"The overall user rating stands at {rate_raw or 'N/A'}. "
        f"{sentiment_sentence} {keyword_clause} {menu_clause} "
        f"Detailed customer review text: {review_text or 'No direct review text available.'}"
    )

    metadata = sanitize_metadata({
        "doc_type": "restaurant",
        "name": name,
        "location": col(row, "location", default=None),
        "cuisines": col(row, "cuisines", default=None),
        "rest_type": col(row, "rest_type", default=None),
        "cost_for_two": col(row, "cost_for_two", default=None),
        "rating": rate_raw,
        "rating_is_imputed": rate_is_imputed,
        "online_order": col(row, "online_order", default=None),
        "book_table": col(row, "book_table", default=None),
        "menu_available": menu_available,
        "sentiment_score": col(row, "avg_sentiment_score", default=None),
        "sentiment_label": sentiment_label,
        "positive_ratio": col(row, "positive_ratio", default=None),
        "keywords": raw_keywords,
        "dish_keywords": col(row, "dish_keywords", default=None),
        "review_quality": col(row, "review_quality_flag", default=None),
        "nlp_source": col(row, "nlp_source", default=None),
    })
    return Document(text=text_content, metadata=metadata)


DEFAULT_FAQS = [
    {"question": "How do I cancel my Zomato order?",
     "answer": "Go to Orders, select the active order, and tap Cancel. Refunds are processed within 5-7 business days."},
    {"question": "How does Zomato table booking work?",
     "answer": "Search a restaurant, check slot availability, and confirm. Some restaurants require a refundable deposit."},
]


def build_faq_document(faq: dict) -> Document:
    return Document(text=f"Q: {faq['question']} A: {faq['answer']}",
                     metadata={"doc_type": "faq", "question": faq["question"]})

# ---------------------------------------------------------------------------
# STEP 4: Build the index (idempotent, chunk-aware)
# ---------------------------------------------------------------------------
def build_index() -> VectorStoreIndex:
    from llama_index.vector_stores.chroma import ChromaVectorStore
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

    if not REBUILD_INDEX:
        try:
            chroma_collection = chroma_client.get_collection(COLLECTION_NAME)
            existing_count = chroma_collection.count()
        except Exception:
            existing_count = 0

        if existing_count > 0:
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            index = VectorStoreIndex.from_vector_store(vector_store)
            print(f"Loaded existing index with {existing_count} documents from {CHROMA_PATH}.\n")
            return index
        else:
            print("REBUILD_INDEX is False but no existing index was found — building fresh.\n")

    df = load_dataframe(CSV_PATH)
    restaurant_docs = [build_restaurant_document(row) for _, row in df.iterrows()]
    faq_docs = [build_faq_document(f) for f in DEFAULT_FAQS]
    all_documents = restaurant_docs + faq_docs
    print(f"Built {len(restaurant_docs)} restaurant docs + {len(faq_docs)} FAQ docs = {len(all_documents)} total.\n")

    if REBUILD_INDEX:
        try:
            chroma_client.delete_collection(COLLECTION_NAME)
            print(f"Cleared existing '{COLLECTION_NAME}' collection before rebuild.\n")
        except Exception:
            pass

    chroma_collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        all_documents,
        storage_context=storage_context,
        transformations=[SPLITTER],
        show_progress=True,
    )
    print(f"\nIndexed {chroma_collection.count()} nodes into ChromaDB at {CHROMA_PATH}\n")
    return index

# ---------------------------------------------------------------------------
# STEP 5: Interactive chat loop (Refactored)
# ---------------------------------------------------------------------------
def run_chat(index: VectorStoreIndex):
    # Configure the chat engine with system prompt and temperature settings
    chat_engine = index.as_chat_engine(
        chat_mode="condense_plus_context",
        similarity_top_k=3,
        system_prompt=SYSTEM_PROMPT,
        temperature=0.4,  # Optimal balance of structured accuracy & conversational warmth
    )
    print("Zomato RAG chatbot ready. Type 'exit' to quit, 'sources' to see last retrieval.\n")
    last_response = None
    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if query.lower() in {"exit", "quit"}:
            break
        if query.lower() == "sources" and last_response is not None:
            for n in last_response.source_nodes:
                print(f"  - [{n.metadata.get('doc_type')}] {n.metadata.get('name', n.metadata.get('question'))} (score={n.score:.3f})")
            continue
        last_response = chat_engine.chat(query)
        print(f"Bot: {last_response}\n")

# ---------------------------------------------------------------------------
# STEP 6: Response evaluation (Faithfulness + Relevancy)
# ---------------------------------------------------------------------------
EVAL_QUESTIONS = [
    "Suggest a good North Indian restaurant.",
    "What are some budget-friendly cafes?",
    "Which restaurants have the best reviews for biryani?",
    "How do I cancel my Zomato order?",
    "Are there any restaurants with online ordering in Koramangala?",
]


def run_evaluation(index: VectorStoreIndex, questions=None, out_path="evaluation_report.csv"):
    from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator

    questions = questions or EVAL_QUESTIONS
    query_engine = index.as_query_engine(similarity_top_k=3)
    faithfulness_eval = FaithfulnessEvaluator(llm=Settings.llm)
    relevancy_eval = RelevancyEvaluator(llm=Settings.llm)

    rows = []
    for q in questions:
        response = query_engine.query(q)
        f_result = faithfulness_eval.evaluate_response(query=q, response=response)
        r_result = relevancy_eval.evaluate_response(query=q, response=response)
        rows.append({
            "question": q,
            "answer": str(response),
            "faithfulness_pass": f_result.passing,
            "faithfulness_feedback": f_result.feedback,
            "relevancy_pass": r_result.passing,
            "relevancy_feedback": r_result.feedback,
            "num_source_nodes": len(response.source_nodes),
        })
        print(f"Q: {q}")
        print(f"  Faithfulness: {'PASS' if f_result.passing else 'FAIL'} | Relevancy: {'PASS' if r_result.passing else 'FAIL'}\n")

    report_df = pd.DataFrame(rows)
    report_df.to_csv(out_path, index=False)
    pass_rate_f = report_df["faithfulness_pass"].mean() * 100
    pass_rate_r = report_df["relevancy_pass"].mean() * 100
    print(f"Faithfulness pass rate: {pass_rate_f:.1f}%")
    print(f"Relevancy pass rate: {pass_rate_r:.1f}%")
    print(f"Full report saved to {out_path}")
    return report_df


if __name__ == "__main__":
    idx = build_index()
    if "--evaluate" in sys.argv:
        run_evaluation(idx)
    elif "--no-chat" not in sys.argv:
        run_chat(idx)