"""
MODULE 4 — AI Career Mentor (RAG), orchestrated with LangChain.

Pipeline: guardrail check -> retrieve top-K chunks from career_notes (+ job
corpus) -> build grounded prompt -> LLM answer -> (optional) refuse if
retrieval was empty/irrelevant, per the hallucination-check requirement in
Section 8.
"""
from pathlib import Path
from typing import List

from langchain_community.vectorstores import FAISS as LangchainFAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.config import (
    CAREER_NOTES_DIR,
    NOTES_INDEX_PATH,
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL,
    RAG_TOP_K,
    EMBEDDING_MODEL_NAME,
)
from src.generate.prompts import MENTOR_SYSTEM_PROMPT
from src.safety.guardrails import check_message, GuardrailViolation

# Minimum similarity score to trust a retrieved chunk. Below this we treat
# retrieval as "nothing relevant found" -> mentor should say "I don't know".
RELEVANCE_THRESHOLD = 0.35


def _get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def build_notes_index(notes_dir: Path = CAREER_NOTES_DIR, out_path: Path = NOTES_INDEX_PATH) -> None:
    """Load every .md/.txt career note, chunk it, embed it, and persist a LangChain FAISS store."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
    docs: List[Document] = []

    for path in sorted(Path(notes_dir).glob("*")):
        if path.suffix.lower() not in (".md", ".txt"):
            continue
        text = path.read_text(encoding="utf-8")
        for chunk in splitter.split_text(text):
            docs.append(Document(page_content=chunk, metadata={"source": path.name}))

    if not docs:
        raise ValueError(f"No .md/.txt career notes found in {notes_dir}")

    store = LangchainFAISS.from_documents(docs, _get_embeddings())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    store.save_local(str(out_path))
    print(f"Indexed {len(docs)} career-note chunks -> {out_path}")


def load_notes_index(path: Path = NOTES_INDEX_PATH) -> LangchainFAISS:
    return LangchainFAISS.load_local(
        str(path), _get_embeddings(), allow_dangerous_deserialization=True
    )


def ask_mentor(question: str, chat_history: List[dict] | None = None) -> dict:
    """
    Full RAG turn:
      1. guardrail check on the raw question (blocks unsafe/off-topic before any retrieval/LLM spend)
      2. retrieve top-K career-note chunks
      3. if nothing relevant retrieved -> refuse rather than hallucinate
      4. otherwise, ground the LLM answer in the retrieved context

    Returns {"answer": str, "sources": [str], "blocked": bool}
    """
    try:
        check_message(question)
    except GuardrailViolation as e:
        return {"answer": str(e), "sources": [], "blocked": True}

    store = load_notes_index()
    results = store.similarity_search_with_relevance_scores(question, k=RAG_TOP_K)
    relevant = [(doc, score) for doc, score in results if score >= RELEVANCE_THRESHOLD]

    if not relevant:
        return {
            "answer": "I don't have enough information in my career notes to answer that "
                      "confidently. Try rephrasing, or ask about a topic covered in the career "
                      "notes (role guides, skill roadmaps, job search strategy).",
            "sources": [],
            "blocked": False,
        }

    context = "\n\n---\n\n".join(doc.page_content for doc, _ in relevant)
    sources = sorted({doc.metadata.get("source", "unknown") for doc, _ in relevant})

    llm = ChatAnthropic(model=ANTHROPIC_MODEL, api_key=ANTHROPIC_API_KEY, max_tokens=800)

    messages = [("system", MENTOR_SYSTEM_PROMPT.format(context=context))]
    for turn in (chat_history or []):
        messages.append((turn["role"], turn["content"]))
    messages.append(("human", question))

    response = llm.invoke(messages)
    return {"answer": response.content, "sources": sources, "blocked": False}
