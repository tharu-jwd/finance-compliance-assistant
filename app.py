"""Sri Lankan Financial Regulation RAG Assistant — Streamlit Cloud edition."""

import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from config import GROQ_API_KEY, GROQ_MODEL, MAX_TOKENS, TEMPERATURE
from document_processor import DocumentProcessor

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Sri Lankan Financial Regulation Assistant",
    page_icon="🏦",
    layout="wide"
)

st.title("Sri Lankan Financial Regulation Assistant")
st.caption("Retrieval-Augmented Generation powered by Groq + HuggingFace")

# ---------------------------------------------------------------------------
# Cached resource — one DocumentProcessor for the whole session
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Initialising vector database…")
def get_processor() -> DocumentProcessor:
    return DocumentProcessor()


processor = get_processor()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_upload, tab_chat, tab_eval = st.tabs(
    ["📂 Upload Documents", "💬 Chat", "📊 RAGAS Evaluation"]
)

# ===========================================================================
# TAB 1 — Upload Documents
# ===========================================================================
with tab_upload:
    st.header("Upload Regulatory PDFs")

    uploaded_files = st.file_uploader(
        "Upload regulatory PDFs",
        type="pdf",
        accept_multiple_files=True,
        key="pdf_uploader"
    )

    if uploaded_files:
        if st.button("Process uploaded files"):
            newly_added = 0
            progress = st.progress(0)
            for i, f in enumerate(uploaded_files):
                with st.spinner(f"Processing {f.name}…"):
                    chunks = processor.process_uploaded_file(f)
                    newly_added += chunks
                progress.progress((i + 1) / len(uploaded_files))

            st.success(f"Added {newly_added} chunks from {len(uploaded_files)} file(s).")

    stats = processor.get_collection_stats()
    total = stats.get("total_documents", 0)
    st.metric("Total chunks in ChromaDB", total)

    if total == 0:
        st.warning("No documents in the database yet. Upload PDFs above to get started.")

# ===========================================================================
# TAB 2 — Chat
# ===========================================================================
with tab_chat:
    st.header("Ask a Regulatory Question")

    stats = processor.get_collection_stats()
    if stats.get("total_documents", 0) == 0:
        st.warning("No documents uploaded yet. Go to **Upload Documents** first.")

    # Session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("Sources"):
                    for src in msg["sources"]:
                        meta = src.get("metadata", {})
                        filename = meta.get("source", "Unknown")
                        page = meta.get("page", "?")
                        score = src.get("similarity_score", 0)
                        preview = src["content"][:200]
                        st.markdown(
                            f"**{filename}** — page {page} &nbsp; *(score: {score:.3f})*\n\n"
                            f"> {preview}…"
                        )
                        st.divider()

    # Input
    if query := st.chat_input("Ask about financial regulations…"):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Searching and generating…"):
                sources = processor.search_similar_documents(query)

                context = "\n\n".join(
                    [f"[Source: {s['metadata'].get('source','?')}, "
                     f"page {s['metadata'].get('page','?')}]\n{s['content']}"
                     for s in sources]
                ) if sources else "No relevant documents found."

                llm = ChatGroq(
                    model=GROQ_MODEL,
                    api_key=GROQ_API_KEY,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS
                )

                messages = [
                    SystemMessage(content=(
                        "You are an expert on Sri Lankan financial regulations. "
                        "Answer accurately and concisely using only the provided context. "
                        "If the context does not contain enough information, say so."
                    )),
                    HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}")
                ]

                response = llm.invoke(messages)
                answer = response.content

            st.markdown(answer)

            if sources:
                with st.expander("Sources"):
                    for src in sources:
                        meta = src.get("metadata", {})
                        filename = meta.get("source", "Unknown")
                        page = meta.get("page", "?")
                        score = src.get("similarity_score", 0)
                        preview = src["content"][:200]
                        st.markdown(
                            f"**{filename}** — page {page} &nbsp; *(score: {score:.3f})*\n\n"
                            f"> {preview}…"
                        )
                        st.divider()

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ===========================================================================
# TAB 3 — RAGAS Evaluation
# ===========================================================================
with tab_eval:
    st.header("RAGAS Evaluation")
    st.caption(
        "Runs 5 test questions through the RAG pipeline and scores with "
        "**faithfulness** and **context_precision**."
    )

    TEST_QUESTIONS = [
        "What are the KYC requirements for digital wallets?",
        "What is the minimum capital adequacy ratio for banks?",
        "What documentation is required for cross-border payments?",
        "Are cryptocurrencies legally recognized in Sri Lanka?",
        "What are the AML reporting requirements for payment providers?",
    ]

    st.subheader("Test questions")
    for i, q in enumerate(TEST_QUESTIONS, 1):
        st.write(f"{i}. {q}")

    if st.button("Run Evaluation"):
        stats = processor.get_collection_stats()
        if stats.get("total_documents", 0) == 0:
            st.error("No documents in the database. Upload PDFs first.")
        else:
            with st.spinner("Running RAG pipeline for each question…"):
                eval_llm = ChatGroq(
                    model=GROQ_MODEL,
                    api_key=GROQ_API_KEY,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS
                )

                questions = []
                answers = []
                contexts_list = []

                progress = st.progress(0)
                for i, question in enumerate(TEST_QUESTIONS):
                    sources = processor.search_similar_documents(question)
                    context_texts = [s["content"] for s in sources]

                    context_str = "\n\n".join(context_texts) if context_texts else "No context found."

                    messages = [
                        SystemMessage(content=(
                            "You are an expert on Sri Lankan financial regulations. "
                            "Answer accurately using only the provided context."
                        )),
                        HumanMessage(content=f"Context:\n{context_str}\n\nQuestion: {question}")
                    ]
                    response = eval_llm.invoke(messages)

                    questions.append(question)
                    answers.append(response.content)
                    contexts_list.append(context_texts)

                    progress.progress((i + 1) / len(TEST_QUESTIONS))

            with st.spinner("Computing RAGAS metrics…"):
                try:
                    from datasets import Dataset
                    from ragas import evaluate
                    from ragas.metrics import faithfulness, context_precision
                    from ragas.llms import LangchainLLMWrapper

                    evaluator_llm = LangchainLLMWrapper(
                        ChatGroq(
                            model=GROQ_MODEL,
                            api_key=GROQ_API_KEY,
                            temperature=0
                        )
                    )

                    eval_dataset = Dataset.from_dict({
                        "question": questions,
                        "answer": answers,
                        "contexts": contexts_list,
                    })

                    result = evaluate(
                        eval_dataset,
                        metrics=[faithfulness, context_precision],
                        llm=evaluator_llm
                    )

                    st.success("Evaluation complete!")
                    st.dataframe(result.to_pandas())

                except Exception as e:
                    st.error(f"RAGAS evaluation failed: {e}")
                    st.info("Showing raw answers instead.")
                    import pandas as pd
                    df = pd.DataFrame({
                        "Question": questions,
                        "Answer": answers,
                        "Num contexts retrieved": [len(c) for c in contexts_list]
                    })
                    st.dataframe(df)
