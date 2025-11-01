# File: rag_chain.py
# Descrizione: Gestione dell'ingestione di PDF e interrogazione tramite RAG con LangChain e Ollama

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

import os
import shutil

PDF_DIR = 'data/pdfs'
VECTOR_DIR = 'data/vectors'
os.makedirs(VECTOR_DIR, exist_ok=True)

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# Funzione per caricare il vectorstore da file se esiste
def load_vectorstore():
    faiss_index = os.path.join(VECTOR_DIR, 'index.faiss')
    if os.path.exists(faiss_index):
        print("[INFO] Caricamento vectorstore FAISS esistente...")
        return FAISS.load_local(VECTOR_DIR, embeddings=embedding, allow_dangerous_deserialization=True)
    else:
        print("[INFO] Nessun vectorstore FAISS trovato.")
        return None

vectorstore = load_vectorstore()
last_used_model = None

def ingest_pdfs(pdf_paths):
    global vectorstore
    all_chunks = []
    for pdf_path in pdf_paths:
        print(f"[INFO] Caricamento PDF: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        print(f"[INFO] Trovate {len(documents)} pagine in {pdf_path}")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(documents)
        print(f"[INFO] Generati {len(chunks)} frammenti da indicizzare.")
        all_chunks.extend(chunks)

    if vectorstore is None:
        vectorstore = FAISS.from_documents(all_chunks, embedding)
    else:
        vectorstore.add_documents(all_chunks)

    vectorstore.save_local(VECTOR_DIR)
    return len(all_chunks), all_chunks  # Restituisce anche i chunk per l'esplorazione


def ask_question(question, model_name='mistral', system_message=None):
    global vectorstore, last_used_model

    if vectorstore is None or model_name != last_used_model:
        vectorstore = load_vectorstore()
        last_used_model = model_name
        if vectorstore is None:
            return "[ERRORE] Nessun documento indicizzato. Caricare un PDF."

    from langchain_community.llms import Ollama
    from langchain_core.prompts import ChatPromptTemplate
    from langchain.chains import RetrievalQA

    llm = Ollama(model=model_name, temperature=0.3)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    system_message = system_message or (
        "Rispondi esclusivamente utilizzando le informazioni fornite nel contesto. "
        "Se la risposta non è presente nei documenti, rispondi: 'Non sono in grado di rispondere con le informazioni disponibili.' "
        "Se la domanda è collegata in modo indiretto al contesto, prova a rispondere usando inferenze dai contenuti presenti. "
        "Non usare conoscenze esterne. Rispondi in italiano."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "{context}\n\n" + system_message),
        ("human", "{question}")
    ])

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={
            "prompt": prompt,
            "document_variable_name": "context"
        }
    )

    return qa_chain.run(question)

def get_indexed_chunks():
    if vectorstore is None:
        return []
    return vectorstore.similarity_search("", k=100)

def clear_vectorstore():
    global vectorstore
    if os.path.exists(VECTOR_DIR):
        shutil.rmtree(VECTOR_DIR)
    os.makedirs(VECTOR_DIR, exist_ok=True)
    vectorstore = None
