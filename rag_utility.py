import os
from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate


_ = load_dotenv(find_dotenv())

working_dir = os.path.dirname(os.path.abspath((__file__)))

# load the embedding model
embedding = HuggingFaceEmbeddings()

# load the llm from groq
llm = ChatGroq(
    model='llama-3.3-70b-versatile',
    temperature=0,
)

def process_document_to_chroma_db(file_name):
    # load the pdfdocument
    loader = PyPDFLoader(f"{working_dir}/{file_name}")
    documents = loader.load()
    # split the text into chunks
    text_spilitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
    )
    texts = text_spilitter.split_documents(documents)
    # store document chunks in the chroma database
    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=embedding,
        persist_directory=f"{working_dir}/doc_vectordb",
    )
    vectordb.persist()
    return True;

def answer_question(user_prompt):
    vectordb = Chroma(
        persist_directory=f"{working_dir}/doc_vectordb",
        embedding_function=embedding,
    )

    retriever = vectordb.as_retriever()

    prompt = ChatPromptTemplate.from_template("""
    Answer the question using only the provided context.

    Context:
    {context}

    Question:
    {input}
    """)

    combine_docs_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt
    )

    retrieval_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=combine_docs_chain
    )

    response = retrieval_chain.invoke({
        "input": user_prompt
    })

    return response["answer"]


