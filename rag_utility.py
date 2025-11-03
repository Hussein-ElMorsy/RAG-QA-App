import os
from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA

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
    loader = UnstructuredFileLoader(f"{working_dir}/{file_name}")
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
    return 0;