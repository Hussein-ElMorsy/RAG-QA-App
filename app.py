import os
from rag_utility import process_document_to_chroma_db, answer_question
import streamlit as st

# set the working directory
working_dir = os.path.dirname(os.path.abspath((__file__)))

# Configure Streamlit page
st.set_page_config(
    page_title='RAG QA Bot',
    page_icon='📖🤖',
    layout='centered'
)

st.title("🦙 Llama-3.3-70b - Document RAG")

# file uploader widget
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    # define save path
    save_path = os.path.join(working_dir, uploaded_file.name)
    # save the file
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    process_document = process_document_to_chroma_db(uploaded_file.name)
    st.info("Document Processed Successfully")

# text widget to get user prompt
user_prompt = st.text_area("Ask your question about the document")

if st.button("Answer"):
    answer = answer_question(user_prompt)

    st.markdown("### Llama-3.3-70b Response")
    st.markdown(answer)
