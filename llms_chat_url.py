import streamlit as st

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains import create_retrieval_chain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub

import os
os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'

def get_documents_from_web(url):
    loader = WebBaseLoader(url)
    docs = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )
    splitDocs = splitter.split_documents(docs)
    return splitDocs

def create_db(docs):
    model_id = 'sentence-transformers/all-MiniLM-L6-v2'
    embedding = HuggingFaceEmbeddings(
        model_name=model_id
    )
    vectorStore = FAISS.from_documents(docs, embedding=embedding)
    return vectorStore

def create_chain(vectorStore):
    model = HuggingFaceHub(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        task="text-generation",
        model_kwargs={
            "max_new_tokens": 512,
            "top_k": 30,
            "temperature": 0.4,
            "repetition_penalty": 1.03,
        },
    )

    prompt = ChatPromptTemplate.from_template("""
    Answer the user's question:
    Context: {context}
    Question: {input}                                       
    """)

    # chain = prompt | model
    chain = create_stuff_documents_chain(
        llm=model,
        prompt=prompt
    )

    retriever = vectorStore.as_retriever(search_kwargs={"k": 1})

    retrieval_chain = create_retrieval_chain(
        retriever,
        chain
    )

    return retrieval_chain

# UI

st.title("Document Query with Ollama")
st.write("Enter URLs (one per line) and a question to query the documents.")

# Input fields
urls = st.text_area("Enter URLs separated by new lines", height=150)
question = st.text_input("Question")

# Button to process input
if st.button('Query Documents'):
    with st.spinner('Processing...'):
        
        docs = get_documents_from_web(urls) # ('https://python.langchain.com/docs/expression_language/')
        vectorStore = create_db(docs)
        chain = create_chain(vectorStore)

        answer = chain.invoke({
            "input": question,
        })
        st.text_area("Answer", value=answer["context"], height=300, disabled=True) 

        