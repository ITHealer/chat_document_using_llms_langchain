import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import HuggingFaceHub

import os
os.environ['HUGGINGFACEHUB_API_TOKEN'] = ''

def chat_llms(question):
    # Instantiate Model
    llm = HuggingFaceHub(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        task="text-generation",
        model_kwargs={
            "max_new_tokens": 512,
            "top_k": 30,
            "temperature": 0.4,
            "repetition_penalty": 1.03,
        },
    )

    # Prompt Template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Generate a list of 10 synonyms for the following word. Return the results as a comma seperated list."),
            ("human", "{input}")
        ]
    )

    # Create LLM Chain
    chain = prompt | llm
    response = chain.invoke({"input": question})
    # print(type(response.content))
    return response

# UI
st.title("Chat with LLMs")
question = st.text_input("Question")

# Button to process input
if st.button('Submit'):
    with st.spinner('Processing...'):
        answer = chat_llms(question)
        st.text_area("Answer", value=answer, height=300, disabled=True) 
