import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain_community.llms import HuggingFaceHub
from langchain_community.chat_message_histories.upstash_redis import (
    UpstashRedisChatMessageHistory,
)
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)

import os
os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'

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

prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a friendly AI assistant."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

# https://console.upstash.com/
# https://python.langchain.com/docs/integrations/memory/upstash_redis_chat_message_history
URL = "https://usw1-ultimate-glider-33764.upstash.io"
TOKEN ="AYPkASQgM2YwMGIzMWEtZmY4YS00MDU0LTg5ZTAtM2NiN2I4ODgxM2E2YjMwOTRjMzNjYzI2NDRkYzgzZTU4NGI1OWMzYTNhMWU="
history = UpstashRedisChatMessageHistory(
    url=URL, token=TOKEN, ttl=500, session_id="chat1"
)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    chat_memory=history,
)

# chain = prompt | model
chain = LLMChain(
    llm=model,
    prompt=prompt,
    verbose=True,
    memory=memory
)


# Prompt 1
q1 = { "input": "My name is Leon" }
resp1 = chain.invoke(q1)
print(resp1["text"])

# Prompt 2
q2 = { "input": "What is my name?" }
resp2 = chain.invoke(q2)
print(resp2["text"])