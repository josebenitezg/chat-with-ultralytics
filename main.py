from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import streamlit as st
from dotenv import load_dotenv
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain.globals import set_debug

from database import database

load_dotenv()
# set_debug(True)

if "store" not in st.session_state:
    st.session_state.store = {}

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "Hi, I'm a chatbot who can look at the Ultralytics docs for you. How can I help you today?",
        }
    ]


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = InMemoryChatMessageHistory()

    return st.session_state.store[session_id]


st.title("📚 Chat with Ultralytics")


for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])

if user_prompt := st.chat_input(
    placeholder="What do you want to know about Ultralytics?"
):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    st.chat_message("user").write(user_prompt)

    documents = database.similarity_search(user_prompt)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an advanced AI capable of analyzing text
            from documents about Ultralytics and providing detailed answers to
            user queries. Your goal is to offer comprehensive responses to
            eliminate the need for users to revisit the document. If you lack
            the answer, please acknowledge it rather than making up
            information.

            {context}
            """,
            ),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ],
    )

    model = ChatOpenAI(model="gpt-4o", temperature=0)

    chain = prompt | model | StrOutputParser()

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )

    # print("Prompt: ", user_prompt)
    # print("Documents: ", documents[0].page_content.split("\n"))
    # print("------------------------------")
    response = chain.invoke(
        {"context": documents[0].page_content, "question": user_prompt},
        {"configurable": {"session_id": "1"}},
    )

    with st.chat_message("assistant"):
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
