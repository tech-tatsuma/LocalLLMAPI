from fastapi import APIRouter
from langserve import add_routes
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import chain

from typing import Any, Dict, TypedDict

from dotenv import load_dotenv

load_dotenv()

from langserve import add_routes

router = APIRouter(
    prefix='/openai',
    tags=['openai']
)

#===============================================================================
# simplechatの実装
template = """
Question: {question}

Answer: step by stepに考え、日本語で回答してください。
"""
prompt = PromptTemplate.from_template(template)

llm = OpenAI(model_name="gpt-3.5-turbo-instruct")

simple_chain = (prompt | llm)

add_routes(
    router,
    simple_chain,
    path="/simplechat"
)

#===============================================================================
# メモリーを持ったchatの実装
# ChatPromptTemplateを作成
hisprompt = ChatPromptTemplate.from_messages(
    [
        ("system", "あなたは日本語で回答するアシスタントです。"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

hischainroot_ = (hisprompt | llm)
# Redisを用いたチェーンの作成
chainwithhistory = RunnableWithMessageHistory(
    hischainroot_,
    lambda session_id: RedisChatMessageHistory(
        session_id, url="redis://localhost:6379"
    ),
    input_messages_key="question",
    history_messages_key="history",
)

# 入力の型を定義
class HisInput(TypedDict):
    session_id: str
    question: str

# 出力の型を定義
class HisOutput(TypedDict):
    response: str

@chain
def hischain(input: HisInput) -> Dict[str, Any]:
    config = {"configurable": {"session_id": input["session_id"]}}
    return chainwithhistory.invoke({"question":input["question"]}, config=config)

# チェーンの作成
hischain_ = (
    hischain.with_types(
        input_type=HisInput,
        output_type=HisOutput
    )
)

add_routes(router, hischain_, path="/his")