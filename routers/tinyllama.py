from fastapi import APIRouter
from langserve import add_routes
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

import logging

from huggingface_hub import snapshot_download

import torch
from typing import Any, Dict, TypedDict
import re

from dotenv import load_dotenv
from langchain_core.runnables import chain
from typing import Dict, Any, Optional, List, Union
from langchain_core.runnables import RunnableConfig

from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

logger = logging.getLogger('uvicorn')  

# huggingfaceのAPI KEYのため読み込む
load_dotenv()

# Appを設定
router = APIRouter(
    prefix='/tinyllama',
    tags=['tinyllama'],
)

# 1.HuggingFaceからモデルをダウンロード
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
download_path = snapshot_download(repo_id=model_id)

# 2.TokenizerとModelをスナップショットからインスタンス化
tokenizer = AutoTokenizer.from_pretrained(download_path)
model = AutoModelForCausalLM.from_pretrained(download_path)

# 3.HuggingFacePipelineを作成
# ref. https://python.langchain.com/docs/integrations/llms/huggingface_pipelines
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256)
llm = HuggingFacePipeline(pipeline=pipe)

#===============================================================================
# シンプルなチャットボット
# ChatPromptTemplateを作成
prompt = ChatPromptTemplate.from_template(
    "<|system|>You are a intelligent chatbot. </s><|user|>{question}</s><|assistant|>"
)

# Chainを設定
simplechat_chain = (
        prompt | llm 
)

# ルートを追加
add_routes(
    router,
    simplechat_chain,
    path="/simplechat",
)

#===============================================================================
# PDFファイルを読み込むRAGの機能を持つチャットボット
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")

# 入力の型を定義
class PDFRAGInput(TypedDict):
    pdf: str
    question: str

# 出力の型を定義
class PDFRAGOutput(TypedDict):
    output: str

@chain
def pdf_rag(input: PDFRAGInput) -> Dict[str, Any]:
    # PDFファイルの読み込み
    loader = PyPDFLoader(input["pdf"])
    documents = loader.load()

    # 文書を分割
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=40)
    texts = text_splitter.split_documents(documents)

    # embeddingsを計算
    vectorstore = FAISS.from_documents(
        documents=texts, embedding=embeddings
    )

    retreiver = vectorstore.as_retriever()

    # Chainの作成
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retreiver)

    # Chainの実行
    return {"response": qa.invoke(input["question"])}

# チェーンの作成
pdfragchain_ = (
    pdf_rag.with_types(
        input_type=PDFRAGInput,
        output_type=PDFRAGOutput
    )
)

add_routes(router, pdfragchain_, path="/pdfrag")

#===============================================================================
# チャットメッセージの履歴を保存するチャットボット

def extract_first_response(text: str) -> str:
    # 正規表現を用いて、"\n~:" 形式のテキストブロックを見つける
    match = re.search(r'\n(.*?):', text)
    if match:
        speaker = match.group(1)  # 話者の部分を抽出（Assistant, Human, etc.）
        # 最初の回答を抽出
        start_pos = match.start()  # マッチした部分の開始位置
        end_pos = text.find('\n', start_pos + 1)  # 次の "\n" の位置を見つける
        if end_pos == -1:  # 次の "\n" がない場合は、テキストの末尾までを取得
            end_pos = len(text)
        return text[start_pos+1:end_pos]  # 最初の回答を返す
    else:
        return text
    
# ChatPromptTemplateを作成
hisprompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a intelligent chatbot. "),
        MessagesPlaceholder(variable_name="second"),
        ("user", "{question}"),
    ]
)
logger.info(hisprompt)
# チャットボットのチェーンを作成
hischainroot_ = (hisprompt | llm)
# Redisを用いたチェーンの作成
chainwithhistory = RunnableWithMessageHistory(
    hischainroot_,
    lambda session_id: RedisChatMessageHistory(
        session_id, url="redis://localhost:6379"
    ),
    input_messages_key="question",
    history_messages_key="second",
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
    res = extract_first_response(chainwithhistory.invoke({"question":input["question"]}, config=config))
    return res

# チェーンの作成
hischain_ = (
    hischain.with_types(
        input_type=HisInput,
        output_type=HisOutput
    )
)

add_routes(router, hischain_, path="/his")