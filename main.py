from fastapi import FastAPI
from langserve import add_routes
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import ChatPromptTemplate

from huggingface_hub import snapshot_download

import torch
from typing import Any, Dict, TypedDict

from dotenv import load_dotenv
from langchain_core.runnables import chain
from typing import Dict, Any, Optional, List, Union
from langchain_core.runnables import RunnableConfig

from googletrans import LANGUAGES, Translator

# huggingfaceのAPI KEYのため読み込む
load_dotenv()

translator = Translator()

# Appを設定
app = FastAPI(
    title="LocalLLM Server",
    version="1.0",
    description="LocalLLM Server",
)

# CORSの設定を行う
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
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
    app,
    simplechat_chain,
    path="/tinyllama/simplechat",
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

add_routes(app, pdfragchain_, path="/tinyllama/pdfrag")

#===============================================================================
# agent機能を持つチャットボット（python + llm-math + web）

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)