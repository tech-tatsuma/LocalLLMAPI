from fastapi import FastAPI
from langserve import add_routes
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import ChatPromptTemplate

from huggingface_hub import snapshot_download

import torch
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_core.runnables import chain
from typing import Dict, Any, Optional, List, Union
from langchain_core.runnables import RunnableConfig

# huggingfaceのAPI KEYのため読み込む
load_dotenv()

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

#===============================================================================
# TinyLlama
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

# 4.ChatPromptTemplateを作成
prompt = ChatPromptTemplate.from_template(
    "<|system|>You are a intelligent chatbot. </s><|user|>{question}</s><|assistant|>"
)

# Chainを設定
chain = (
        prompt | llm 
)

add_routes(
    app,
    chain,
    path="/tinyllama",
)


#===============================================================================
# codellama
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model_id2 = "elyza/ELYZA-japanese-Llama-2-7b-instruct"
# download_path2 = snapshot_download(repo_id=model_id2)

# tokenizer2 = AutoTokenizer.from_pretrained(download_path2)
# model2 = AutoModelForCausalLM.from_pretrained(download_path2).to(device)

# pipe2 = pipeline("text-generation", model=model2, tokenizer=tokenizer2, framework='pt', max_new_tokens=1024)
# llm2 = HuggingFacePipeline(pipeline=pipe2)

# prompt2 = ChatPromptTemplate.from_template(
#     "<|system|>あなたは日本語対応を行うチャットボットです．</s><|user|>{question}</s><|assistant|>"
# )

# chain2 = (
#         prompt2 | llm2 
# )   

# add_routes(
#     app,
#     chain2,
#     path="/elyza",
# )

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)