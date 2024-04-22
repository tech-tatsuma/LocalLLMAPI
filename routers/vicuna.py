from fastapi import APIRouter
from langserve import add_routes

from langchain_community.llms import LlamaCpp
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler

router = APIRouter(
    prefix='/vicuna',
    tags=['vicuna'],
)

PATH_MODEL = "models/llama-cpp-1.1B-Chat-v1.0"
# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
llm = LlamaCpp(
    model_path="/Users/rlm/Desktop/Code/llama.cpp/models/openorca-platypus2-13b.gguf.q4_0.bin",
    temperature=0.75,
    max_tokens=2000,
    top_p=1,
    callback_manager=callback_manager,
    verbose=True,  # Verbose is required to pass to the callback manager)
)

TEMPLATE = ChatPromptTemplate.from_template(
    "USER: {user_input} ASSISTANT:"
)

output_parser = StrOutputParser()
simplechain = (
    {"user_input": RunnablePassthrough()}
    | TEMPLATE
    | llm
    | output_parser
 )

add_routes(router, simplechain, path="/simplechat")