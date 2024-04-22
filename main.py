from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from starlette.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html

from routers import tinyllama, openai

security = HTTPBasic()

# FastAPIドキュメントの認証機能の追加
def get_current_username(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = "LLM"
    correct_password = "@pp"
    if credentials.username != correct_username or credentials.password != correct_password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

app = FastAPI(docs_url=None)

app.include_router(tinyllama.router)
app.include_router(openai.router)

# CORSを回避する
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# FastAPIドキュメントの認証機能の追加
@app.get("/docs", dependencies=[Depends(get_current_username)], include_in_schema=False)
async def get_documentation():
    return get_swagger_ui_html(openapi_url="/openapi.json", title=app.title)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)