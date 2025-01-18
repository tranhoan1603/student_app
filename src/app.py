import os
os.environ['TOKENIZERS_PARALLELISM'] = "false"
os.environ['NGROK_AUTHTOKEN'] = 'Your ngrok authtoken'

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from langserve import add_routes

from src.base.llm_model import get_hf_llm
from src.rag.main import build_rag_chain, InputQA, OutputQA

from pyngrok import ngrok
import nest_asyncio
import uvicorn


gen_kwarg = {
    'temperature': 0.7
}
llm = get_hf_llm(kwargs=gen_kwarg)
genai_docs = './data_source/generative_ai'

#---------------------- Chains ----------------------------
genai_chain = build_rag_chain(llm, data_dir=genai_docs, data_type='pdf')

#-------------------- App - FastAPI -----------------------
app = FastAPI(
    title='Langchain Server',
    version='1.0',
    description="A simple API server using LangChain's Runnable interfaces",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
    expose_headers=['*']
)

#----------------- Routes - FastAPI ------------------------
@app.get('/check')
async def check():
    return {'status': 'ok'}

@app.post('/generative_ai', response_model=OutputQA)
async def generative_ai(inputs: InputQA):
    answer = genai_chain.invoke(inputs.question)
    return {'answer': answer}

#-------------- Langserve Routes - Playground ---------------
add_routes(app,
           genai_chain,
           playground_type='default',
           path='/generative_ai'
           )

#--------------------- Run API ------------------------------
ngrok_tunnel = ngrok.connect(8000)
print('Public URL:', ngrok_tunnel.public_url)
nest_asyncio.apply()
uvicorn.run(app, port=8000)
