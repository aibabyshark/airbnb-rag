import os
import chainlit as cl
from dotenv import load_dotenv
from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyMuPDFLoader
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_core.prompts import ChatPromptTemplate
from operator import itemgetter
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.runnable.config import RunnableConfig


# GLOBAL SCOPE - ENTIRE APPLICATION HAS ACCESS TO VALUES SET IN THIS SCOPE #
# ---- ENV VARIABLES ---- # 
"""
This function will load our environment file (.env) if it is present.

NOTE: Make sure that .env is in your .gitignore file - it is by default, but please ensure it remains there.
"""
load_dotenv()

"""
We will load our environment variables here.
"""
# ---- OpenAI LLM ---- #

openai_chat_model = ChatOpenAI(model="gpt-4o")



# ---- GLOBAL DECLARATIONS ---- #

# -- RETRIEVAL -- #

### 1. CREATE TEXT LOADER AND LOAD DOCUMENTS
### NOTE: PAY ATTENTION TO THE PATH THEY ARE IN. 
file_path = "./data/airbnb10kfilling.pdf"
docs = PyMuPDFLoader(file_path).load()


### 2. CREATE TEXT SPLITTER AND SPLIT DOCUMENTS

def tiktoken_len(text):
    tokens = tiktoken.encoding_for_model("gpt-4o").encode(
        text,
    )
    return len(tokens)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 200,
    chunk_overlap = 10,
    length_function = tiktoken_len,
)

split_chunks = text_splitter.split_documents(docs)
### 3. LOAD HUGGINGFACE EMBEDDINGS

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")



qdrant_vectorstore = Qdrant.from_documents(
    split_chunks,
    embedding_model,
    location=":memory:",
    collection_name="airbnb 10K filling",
)

qdrant_retriever = qdrant_vectorstore.as_retriever()

# -- AUGMENTED -- #
"""
1. Define a String Template
2. Create a Prompt Template from the String Template
"""

RAG_PROMPT_TEMPLATE = """
CONTEXT:
{context}

QUERY:
{query}

You are a helpful assistant. You answer user questions based on provided context. If you can't answer the question with the provided context, say you don't know."
"""

rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

# -- GENERATION -- #

@cl.on_chat_start
async def start_chat():
    lcel_rag_chain = (
        {"context": itemgetter("query") | qdrant_retriever, "query": itemgetter("query")}
        | rag_prompt | openai_chat_model
    )

    cl.user_session.set("lcel_rag_chain", lcel_rag_chain)

@cl.on_message
async def main(message: cl.Message):
    lcel_rag_chain = cl.user_session.get("lcel_rag_chain")

    msg = cl.Message(content="")

    results = await cl.make_async(lcel_rag_chain.stream)(
        {"query": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    )

    for chunk in results:
        await msg.stream_token(chunk.content)

    await msg.send()

