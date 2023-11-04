import openai
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
from utils.helpers import define_model

openai.api_key  = os.getenv('OPENAI_API_KEY')

from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
chunk_size = 1000 # tamaño de cada fragmento
chunk_overlap = 200   # superposición entre fragmentos 
r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)
c_splitter = CharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)
import langchain.document_loaders as loaders

from langchain.document_loaders import TextLoader
loader = TextLoader("eBook-How-to-Build-a-Career-in-AI_translated.txt", encoding = "utf-8")
paginas = loader.load()

trozos = r_splitter.split_documents(paginas)
from langchain.embeddings.openai import OpenAIEmbeddings
embedding = OpenAIEmbeddings()


from langchain.vectorstores import Chroma
persist_directory = 'docs/chroma/'
vectordb = Chroma.from_documents(
    documents=trozos,
    embedding=embedding,
    persist_directory=persist_directory
)

vectordb.persist()

from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(model_name = define_model(), temperature = 0)
from langchain.chains import RetrievalQA
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(K = 5)
)

while True:
    pregunta = input("Hola Mundo! Estoy listo para tu pregunta :)    ")
    if pregunta == "FIN":
        break 
    else:
        result = qa_chain({"query": pregunta})
        print('---')    
        print(result["result"])
        print('---')    
        print()
