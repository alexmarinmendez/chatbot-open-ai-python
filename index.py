import openai
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
from utils.helpers import define_model

openai.api_key  = os.getenv('OPENAI_API_KEY')
llm_model = define_model()



def get_completion(prompt, model = llm_model):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # Es el grado de aleatoriedad de la salida del modelo 
    )
    return response.choices[0].message["content"]


# from langchain.document_loaders import PyPDFLoader
# loader = PyPDFLoader("eBook-How-to-Build-a-Career-in-AI.pdf")
# pages = loader.load()


# def traducir(texto, model="gpt-3.5-turbo"):
#     messages = [
#             {"role": "user", "content": f"""
#             Tu rol es el de un traductor profesional inglés español. \
#             Traduce todo el texto de cada página del inglés al español (latinoamericano) en \
#             la forma mas exacta, completa y profesional que se pueda:{texto} """
#             }
#         ]
#     response = openai.ChatCompletion.create(
#         model=model,
#         messages=messages,
#         temperature=0, #  # Es el grado de aleatoriedad de la salida del modelo
#     )
#     return response.choices[0].message["content"]

# with open("eBook-How-to-Build-a-Career-in-AI_translated.txt", "w") as file:
#     hoja = 1
#     for pagina in pages:
#         pagina_esp = traducir(pagina.page_content)
#         print('Traduciendo la pagina {} de {}'.format(pagina, len(pages)))   

# with open("eBook-How-to-Build-a-Career-in-AI_translated.txt", "w", encoding = "UTF-8", errors = "ignore") as file:
#     hoja = 1
#     for pagina in pages:
#         print("Imprimiendo la hoja..." + str(hoja) + ".........")
#         pagina_esp = traducir(pagina.page_content)     
#         file.write(pagina_esp + "\n\n"  )
#         hoja +=1

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
# print(vectordb._collection.count())

vectordb.persist()

question = "¿Cuáles son los pasos principales para desarrollar una carrera profesional\
en Inteligencia Artificial?"

respuestas = vectordb.similarity_search(question, k = 3)

from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(model_name = llm_model, temperature = 0)
from langchain.chains import RetrievalQA
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(K = 5)
)

# result = qa_chain({"query": question})
# print(result["result"])

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
