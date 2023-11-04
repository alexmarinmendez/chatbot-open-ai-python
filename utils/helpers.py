import openai
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai.api_key  = os.getenv('OPENAI_API_KEY')

# account for deprecation of LLM model
import datetime

def define_model():
    # Get the current date
    current_date = datetime.datetime.now().date()

    # Define the date after which the model should be set to "gpt-3.5-turbo"
    target_date = datetime.date(2024, 6, 12)

    # Set the model variable based on the current date
    if current_date > target_date:
        llm_model = "gpt-3.5-turbo"
    else:
        llm_model = "gpt-3.5-turbo-0301"
    return llm_model

def get_completion(prompt, model = define_model()):
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