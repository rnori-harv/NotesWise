import PyPDF2
from dotenv import dotenv_values, load_dotenv, find_dotenv
import openai
import langchain
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
import pypdf
import time

from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

import streamlit as st  

env_vars = dotenv_values('../.env')
load_dotenv(dotenv_path='../.env')

# Access the values
openai.api_key = env_vars['OPENAI_API_KEY']
OPENAI_API_KEY = env_vars['OPENAI_API_KEY']

st.title('NotesWise')
st.write('NotesWise is a tool that allows you to ask questions about your notes and get answers from your notes. It is powered by OpenAI\'s GPT-3 and LangChain\'s LangLearner Model. To get started, upload your notes in PDF format below.')

# BASIC MODEL with Prompt engineering
def read_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        num_pages = len(reader.pages)
        ans = ""

        for page_number in range(num_pages):
            page = reader.pages[page_number]
            text = page.extract_text()
            ans += text
        return ans

def pass_knowledge_to_openai(text):
    prompt = "This is your knowledge base, only use the following content in this prompt for your answers. If a question response cannot be found within the text provided, respond to the question with \"NOT POSSIBLE TO ANSWER\". Here is your knowledge base: " + text + "Question: What is a Y-combinator?"
    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=prompt,
        max_tokens=100,
        temperature=0.7,
        n=1,
        stop=None
    )
    generated_text = response.choices[0].text.strip()
    return generated_text


def load_langchain_model(file_paths):
    loaders = [PyPDFLoader(file_path) for file_path in file_paths]
    index = VectorstoreIndexCreator().from_loaders(loaders)
    return index

def query_model(model, query):
    results = model.query(query)
    return results

    
pdf_file_paths = ['./152/lec01-intro.pdf', './152/lec02-smallstep.pdf', './152/lec03-inductive-proof.pdf'
                  , './152/lec04-largestep.pdf', './152/lec05-imp.pdf', './152/lec06-denotational.pdf']

# BASIC MODEL with Prompt engineering
#pased_text = read_pdf(pdf_file_path)
#print(pass_knowledge_to_openai(pased_text))

# LANGCHAIN MODEL:
files = st.file_uploader("Upload your lecture note files (PDF)", type=["pdf"], accept_multiple_files=True)
while files == []:
    time.sleep(0.5)
file_paths = []
for file in files:
    file_paths.append(file.name)



model = load_langchain_model(file_paths)

# User input
prompt = st.text_input('Enter your question here:')
if prompt != '':
    res = query_model(model, prompt)
    st.write(res)

