import PyPDF2
from dotenv import dotenv_values, load_dotenv
import openai
from langchain import SerpAPIWrapper
import time
import re
from io import BytesIO
from typing import List, Union
import os



from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate, BaseChatPromptTemplate
from langchain.chains import LLMChain
from langchain.agents import load_tools, Tool, AgentOutputParser
from langchain.agents import initialize_agent
from langchain.schema import HumanMessage, Document
from langchain import LLMMathChain

from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from langchain.prompts import BaseChatPromptTemplate, ChatPromptTemplate
from langchain import SerpAPIWrapper, LLMChain
from langchain.schema import AgentAction, AgentFinish, HumanMessage, SystemMessage
from langchain.chat_models import ChatOpenAI
# LLM wrapper
from langchain import OpenAI
# Conversational memory
# Embeddings and vectorstore
from langchain.embeddings.openai import OpenAIEmbeddings

import streamlit as st
import os

# Access the values
st.title('NotesWise')
st.write('NotesWise is a tool that allows you to use your notes to help you solve your problem sets! Put in your lecture notes, ask a question about your problem set, and Noteswise uses your notes as well as the internet to solve it. It is powered by OpenAI\'s GPT-4 and Langchain. To get started, upload your notes in PDF format below.')

os.environ["SERPAPI_API_KEY"] = st.secrets["SERPAPI_API_KEY"]

GPT_MODEL_VERSION = 'gpt-4'
if 'OPENAI_ORG' in st.secrets:
    openai.organization = st.secrets['OPENAI_ORG']
    GPT_MODEL_VERSION = 'gpt-3.5-turbo-16k-0613'

openai.api_key = st.secrets['OPENAI_API_KEY']

# BASIC MODEL with Prompt engineering
def load_langchain_model(docs):
    # select which embeddings we want to use
    embeddings = OpenAIEmbeddings()
    # create the vectorestore to use as the index
    db = Chroma.from_documents(docs, embeddings)
    # expose this index in a retriever interface
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":2})
    # create a chain to answer questions 
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(), chain_type="stuff", retriever=retriever, return_source_documents=True)
    return qa

def query_langchain_model(model, query):
    ans = model({"query": query})
    return ans["result"], ans["source_documents"]

def get_source_info(prompt):
    res, source_docs = query_langchain_model(model, prompt)
    information_consulted = []
    for doc in source_docs:
        information_consulted.append(doc.page_content)
    return information_consulted

# Set up a prompt template
def generate_prompt(prompt, source_info):
    prompt = f"""
    Please help the student solve the following problem set question in a step by step format using the source information provided.
    Question:
    {prompt}
    Source information:
    {source_info}
    Answer:
    """
    return prompt

search = SerpAPIWrapper()

def llm_agent():
    llm = OpenAI(temperature=0, model = GPT_MODEL_VERSION)
    llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
    tools = [
        Tool(name = "Check lecture notes", func = get_source_info, description = "Useful for when you need to consult information within your knowledge base. Use this before searching online."),
        Tool(name = "Search Online", func = search.run, description = "Useful for when you need to consult information when check lecture notes does not give you enough information."),
         Tool(name="Calculator", func=llm_math_chain.run, description="useful for when you need to answer questions about math")
    ]
    openai_model = ChatOpenAI(temperature=0, model = GPT_MODEL_VERSION)
    planner = load_chat_planner(openai_model)
    executor = load_agent_executor(openai_model, tools, verbose=True)
    planner_agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)
    return planner_agent


files = st.file_uploader("Upload your lecture note files (PDF)", type=["pdf"], accept_multiple_files=True)
while files == []:
    time.sleep(0.5)
pdf_files = []
docs = []
with st.spinner('Reading your notes...'):
    for file in files:
        reader = PyPDF2.PdfReader(BytesIO(file.read()))
        file_content = []
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            page_content = page.extract_text().splitlines()
            page_content_str = ''.join(page_content)
            curr_doc = Document(page_content=page_content_str, metadata={"source": file.name, "page": page_num + 1})
            docs.append(curr_doc)
    model = load_langchain_model(docs)
    my_agent = llm_agent()


# User input
prompt = st.text_area('Enter your question here:')
if prompt != '':
    res, source_docs = query_langchain_model(model, prompt)
    st.markdown("<h1 style='text-align: center; color: green; font-family: sans-serif'>Answer from knowledge base:</h1>", unsafe_allow_html=True)
    st.write(res)
    st.markdown("<h2 style='text-align: center; color: orange; font-family: sans-serif'>Lecture notes consulted:</h2>", unsafe_allow_html=True)
    information_consulted = []
    for doc in source_docs:
        information_consulted.append(doc.page_content)
        source_loc = doc.metadata["source"] + ", Page " + str(doc.metadata["page"])
        st.markdown(f"<p style='text-align: center; color: orange; font-family: sans-serif'>{source_loc}</p>", unsafe_allow_html=True)

    full_prompt = generate_prompt(prompt, information_consulted)
    ans = my_agent.run(full_prompt)
    st.markdown("<h1 style='text-align: center; color: green; font-family: sans-serif'>Answer from Agent:</h1>", unsafe_allow_html=True)
    st.write(ans)

    

