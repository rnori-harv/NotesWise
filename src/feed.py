import PyPDF2
from dotenv import dotenv_values, load_dotenv
import openai
from langchain import SerpAPIWrapper
import time
import re
from io import BytesIO
from typing import List, Union
import os
import json



from langchain.chains import RetrievalQA
from langchain.chains.summarize import load_summarize_chain
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
    GPT_MODEL_VERSION = 'gpt-3.5-turbo-16k'

openai.api_key = st.secrets['OPENAI_API_KEY']

@st.cache_resource(show_spinner=False)
def load_llm_chain():
    llm = ChatOpenAI(temperature=0, model = GPT_MODEL_VERSION)
    summarizer = load_summarize_chain(llm, chain_type = "stuff")
    return llm, summarizer

llm, summarizer = load_llm_chain()

def parse_ans_gpt35(message):
    split_message = message.split('Action:\n')
    if len(split_message) == 1:
        return message
    json_part = message.split('Action:\n')[1]
    # Parse the JSON string
    data = json.loads(json_part)
    # Extract the value of "action_input"
    action_input = data["action_input"]
    return action_input

    

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
        llm=ChatOpenAI(model = GPT_MODEL_VERSION), chain_type="stuff", retriever=retriever, return_source_documents=True)
    return qa

def query_langchain_model(model, query):
    ans = model({"query": query})
    summary = summarizer.run(ans["source_documents"])
    return ans["result"], ans["source_documents"], summary

def get_source_info(prompt):
    if prompt == "":
        return "Please provide a non-empty prompt."
    res, source_docs, summary = query_langchain_model(model, prompt)
    return summary
    

# Set up a prompt template
def generate_prompt(prompt, source_info):
    prompt = f"""
    Please answer the student's question in a step by step manner with clear explanation using the source information provided.
    Source information:
    {source_info}
    Student's question:
    {prompt}
    Answer:
    """
    return prompt



def llm_agent():
    search = SerpAPIWrapper()
    llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
    tools = [
        Tool(name = "Check lecture notes", func = get_source_info, description = "Useful for when you need to consult information within your knowledge base. Provide a non-empty query as the argument to this. Use this before searching online."),
        Tool(name = "Search Online", func = search.run, description = "Useful for when you need to consult extra information not found in the lecture notes."),
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

@st.cache_resource(show_spinner=False)
def setup_ta():
    docs = []
    with st.spinner('Reading your notes...'):
        for file in files:
            reader = PyPDF2.PdfReader(BytesIO(file.read()))
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                page_content = page.extract_text().splitlines()
                page_content_str = ''.join(page_content)
                curr_doc = Document(page_content=page_content_str, metadata={"source": file.name, "page": page_num + 1})
                docs.append(curr_doc)
    with st.spinner('Preparing your TA'):
        model = load_langchain_model(docs)
        my_agent = llm_agent()
        return model, my_agent
        

model, my_agent = setup_ta()

def online_agent(prompt, information_consulted):
    with st.spinner('TA is searching online and coming up with an answer. Note that this may take some time...'):
        full_prompt = generate_prompt(prompt, information_consulted)
        output = my_agent.run(full_prompt)
        if GPT_MODEL_VERSION == 'gpt-3.5-turbo-16k':
            ans = parse_ans_gpt35(output)
        else:
            ans = output
    st.markdown("<h1 style='text-align: center; color: green; font-family: sans-serif'>Final Answer from Agent:</h1>", unsafe_allow_html=True)
    st.write(ans)



# User input
# Initialize the session state for the text area if it doesn't exist

# Use the session state in the text area
# Create a placeholder for the text area

prompt = st.text_area('Enter your question here:', key = "prompt")


def clear_prompt():
    st.session_state["prompt"] = ""
    st.session_state['ask_ta_clicked'] = False

if 'ask_ta_clicked' not in st.session_state:
    st.session_state['ask_ta_clicked'] = False


@st.cache_data(show_spinner=False)
def print_docsearch(prompt):
    with st.spinner('TA is thinking...'):
        res, source_docs, summary = query_langchain_model(model, prompt)
    st.markdown("<h1 style='text-align: center; color: green; font-family: sans-serif'>Answer from knowledge base:</h1>", unsafe_allow_html=True)
    st.write(res)
    st.markdown("<h2 style='text-align: center; color: orange; font-family: sans-serif'>Source information consulted:</h2>", unsafe_allow_html=True)
    for doc in source_docs:
        source_loc = doc.metadata["source"] + ", Page " + str(doc.metadata["page"])
        st.markdown(f"<p style='text-align: center; color: orange; font-family: sans-serif'>{source_loc}</p>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: orange; font-family: sans-serif'>Summary of notes consulted:</h2>", unsafe_allow_html=True)
    st.write(summary)
    return summary


# Use the session state for the 'Ask TA' button
if st.button('Ask TA'):
    st.session_state['ask_ta_clicked'] = True
# Display the initial text area in the placeholder
if st.session_state['ask_ta_clicked'] and prompt != "":
    summary = print_docsearch(prompt)
    if st.button('Run online agent'):
        st.session_state['ask_ta_clicked'] = True
        online_agent(prompt, summary)

    if summary:
        st.button('Ask another question', on_click = clear_prompt)

