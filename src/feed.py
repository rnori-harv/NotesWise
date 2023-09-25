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
from langchain.callbacks import StreamlitCallbackHandler
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate, BaseChatPromptTemplate
from langchain.chains import LLMChain
from langchain.agents import load_tools, AgentOutputParser, AgentType
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from langchain.agents import initialize_agent
from langchain.schema import HumanMessage, Document
from langchain import LLMMathChain
from pydantic import BaseModel, Field
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

openai.api_key = st.secrets['OPENAI_API_KEY']



SYSTEM_PROMPT = "You are an AI tutor that helps students solve complex problem set questions using a variety of online tools: the student's lecture notes, online search, and a python interpreter. You will do tasks in the following order: determine the problem's topic, search lecture notes for similar problems / answers of that same topic as well as relevant concepts of that topic, search online for any additional information you might need to solve the problem, and finally a calculator."

TOPIC_PROMPT = '''
In this user's question, extract the topic that this question is asking about and present it to the user. Here is an example: 
\nQuestion: 
True or False: If X and Y have the same CDF, they have the same expectation.

Probability Distribution and expectation

Question: \n
''' 

def reasoning_prompt(user_question, question_topic, notes_info, online_info):
    return f'''
    System context: {SYSTEM_PROMPT}
    Question: {user_question}
    Topic: {question_topic}
    Notes Info: {notes_info}
    Online Info: {online_info}
    Given all of this information, answer the student's question in a step by step manner with clear explanation. If you need to use a calculator, use the calculator tool.
    Answer: 
    '''

@st.cache_resource(show_spinner=False)
def load_llm_chain():
    llm = ChatOpenAI(temperature=0, model = GPT_MODEL_VERSION, streaming = True)
    summarizer = load_summarize_chain(llm, chain_type = "stuff")
    return llm, summarizer

llm, summarizer = load_llm_chain()

def get_topic(user_question):
    return ChatOpenAI(temperature=0, model = GPT_MODEL_VERSION, streaming = True).predict(TOPIC_PROMPT + user_question)
    

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
        llm=ChatOpenAI(model = GPT_MODEL_VERSION, streaming = True), chain_type="stuff", retriever=retriever, return_source_documents=True)
    return qa


def search_notes(notes_llm, user_question, topic):
    search_prompt = f'''
    Question: {user_question}
    Topic: {topic}
    Search the student's lecture notes for similar problems / answers of that same topic as well as relevant concepts of that topic.
    '''
    ans = notes_llm({"query": search_prompt})

    notes_info = ""
    for page in ans["source_documents"]:
        notes_info += page.page_content
    
    return notes_info


class CalculatorInput(BaseModel):
    question: str = Field(..., description="The input must be a numerical expression")

llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
def calculator_func(input):
    try:
        if input is None or input.strip() == '':
            return 'Try again with a non-empty input.'
        else:
            return llm_math_chain.run(input)
    except Exception as e:
        return 'Try again with a valid numerical expression.'

def online_search_prompt(user_question, topic):
    return f'''
    Question: {user_question}
    Topic: {topic}
    Search online for any additional information you might need to solve the problem.
    '''
    
@st.cache_resource(show_spinner=False)
def reasoning_agent(user_question):
    search = SerpAPIWrapper()
    openai_model = ChatOpenAI(temperature=0, model = GPT_MODEL_VERSION, streaming = True)
    user_topic = get_topic(user_question)
    st.session_state.messages.append({"role": "assistant", "content": "seems like the question is about "+ user_topic})
    st.write("your question topic: " + user_topic)
    notes_info = search_notes(notes_llm, user_question, user_topic)
    st.session_state.messages.append({"role": "assistant", "content": "here is some information from your notes that might be helpful: "+ notes_info})
    st.write("some infromation from your notes: " + notes_info)
    online_info = search.run(online_search_prompt(user_question, user_topic))
    st.session_state.messages.append({"role": "assistant", "content": "here is some information from the internet that might be helpful: "+ online_info})
    st.write("relevant information from the internet: " + online_info)

    full_prompt = reasoning_prompt(user_question, user_topic, notes_info, online_info)
    calc_tool = Tool(name="Calculator", func=calculator_func , description="useful for when you need to answer questions about math. Only put numerical expressions in this.", args_schema=CalculatorInput)

    agent = initialize_agent([calc_tool], openai_model, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    st_callback = StreamlitCallbackHandler(st.container())
    output = agent.run(full_prompt, callbacks = [st_callback])
    return output


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
        notes_llm = load_langchain_model(docs)
    return notes_llm
        

notes_llm = setup_ta()

# User input
# Initialize the session state for the text area if it doesn't exist

# Use the session state in the text area
# Create a placeholder for the text area

# prompt = st.text_area('Enter your question here:', key = "prompt")


def clear_prompt():
    st.session_state["prompt"] = ""
    st.session_state['ask_ta_clicked'] = False

if 'ask_ta_clicked' not in st.session_state:
    st.session_state['ask_ta_clicked'] = False

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask your question here:"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})


    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = reasoning_agent(prompt)    # Add assistant response to chat history
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

