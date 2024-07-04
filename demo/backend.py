from pydantic import BaseModel
import langchain
import openai
import json
import os
import pandas as pd
import typing
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document

from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.agents import AgentExecutor
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema import HumanMessage, SystemMessage
from langchain.chat_models import ChatOpenAI, ChatOllama
import time


import PIRO
import PRO
def sleep(model_type: str):
    if model_type == "openAI":
        time.sleep(10)
    else:
        return


def get_chain(prompt_template, model_name: str = "gpt-3.5-turbo",
              model_type: str = "openAI", temperature: float = 0, openai_key: str = "openai_key"):
    prompt = PromptTemplate.from_template(template=prompt_template)
    if model_type == "openAI":
        model = ChatOpenAI(temperature=0, model_name=model_name,
                           openai_api_key=openai_key)
    else:
        model = ChatOllama(model_name=model_name, temperature=temperature)
    chain = prompt | model | StrOutputParser()
    return chain, prompt


def load_model(model_name: str = "gpt-3.5-turbo", model_type: str = "openAI", temperature: float = 0, openai_key: str = "openai_key"):
    if model_type == "openAI":
        model = ChatOpenAI(temperature=0, model_name=model_name,
                           openai_api_key=openai_key)
    else:
        model = ChatOllama(model_name=model_name, temperature=temperature)
    return model


def get_direct_result(query: str, tools: typing.List, examples: typing.List, model_type: str = "openAI", model_name: str = "gpt-3.5-turbo", openai_key: str = "openai_key"):
    # sleep(model_type)
    model = load_model(model_name=model_name,
                       model_type=model_type, temperature=0, openai_key=openai_key)
    # #print(query)
    sol_1 = model([HumanMessage(content=query)])
    return sol_1, ((query, sol_1))


def get_result_piro(query: str, tools: typing.List, examples: typing.List, model_type: str = "openAI", model_name: str = "gpt-3.5-turbo", openai_key: str = "openai_key"):
    message_history = []
    # sleep(model_type)
    #!PLANNING
    planning_chain, planning_prompt = get_chain(
        prompt_template=PIRO.planning_template, model_name=model_name, model_type=model_type,
        temperature=0, openai_key=openai_key
    )
    prompt_1 = planning_prompt.format(query=query, tools=tools)
    sol_1 = planning_chain.invoke({"query": query, "tools": tools})
    # print(f"Output 1: {sol_1}")

    message_history.append((prompt_1, sol_1))

    # sleep(model_type)
    #! IMPROVEMENT
    improvement_chain, improvement_prompt = get_chain(
        prompt_template=PIRO.improvement_template, model_name=model_name, model_type=model_type,
        temperature=0, openai_key=openai_key
    )
    sol_2 = improvement_chain.invoke(
        {"examples": examples, "query": query, "solution": sol_1})
    # print(f"Output 2: {sol_2}")
    prompt_2 = improvement_prompt.format(
        examples=examples, query=query, solution=sol_1)
    message_history.append((prompt_2, sol_2))

    #! IMPROVEMENT2
    # sleep(model_type)
    improvement_chain, improvement_prompt = get_chain(
        prompt_template=PIRO.improvement_template, model_name=model_name, model_type=model_type,
        temperature=0, openai_key=openai_key
    )
    sol_3 = improvement_chain.invoke(
        {"examples": examples, "query": query, "solution": sol_2})
    prompt_3 = improvement_prompt.format(
        examples=examples, query=query, solution=sol_2)
    message_history.append((prompt_3, sol_3))

    #! OPTIMIZATION
    # sleep(model_type)
    optimization_chain, optimization_prompt = get_chain(
        prompt_template=PIRO.optimization_template, model_name=model_name, model_type=model_type,
        temperature=0, openai_key=openai_key
    )
    sol_4 = optimization_chain.invoke(
        {"query": query, "solution": sol_3, "tools": tools})
    prompt_4 = optimization_prompt.format(
        query=query, solution=sol_3, tools=tools)
    message_history.append((prompt_4, sol_4))

    return sol_4, message_history




def get_result_pro(query: str, tools: typing.List, examples: typing.List, model_type: str = "openAI", model_name: str = "gpt-4-1106-preview", openai_key: str = "openai_key"):
    message_history = []
    # sleep(model_type)
    #!PLANNING
    planning_chain, planning_prompt = get_chain(
        prompt_template=PRO.planning_template, model_name=model_name, model_type=model_type,
        temperature=0, openai_key=openai_key
    )
    prompt_1 = planning_prompt.format(query=query, tools=tools)
    sol_1 = planning_chain.invoke({"query": query, "tools": tools})
    # print(f"Output 1: {sol_1}")

    message_history.append((prompt_1, sol_1))

    # sleep(model_type)
    #! IMPROVEMENT
    improvement_chain, improvement_prompt = get_chain(
        prompt_template=PRO.improvement_template, model_name=model_name, model_type=model_type,
        temperature=0, openai_key=openai_key
    )
    sol_2 = improvement_chain.invoke(
        {"examples": examples, "query": query, "solution": sol_1})
    # print(f"Output 2: {sol_2}")
    prompt_2 = improvement_prompt.format(
        examples=examples, query=query, solution=sol_1)
    message_history.append((prompt_2, sol_2))

    #! OPTIMIZATION
    # sleep(model_type)
    optimization_chain, optimization_prompt = get_chain(
        prompt_template=PRO.optimization_template, model_name=model_name, model_type=model_type,
        temperature=0, openai_key=openai_key
    )
    sol_3 = optimization_chain.invoke(
        {"query": query, "solution": sol_2, "tools": tools})
    prompt_3 = optimization_prompt.format(
        query=query, solution=sol_2, tools=tools)
    message_history.append((prompt_3, sol_3))
    # print(f"Output 3: {sol_4}")

    return sol_3, message_history

