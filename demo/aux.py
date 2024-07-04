
#! Auxiliary functions :).
import requests
import json
import typing
import langchain
import json
import openai
import os
import pandas as pd

from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document

from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.agents import AgentExecutor
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.chat_models import ChatOpenAI
import nltk
import stanza
import backend
import re
stanza.download('en')
nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')





#++++++++++++++++++++++++++++++ STANZA +++++++++++++++++++++++++++
def get_verb_phrases(t):
    verb_phrases = []
    num_children = len(t)
    num_VP = sum(1 if t[i].label() == "VP" else 0 for i in range(0, num_children))

    if t.label() != "VP":
        for i in range(0, num_children):
            if t[i].height() > 2:
                verb_phrases.extend(get_verb_phrases(t[i]))
    elif t.label() == "VP" and num_VP > 1:
        for i in range(0, num_children):
            if t[i].label() == "VP":
                if t[i].height() > 2:
                    verb_phrases.extend(get_verb_phrases(t[i]))
    else:
        verb_phrases.append(' '.join(t.leaves()))
    return verb_phrases

def get_pos(t):
    vp_pos = []
    sub_conj_pos = []
    num_children = len(t)
    children = [t[i].label() for i in range(0, num_children)]
    flag = re.search(r"(S|SBAR|SBARQ|SINV|SQ)", ' '.join(children))
    if "VP" in children and not flag:
        for i in range(0, num_children):
            if t[i].label() == "VP":
                vp_pos.append(t[i].treeposition())
    elif not "VP" in children and not flag:
        for i in range(0, num_children):
            if t[i].height() > 2:
                temp1, temp2 = get_pos(t[i])
                vp_pos.extend(temp1)
                sub_conj_pos.extend(temp2)
    else:
        for i in range(0, num_children):
            if t[i].label() in ["S", "SBAR", "SBARQ", "SINV", "SQ"]:
                temp1, temp2 = get_pos(t[i])
                vp_pos.extend(temp1)
                sub_conj_pos.extend(temp2)
            else:
                sub_conj_pos.append(t[i].treeposition())
    return (vp_pos, sub_conj_pos)

def print_clauses(parse_str):
    sent_tree = nltk.tree.ParentedTree.fromstring(parse_str)
    clause_level_list = ["S", "SBAR", "SBARQ", "SINV", "SQ"]
    clause_list = []
    sub_trees = []
    for sub_tree in reversed(list(sent_tree.subtrees())):
        if sub_tree.label() in clause_level_list:
            if sub_tree.parent().label() in clause_level_list:
                continue
            if (len(sub_tree) == 1 and sub_tree.label() == "S" and sub_tree[0].label() == "VP"
                    and not sub_tree.parent().label() in clause_level_list):
                continue
            sub_trees.append(sub_tree)
            del sent_tree[sub_tree.treeposition()]
    for t in sub_trees:
        verb_phrases = get_verb_phrases(t)
        vp_pos, sub_conj_pos = get_pos(t)
        for i in reversed(vp_pos):
            del t[i]
        for i in reversed(sub_conj_pos):
            del t[i]
        subject_phrase = ' '.join(t.leaves())
        for i in verb_phrases:
            clause_list.append(subject_phrase + " " + i)
    return clause_list

def get_clauses(query):
    doc = nlp(query)
    z = doc.sentences
    all_clauses = []
    for i in range(len(z)):
        clauses = print_clauses(parse_str = str(z[i].constituency))
        all_clauses += clauses
    if len(all_clauses) == 0:
        all_clauses = [query]
    return all_clauses

#++++++++++++++++++++++++++++++ STANZA +++++++++++++++++++++++++++

def get_tools(query : str, documentation : typing.List, retriever : typing.Any):
    docs = retriever.get_relevant_documents(query)
    tools = [documentation[d.metadata["index"]]['tool_name'] for d in docs]
    arguments = [documentation[d.metadata["index"]] for d in docs]
    return tools, arguments

def get_tools_stanza(query : str, documentation : typing.List, retriever : typing.Any):
    all_clauses = get_clauses(query)
    all_docs = []
    
    for clauses in all_clauses:
        docs = retriever.get_relevant_documents(clauses)
        for doc in docs:
            if doc.metadata["index"] not in all_docs:
                all_docs.append(doc.metadata["index"])
    tool_names = [documentation[idx]['tool_name'] for idx in all_docs]
    tools_with_arguments = [documentation[idx] for idx in all_docs]
    return tool_names, tools_with_arguments

def get_examples(query : str, examples : typing.List, retriever : typing.Any):
    """    Returns:
        _type_: _description_
    """
    docs = retriever.get_relevant_documents(query)
    return docs, [examples[d.metadata["index"]] for d in docs]



def post_request(query : str, documentation : typing.List, examples : typing.List, open_api_key : str,
                  model_name : str, prompting_technique : str, use_stanza : bool, api_ret : typing.Any, ex_ret : typing.Any):
    """Posts a request to the server

    Args:
        query (str): Query
        documentation (typing.List): API Docs
        examples (typing.List): Examples
        open_api_key (str): Duh
        model_name : Duh
        parse_piro (bool): Whether to use PIRO Prompting
        api_ret, ex_ret : Retriever

    Returns:
        JSON
    """
    parse_piro = False if prompting_technique == 'None' else True
    if not parse_piro:
        tools = []
        examples = []
    else:
        _, examples = get_examples(query, examples, ex_ret)
        if use_stanza:
            _, tools = get_tools_stanza(query, documentation, api_ret)
        else:
            _, tools = get_tools(query, documentation, api_ret)
    
    if prompting_technique == 'None':
        return backend.get_direct_result(query, tools, examples, model_type="openAI", model_name=model_name, openai_key=open_api_key)
    elif prompting_technique == 'PIRO':
        return backend.get_result_piro(query, tools, examples, model_type="openAI", model_name=model_name, openai_key=open_api_key)
    else:
        return backend.get_result_pro(query, tools, examples, model_type="openAI", model_name=model_name, openai_key=open_api_key)
