#!/usr/bin/python3

from os import environ
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor, load_tools
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActJsonSingleInputOutputParser
from models import Llama3, CodeLlama, Qwen2, CodeQwen1_5, Finance
from prompts import agent_template
from tools import load_vectordb, load_knowledge_graph, load_database
import config

class Agent(object):
  def __init__(self, model = 'llama3', code_model = 'codellama', tools = ["google-serper", "llm-math", "wikipedia", "arxiv"]):
    llms_types = {
      'llama3': Llama3,
      'codellama': CodeLlama,
      'qwen2': Qwen2,
      'codeqwen': CodeQwen1_5,
      'finance': Finance
    }
    tokenizer, llm = llms_types[model](config.run_locally)
    if model == code_model:
      code_tokenizer, code_llm = tokenizer, llm
    else:
      code_tokenizer, code_llm = llms_types[code_model](config.run_locally)
    unstructure_tool = load_vectordb(
                         host = config.neo4j_host,
                         username = config.neo4j_username,
                         password = config.neo4j_password,
                         database = config.neo4j_db,
                         tokenizer = tokenizer,
                         llm = llm) if config.unstructure_method == 'RAG' else \
                       load_knowledge_graph(
                         host = config.neo4j_host,
                         username = config.neo4j_username,
                         password = config.neo4j_password,
                         database = config.neo4j_db,
                         tokenizer = tokenizer,
                         llm = llm)
    structure_tool = load_database(
                       'bs_challenge_financial_14b_dataset/dataset/博金杯比赛数据.db',
                       tokenizer = code_tokenizer,
                       llm = code_llm)
    tools = load_tools(tools, llm = llm, serper_api_key = 'd075ad1b698043747f232ec1f00f18ee0e7e8663') + \
      [unstructure_tool, structure_tool]
    prompt = agent_template(tokenizer, tools)
    llm = llm.bind(stop = ["<|eot_id|>"])
    chain = {"input": lambda x: x["input"], "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"])} | prompt | llm | ReActJsonSingleInputOutputParser()

    memory = ConversationBufferMemory(memory_key="chat_history")
    self.agent_chain = AgentExecutor(agent = chain, tools = tools, memory = memory, verbose = True)
  def query(self, question):
    return self.agent_chain.invoke({"input": question})

if __name__ == "__main__":
  agent = Agent(model = 'llama3')
  print(agent.query("what is SwiGLU?"))
