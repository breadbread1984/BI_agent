#!/usr/bin/python3

from os import environ, remove
from os.path import exists, join, isdir
from typing import Optional, Type
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from langchain.graphs import Neo4jGraph
from langchain_community.llms import HuggingFaceEndpoint
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from prompts import entity_generation_template, triplets_qa_template, sqlite_prompt

def load_knowledge_graph(host = 'bolt://localhost:7687', username = 'neo4j', password = None, database = 'neo4j'):
  class ProspectusInput(BaseModel):
    query: str = Field(description = "招股说明书相关的问题")

  class ProspectusConfig(BaseModel):
    class Config:
      arbitrary_types_allowed = True
    neo4j: Neo4jGraph
    tokenizer: PreTrainedTokenizerFast
    llm: HuggingFaceEndpoint

  class ProspectusTool(BaseTool):
    name = "招股说明书"
    description = '当你有招股说明书相关问题，可以调用这个工具'
    args_schema: Type[BaseModel] = ProspectusInput
    return_direct: bool = True
    config: ProspectusConfig
    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
      # 1) extract entities of known entity types
      results = self.config.neo4j.query("match (n) return distinct labels(n)")
      entity_types = [result['labels(n)'][0] for result in results]
      prompt = entity_generation_template(self.config.tokenizer, entity_types)
      chain = prompt | self.config.llm
      entities = chain.invoke({'question': query})
      entities = eval(entities)
      print('extracted entityes:', entities)
      # 2) search for triplets related to these triplets
      triplets = list()
      for entity_type, keywords in entities.items():
        if len(keywords) == 0: continue
        for keyword in keywords:
          #cypher_cmd = 'match (a:`%s`)-[r]->(b) where tolower(a.id) contains tolower(\'%s\') return a,r,b' % (entity_type, keyword)
          cypher_cmd = 'match (a)-[r]->(b) where tolower(a.id) contains tolower(\'%s\') return a,r,b' % (keyword)
          matches = self.config.neo4j.query(cypher_cmd)
          triplets.extend([(match['a']['id'],match['r'][1],match['b']['id']) for match in matches])
          #cypher_cmd = 'match (b)-[r]->(a:`%s`) where tolower(a.id) contains tolower(\'%s\') return b,r,a' % (entity_type, keyword)
          cypher_cmd = 'match (b)-[r]->(a) where tolower(a.id) contains tolower(\'%s\') return b,r,a' % (keyword)
          matches = self.config.neo4j.query(cypher_cmd)
          triplets.extend([(match['b']['id'],match['r'][1],match['a']['id']) for match in matches])
      print('matched triplets:', triplets)
      # 3) ask llm for answer according to matched triplets
      prompt = triplets_qa_template(self.config.tokenizer, triplets)
      chain = prompt | self.config.llm
      answer = chain.invoke({'question': query})
      return answer
    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
      raise NotImplementedError("Prospectus does not suppert async!")

  environ['HUGGINGFACEHUB_API_TOKEN'] = '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'
  neo4j = Neo4jGraph(url = host, username = username, password = password, database = database)
  tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct')
  llm = HuggingFaceEndpoint(
    endpoint_url = "meta-llama/Meta-Llama-3-8B-Instruct",
    task = "text-generation",
    max_length = 16384,
    do_sample = False,
    temperature = 0.6,
    top_p = 0.9,
    eos_token_id = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")],
    use_cache = True,
  )
  return ProspectusTool(config = ProspectusConfig(neo4j = neo4j, tokenizer = tokenizer, llm = llm))

def load_database(sqlite_path):
  class DatabaseInput(BaseModel):
    query: str = Field(description = "需要询问的金融问题")

  class DatabaseConfig(BaseModel):
    class Config:
      arbitrary_types_allowed = True
    db: SQLDatabase
    tokenizer: PreTrainedTokenizerFast
    llm: HuggingFaceEndpoint

  class DatabaseTool(BaseTool):
    name = "金融数据查询工具"
    description = '当你有基金基本信息，基金股票持仓明细，基金债券持仓明细，基金可转债持仓明细，基金日行情表，A股票日行情表，港股票日行情表，A股公司行业划分表，基金规模变动表，基金份额持有人结构，相关问题可以调用这个工具'
    args_schema: Type[BaseModel] = DatabaseInput
    return_direct: bool = True
    config: DatabaseConfig
    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
      prompt = sqlite_prompt(self.config.tokenizer)
      chain = SQLDatabaseChain.from_llm(llm = self.config.llm, db = self.config.db, prompt = prompt)
      return chain.run(query)
    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
      raise NotImplementedError("DatabaseTool does not support async!")

  environ['HUGGINGFACEHUB_API_TOKEN'] = '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'
  tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct')
  llm = HuggingFaceEndpoint(
    endpoint_url = "meta-llama/Meta-Llama-3-8B-Instruct",
    task = "text-generation",
    max_length = 16384,
    do_sample = False,
    temperature = 0.6,
    top_p = 0.9,
    eos_token_id = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")],
    use_cache = True,
  )
  db = SQLDatabase.from_uri('sqlite:///%s' % sqlite_path)
  return DatabaseTool(config = DatabaseConfig(db = db, tokenizer = tokenizer, llm = llm))

if __name__ == "__main__":
  '''
  # 1) test knowledge graph
  kb = load_knowledge_graph(password = '19841124')
  print('name:',kb.name)
  print('description:', kb.description)
  print('args:',kb.args)
  res = kb.invoke({'query': 'what is the application of sodium chloride?'})
  print(res)
  # NOTE: https://github.com/langchain-ai/langchain/discussions/15927
  kb.config.neo4j._driver.close()
  '''
  # 2) test sql base
  db = load_database('bs_challenge_financial_14b_dataset/dataset/博金杯比赛数据.db')
  print('name:', db.name)
  print('description:', db.description)
  print('args:', db.args)
  res = db.invoke({'query': '请查询在2021年度，688338股票涨停天数？'})
  print(res)
