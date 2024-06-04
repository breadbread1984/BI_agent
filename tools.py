#!/usr/bin/python3

from os import environ, remove
from os.path import exists, join, isdir
from typing import Optional, Type, List, Dict, Union, Any
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from langchain.graphs import Neo4jGraph
from langchain.vectorstores import Neo4jVector
from langchain.memory import ChatMessageHistory
from langchain.schema import AIMessage, HumanMessage
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceEndpoint, HuggingFacePipeline
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from prompts import entity_generation_template, triplets_qa_template, sqlite_prompt
from models import Llama3, CodeLlama

def load_vectordb(host = "bolt://localhost:7687", username = "neo4j", password = None, database = 'neo4j', locally = False):
  class ProspectusInput(BaseModel):
    query: str = Field(description = "招股说明书相关的问题")

  class ProspectusConfig(BaseModel):
    class Config:
      arbitrary_types_allowed = True
    neo4j: Neo4jGraph
    retriever: VectorStoreRetriever

  class ProspectusTool(BaseModel):
    name = "招股说明书"
    description = "当你有招股说明书相关问题，可以调用这个工具"
    args_schema: Type[BaseModel] = ProspectusInput
    return_direct: bool = True
    config: ProspectusConfig
    def get_vector_history(self, input: Dict[str, Any]) -> List[Union[HumanMessage, AIMessage]]:
      window = 3
      data = self.neo4j.query("""
          MATCH (u:User {id:$user_id})-[:HAS_SESSION]->(s:Session {id:$session_id}),
              (s)-[:LAST_MESSAGE]->(last_message)
          MATCH p=(last_message)<-[:NEXT*0..%d]-()
          WITH p, length(p) AS length
          ORDER BY length DESC LIMIT 1
          UNWIND reverse(nodes(p)) AS node
          MATCH (node)-[:HAS_ANSWER]->(answer)
          RETURN {question:node.text, answer:answer.text} AS result
      """ % window)
    def _run(self, query:str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
      # TODO

  neo4j = Neo4jGraph(url = host, username = username, password = password, database = database)
  vectordb = Neo4jVector(
    embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
    url = host, username = username, password = password, database = database,
    index_name = "typical_rag"
  )
  parent_vectordb = Neo4jVector(
    embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
    url = host, username = username, password = password, database = database,
    retrieval_query = """
    MATCH (node)<-[:HAS_CHILD]-(parent)
    WITH parent, max(score) AS score // deduplicate parents
    RETURN parent.text AS text, score, {} AS metadata LIMIT 1
    """,
    index_name = "parent_document"
  )
  hypothetic_question_vectordb = Neo4jVector(
    embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
    url = host, username = username, password = password, database = database,
    retrieval_query = """
    MATCH (node)<-[:HAS_QUESTION]-(parent)
    WITH parent, max(score) AS score // deduplicate parents
    RETURN parent.text AS text, score, {} AS metadata
    """,
    index_name = "hypothetic_question_query"
  )
  summary_vectordb = Neo4jVector(
    embedding = HuggingFaceEmebeddings(model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
    url = host, username = username, password = password, database = database,
    retrieval_query = """
    MATCH (node)<-[:HAS_SUMMARY]-(parent)
    WITH parent, max(score) AS score // deduplicate parents
    RETURN parent.text AS text, score, {} AS metadata
    """,
    index_name = "summary"
  )

  return ProspectusTool(config = ProspectusConfig(
    neo4j = neo4j,
    retriever = vectordb.as_retriever().configurable_alternatives(
      ConfigurableField(id = "strategy"),
      default_key = "typical_rag",
      parent_strategy = parent_vectordb.as_triever(),
      hypothetical_questions = hypothetic_question_vectordb.as_triever(),
      summary_strategy = summary_vectordb.as_triever()
    )
  ))

def load_knowledge_graph(host = 'bolt://localhost:7687', username = 'neo4j', password = None, database = 'neo4j', locally = False):
  class ProspectusInput(BaseModel):
    query: str = Field(description = "招股说明书相关的问题")

  class ProspectusConfig(BaseModel):
    class Config:
      arbitrary_types_allowed = True
    neo4j: Neo4jGraph
    tokenizer: PreTrainedTokenizerFast
    llm: HuggingFaceEndpoint if not locally else HuggingFacePipeline

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

  neo4j = Neo4jGraph(url = host, username = username, password = password, database = database)
  tokenizer, llm = Llama3(locally = locally)
  return ProspectusTool(config = ProspectusConfig(neo4j = neo4j, tokenizer = tokenizer, llm = llm))

def load_database(sqlite_path, locally = False):
  class DatabaseInput(BaseModel):
    query: str = Field(description = "需要询问的金融问题")

  class DatabaseConfig(BaseModel):
    class Config:
      arbitrary_types_allowed = True
    db: SQLDatabase
    tokenizer: PreTrainedTokenizerFast
    llm: HuggingFaceEndpoint if not locally else HuggingFacePipeline

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

  tokenizer, llm = Llama3(locally)
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
  db = load_database('bs_challenge_financial_14b_dataset/dataset/博金杯比赛数据.db', locally = True)
  print('name:', db.name)
  print('description:', db.description)
  print('args:', db.args)
  res = db.invoke({'query': '请查询在2021年度，688338股票涨停天数？'})
  print(res)
