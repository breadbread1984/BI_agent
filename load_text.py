#!/usr/bin/python3

from absl import flags, app
from tqdm import tqdm
from os import walk
from os.path import splitext, join
from langchain_community.document_loaders import UnstructuredPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.graphs import Neo4jGraph
from langchain.vectorstores import Neo4jVector
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_experimental.graph_transformers.llm import LLMGraphTransformer
from prompts import extract_triplets_template
from models import Llama3
import config

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('doc_dir', default = None, help = 'path to directory containing documents')

def main(unused_argv):
  tokenizer, llm = Llama3(config.run_locally)
  if config.unstructure_method == 'KG':
    neo4j = Neo4jGraph(url = config.neo4j_host, username = config.neo4j_username, password = config.neo4j_password, database = config.neo4j_db)
  # 1) load text into list
  docs = list()
  for root, dirs, files in tqdm(walk(FLAGS.doc_dir)):
    for f in files:
      stem, ext = splitext(f)
      ext = ext.lower()
      loader_types = {'.md': UnstructuredMarkdownLoader,
                      '.txt': TextLoader,
                      '.pdf': UnstructuredPDFLoader}
      if ext != '.txt':
        loader = loader_types[ext](join(root, f), mode = "single", strategy = "fast")
      else:
        loader = loader_types[ext](join(root, f))
      docs.extend(loader.load())
  # 2) split pages into chunks and save to split_docs
  print('split pages into chunks')
  text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500 if config.unstructure_method == 'RAG' else 50, chunk_overlap = 150 if config.unstructure_method == 'RAG' else 25)
  split_docs = text_splitter.split_documents(docs)
  # 3) erase content of neo4j
  neo4j.query('match (a)-[r]-(b) delete a,r,b')
  # 4) extract triplets from documents
  if config.unstructure_method == 'RAG':
    print('extract embedding from document trunks')
    embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vectordb = Neo4jVector.from_documents(
        documents = split_docs,
        embedding = embedding,
        url = config.neo4j_host,
        username = config.neo4j_username,
        password = config.neo4j_password,
        database = config.neo4j_db,
        index_name = "typical_rag",
        search_type = "hybrid"
    )
  elif config.unstructure_method == 'KG':
    print('extract triplets from documents')
    prompt, _ = extract_triplets_template(tokenizer)
    graph = LLMGraphTransformer(
        llm = llm,
        prompt = prompt
    ).convert_to_graph_documents(split_docs)
    neo4j.add_graph_documents(graph)

if __name__ == "__main__":
  add_options()
  app.run(main)

