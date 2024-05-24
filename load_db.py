#!/usr/bin/python3

from absl import flags, app
from tqdm import tqdm
from os import walk
from os.path import splitext, join
from langchain_community.document_loaders import UnstructuredPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.graphs import Neo4jGraph
from langchain_experimental.graph_transformers.llm import LLMGraphTransformer
from prompts import extract_triplets_template
from models import Llama3

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('doc_dir', default = None, help = 'path to directory containing documents')
  flags.DEFINE_string('host', default = 'localhost', help = 'host of neo4j')
  flags.DEFINE_integer('port', default = 7687, help = 'port number')
  flags.DEFINE_string('user', default = 'neo4j', help = 'username of neo4j')
  flags.DEFINE_string('password', default = None, help = 'password of neo4j')
  flags.DEFINE_string('db', default = 'neo4j', help = 'database of neo4j')
  flags.DEFINE_boolean('run_locally', default = False, help = 'whether the LLM is runned locally')

def main(unused_argv):
  tokenizer, llm = Llama3(FLAGS.run_locally)
  neo4j = Neo4jGraph(url = 'bolt://%s:%d' % (FLAGS.host, FLAGS.port), username = FLAGS.user, password = FLAGS.password, database = FLAGS.db)
  # 1) load text into list
  docs = list()
  for root, dirs, files in tqdm(walk(FLAGS.doc_dir)):
    for f in files:
      stem, ext = splitext(f)
      loader_types = {'.md': UnstructuredMarkdownLoader,
                      '.txt': TextLoader,
                      '.pdf': UnstructuredPDFLoader}
      loader = loader_types[ext](join(root, f), mode = "single", strategy = "fast")
      docs.extend(loader.load())
  # 2) split pages into chunks and save to split_docs
  print('split pages into chunks')
  text_splitter = RecursiveCharacterTextSplitter(chunk_size = 200, chunk_overlap = 50)
  split_docs = text_splitter.split_documents(docs)
  # 3) erase content of neo4j
  neo4j.query('match (a)-[r]-(b) delete a,r,b')
  # 4) extract triplets from documents
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

