#!/usr/bin/python3

huggingface_token = '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'

#'''
# aura db
neo4j_host = "neo4j+s://ce5c573a.databases.neo4j.io:7687"
neo4j_username = "neo4j"
neo4j_password = "RdKzW-YnyyRd5h1Q6OoRQUDza5edKXrmqDzSTN7QkZg"
neo4j_db = "neo4j"
#'''
'''
# local db
neo4j_host = "bolt://147.8.234.16:7687"
neo4j_username = "neo4j"
neo4j_password = "neo4j"
neo4j_db = "experiments"
#'''

unstructure_method = "RAG" # "RAG" or "KG"

run_locally = False

service_host = "0.0.0.0"
service_port = 8081
