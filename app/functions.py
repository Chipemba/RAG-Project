
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from tempfile import NamedTemporaryFile

import base64
import os
import tempfile
import uuid
import pandas as pd
import re
import json
import random
import time
import requests
import streamlit as st


urls = [
        "https://www.jdpower.com/motorcycles/shopping-guides/beginner-dirt-bike-for-adults",
        "https://www.riderswestmag.com/motomayhem/article/your_first_dirt_bike_buy_the_reality_not_the_dream",
        "https://www.frpmoto.com/blogs/dirt-bike-specifications/understanding-dirt-bike-specifications-a-beginners-guide",
        "https://blog.amsoil.com/what-to-look-for-when-buying-a-used-dirt-bike/",
        "https://ca.riskracing.com/blogs/news/what-do-i-need-to-know-before-buying-a-dirt-bike?srsltid=AfmBOoporUN6YY4RcrcyPV3MmkIRgy8OMe61-yxN3kj4GzyTzQbMaKfx&shpxid=a3a2c14b-3544-4bbe-98d2-b8dcb508d67f"
        ]



def clean_filename(filename):
    """
    Remove "(number)" pattern from a filename 
    (because this could cause error when used as collection name when creating Chroma database).

    Parameters:
        filename (str): The filename to clean

    Returns:
        str: The cleaned filename
    """
    # Regular expression to find "(number)" pattern
    new_filename = re.sub(r'\s\(\d+\)', '', filename)
    return new_filename



def get_pdf_text(): 

    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--print-to-pdf-no-header')  # optional

    # Enable DevTools Protocol
    chrome_prefs = {
        'printing.print_preview_sticky_settings.appState': '{"recentDestinations":[{"id":"Save as PDF","origin":"local"}],"selectedDestinationId":"Save as PDF","version":2}'
    }
    chrome_options.add_experimental_option('prefs', chrome_prefs)
    chrome_options.add_argument('--kiosk-printing')

    driver = webdriver.Chrome(options=chrome_options)
    documents=[]
    temps=[]
    i=0
    for url in urls: 
        # random.choices(urls, k=5):
        
        try:
            
            driver.get(url)

            pdf = driver.execute_cdp_cmd("Page.printToPDF", {"landscape": False,"printBackground": True})
            pdf_bytes = base64.b64decode(pdf['data'])

            pdf_base64 = base64.b64encode(pdf_bytes).decode('utf-8')
     
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
                temp_pdf.write(pdf_bytes)
                temp_pdf_path = temp_pdf.name   
                loader = PyPDFLoader(temp_pdf_path )
                documents.extend(loader.load())
    
        except Exception as e:
            print(f"Failed to save {url}: {e}")
        finally:
            os.unlink(temp_pdf.name)
        
        i=i+1

    driver.quit()
   
    return documents



def split_document(documents, chunk_size, chunk_overlap):    
    """
    Function to split generic text into smaller chunks.
    chunk_size: The desired maximum size of each chunk (default: 400)
    chunk_overlap: The number of characters to overlap between consecutive chunks (default: 20).

    Returns:
        list: A list of smaller text chunks created from the generic text
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                          chunk_overlap=chunk_overlap,
                                          length_function=len,
                                          separators=["\n\n", "\n", " "])
    
    return text_splitter.split_documents(documents)



def get_embedding_function(api_key):
    """
    Return an OpenAIEmbeddings object, which is used to create vector embeddings from text.
    The embeddings model used is "text-embedding-ada-002" and the OpenAI API key is provided
    as an argument to the function.

    Parameters:
        api_key (str): The OpenAI API key to use when calling the OpenAI Embeddings API.

    Returns:
        OpenAIEmbeddings: An OpenAIEmbeddings object, which can be used to create vector embeddings from text.
    """
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002", openai_api_key=api_key
    )
    return embeddings



def create_vectorstore(chunks, embedding_function, file_name, vector_store_path="db"):

    """
    Create a vector store from a list of text chunks.

    :param chunks: A list of generic text chunks
    :param embedding_function: A function that takes a string and returns a vector
    :param file_name: The name of the file to associate with the vector store
    :param vector_store_path: The directory to store the vector store

    :return: A Chroma vector store object
    """

    # Create a list of unique ids for each document based on the content
    ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.page_content)) for doc in chunks]
    
    # Ensure that only unique docs with unique ids are kept
    unique_ids = set()
    unique_chunks = []
    
    unique_chunks = [] 
    for chunk, id in zip(chunks, ids):     
        if id not in unique_ids:       
            unique_ids.add(id)
            unique_chunks.append(chunk)        

    # Create a new Chroma database from the documents
    vectorstore = Chroma.from_documents(documents=unique_chunks, 
                                        collection_name=clean_filename(file_name),
                                        embedding=embedding_function, 
                                        ids=list(unique_ids), 
                                        persist_directory = vector_store_path)

    # The database should save automatically after we create it
    # but we can also force it to save using the persist() method
    vectorstore.persist()
    
    return vectorstore




def create_vectorstore_from_texts(documents, api_key, filename):
    
    # Step 2 split the documents  
    """
    Create a vector store from a list of texts.

    :param documents: A list of generic text documents
    :param api_key: The OpenAI API key used to create the vector store
    :param file_name: The name of the file to associate with the vector store

    :return: A Chroma vector store object
    """
    docs = split_document(documents, chunk_size=1000, chunk_overlap=200)
    
    # Step 3 define embedding function
    embedding_function = get_embedding_function(api_key)

    # Step 4 create a vector store  
    vectorstore = create_vectorstore(docs, embedding_function, filename)
    
    return vectorstore




def load_vectorstore(file_name, api_key, vectorstore_path="db"):

    """
    Load a previously saved Chroma vector store from disk.

    :param file_name: The name of the file to load (without the path)
    :param api_key: The OpenAI API key used to create the vector store
    :param vectorstore_path: The path to the directory where the vector store was saved (default: "db")
    
    :return: A Chroma vector store object
    """
    embedding_function = get_embedding_function(api_key)
    return Chroma(persist_directory=vectorstore_path, 
                  embedding_function=embedding_function, 
                  collection_name=clean_filename(file_name))

# Prompt template
PROMPT_TEMPLATE = """
You are an chatbot assisting with answering frequently asked 
question for a website that sales outdoor vehicles.
Your main focus is only on two wheel and four wheel motorcycle bikes.
Use the following pieces of retrieved context to answer
the question. If you don't know the answer, say that you
don't know. DON'T MAKE UP ANYTHING.
{context}
---
Answer the question based on the above context: {question}
"""

class AnswerWithSources(BaseModel):
    """An answer to the question, with sources and reasoning."""
    answer: str = Field(description="Answer to question")
    sources: str = Field(description="Full direct text chunk from the context used to answer the question")
    reasoning: str = Field(description="Explain the reasoning of the answer based on the sources")
    

class ExtractedInfoWithSources(BaseModel):
    """Extracted information about the research article"""
    paper_title: AnswerWithSources
    paper_summary: AnswerWithSources
    publication_year: AnswerWithSources
    paper_authors: AnswerWithSources



def format_docs(docs):
    """
    Format a list of Document objects into a single string.

    :param docs: A list of Document objects

    :return: A string containing the text of all the documents joined by two newlines
    """
    return "\n\n".join(doc.page_content for doc in docs)

# retriever | format_docs passes the question through the retriever, generating Document objects, and then to format_docs to generate strings;
# RunnablePassthrough() passes through the input question unchanged.


def query_document(vectorstore, query, api_key):

    """
    Query a vector store with a question and return a structured response.

    :param vectorstore: A Chroma vector store object
    :param query: The question to ask the vector store
    :param api_key: The OpenAI API key to use when calling the OpenAI Embeddings API

    :return: A pandas DataFrame with three rows: 'answer', 'source', and 'reasoning'
    """
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)

    retriever=vectorstore.as_retriever(search_type="similarity")

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt_template
            | llm.with_structured_output(ExtractedInfoWithSources, strict=True)
        )

    structured_response = rag_chain.invoke(query)
    df = pd.DataFrame([structured_response.dict()])

    # Transforming into a table with two rows: 'answer' and 'source'
    answer_row = []
    source_row = []
    reasoning_row = []

    for col in df.columns:
        answer_row.append(df[col][0]['answer'])
        source_row.append(df[col][0]['sources'])
        reasoning_row.append(df[col][0]['reasoning'])

    # Create new dataframe with two rows: 'answer' and 'source'
    structured_response_df = pd.DataFrame([answer_row, source_row, reasoning_row], columns=df.columns, index=['answer', 'source', 'reasoning'])
  
    return structured_response_df.T