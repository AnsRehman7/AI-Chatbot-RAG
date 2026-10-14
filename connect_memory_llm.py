#setup llm with groq
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()
GROQ_API_KEY = os.environ.get("LLM_API_KEY")
groq_model_name = os.environ.get("LLM_MODEL")

def load_llm(groq_model_name):
    llm = ChatGroq(
        model=groq_model_name,
        temperature=0.7,
        api_key=GROQ_API_KEY,
        max_tokens=512
    )
    return llm

# connect llm with FAISS and create chain
DB_FAISS_PATH="vectorstore/db_faiss"
custom_prompt_template = """You are a helpful assistant that answers questions based on the context provided. If you don't know the answer, just say "I don't know". Do not try to make up an answer. Use the following context to answer the question at the end.
Context: {context}
Question: {question}
Answer:"""

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )
    return prompt

#load DB
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

#create a QA Chain

qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(groq_model_name),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": set_custom_prompt(custom_prompt_template)}
)

#invoke with a single query

user_query = input("Enter your question: ")
response = qa_chain.invoke({"query": user_query})
print("Result:", response["result"])
print("Source Documents:", response["source_documents"])