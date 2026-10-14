import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()
GROQ_API_KEY = os.environ.get("LLM_API_KEY")
groq_model_name = os.environ.get("LLM_MODEL")
DB_FAISS_PATH="vectorstore/db_faiss"

#load vector store FAISS in cache (loads once, reused on every rerun)
@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )
    return prompt

def load_llm(groq_model_name):
    llm = ChatGroq(
        model=groq_model_name,
        temperature=0.7,
        api_key=GROQ_API_KEY,
        max_tokens=512
    )
    return llm

def main():
    st.title("Ask Chatbot!")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt=st.chat_input("Ask your question here...")

    if prompt:

        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        custom_prompt_template = """You are a helpful assistant that answers questions based on the context provided. If you don't know the answer, just say "I don't know". Do not try to make up an answer. Use the following context to answer the question at the end.
Context: {context}
Question: {question}
Answer:"""

        try:
            vectorstore = get_vectorstore()

            #retrieval augmented generation: fetch top 3 chunks from FAISS, then LLM answers from them
            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(groq_model_name),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True,
                chain_type_kwargs={"prompt": set_custom_prompt(custom_prompt_template)}
            )

            with st.spinner("Thinking..."):
                response = qa_chain.invoke({"query": prompt})

            result = response["result"]
            source_documents = response["source_documents"]

            sources = []
            for doc in source_documents:
                source = f"{os.path.basename(doc.metadata.get('source', ''))} (page {doc.metadata.get('page_label', '?')})"
                if source not in sources:
                    sources.append(source)

            result_to_show = result + "\n\n**Sources:** " + ", ".join(sources)
            st.chat_message('assistant').markdown(result_to_show)
            st.session_state.messages.append({"role": "assistant", "content": result_to_show})

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
