import streamlit as st              # for GUI
from dotenv import load_dotenv      # for API key
from PyPDF2 import PdfReader        # for PDF upload
from langchain.text_splitter import CharacterTextSplitter       # for dividing text into chunks
from langchain.embeddings import HuggingFaceInstructEmbeddings        #import our embedding model##
from langchain.vectorstores import FAISS      #store our Embeddings##
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub    # importing llm model
from htmlTemplate import css, bot_template, user_template


#Function to ectract pdf text 
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text


#function to divide Text into chunksstream
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size = 500,
        chunk_overlap = 100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


#functions to create embeddings using hugging face
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name = "hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts = text_chunks, embedding=embeddings) 
    return vectorstore


# function to create question-answer chain
def get_conversation_chain(vectorstore):
    llm =HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever = vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


# function to handle user input
def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i%2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}",message.content), unsafe_allow_html=True)





def main():
    #load the api key
    load_dotenv()
    #set-up GUI
    st.set_page_config(page_title="RAG",page_icon=":computer:")

    st.write(css, unsafe_allow_html=True)
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None


    st.header("Retrieval Augmented Generation  :computer:")
    user_question = st.text_input("Ask your question : ")
    if user_question:
        handle_userinput(user_question)



    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Upload your PDFs",accept_multiple_files=True)
        if st.button("Upload"):
            with st.spinner("Uploading"):
                #  Get PDF text
                raw_text = get_pdf_text(pdf_docs)

                # Divide the text chunks 
                text_chunks = get_text_chunks(raw_text)

                # Create the vector store
                vectorstore = get_vectorstore(text_chunks)

                # create coversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)

                






# Calling the main Function
if __name__ == '__main__':
    main()