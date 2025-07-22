from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
import nest_asyncio
from dotenv import load_dotenv
import os
import re
import streamlit as st

load_dotenv()
nest_asyncio.apply()

def extract_video_id(youtube_url: str) -> str | None:
    patterns = [
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=([a-zA-Z0-9_-]{11})',
        r'(?:https?:\/\/)?youtu\.be\/([a-zA-Z0-9_-]{11})',
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/embed\/([a-zA-Z0-9_-]{11})'
        ]
    for pattern in patterns:
        match = re.search(pattern, youtube_url)
        if match:
            return match.group(1)
        return None


### Data - YouTube Transcript
def get_transcript(video_id: str) -> str | None:
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        try:
            transcript = transcript_list.find_manually_created_transcript(['en'])
        except NoTranscriptFound:
            try:
                transcript = transcript_list.find_generated_transcript(['en'])
            except NoTranscriptFound:
                return None
        return " ".join([t.text for t in transcript.fetch()])
    except TranscriptsDisabled:
            st.error("Transcripts are disabled for this video.")
            return None
    except Exception as e:
            st.error(f"An error occurred: {e}")
            return None


### Vector Store

@st.cache_resource(show_spinner="Embedding video content...")
def create_vector_store(_transcript: str):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = splitter.create_documents([_transcript])
        vector_store = FAISS.from_documents(texts, embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Failed to create vector store: {e}")
        return None

### Retrieval and Response Generation

def get_response(question: str, _vector_store, chat_history: list):
    try:
        retriever = _vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        model = GoogleGenerativeAI(model="gemini-1.5-flash-latest")
        parser = StrOutputParser()
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant. Answer the user's questions based on the context provided below. If you don't know the answer,
             just say that you don't know.\n\nContext:\n{context}"""),
             *chat_history,
             ("human", "{question}"),
             ])
        def format_docs(retrieved_text):
            return "\n\n".join([doc.page_content for doc in retrieved_text])
        
        chain = RunnableParallel({
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
            "chat_history": lambda x: chat_history
            }) | prompt | model | parser
        
        result = chain.invoke(question)
        return result
    
    except Exception as e:
        
        st.error(f"Failed to get response: {e}")
        return "An error occurred while processing your request."
    


### Streamlit app 
st.set_page_config(page_title="VidQuery", layout="centered")
st.title("VidQuery - Ask your Questions")

input_container = st.container()
with input_container:
    youtube_url = st.text_input("Enter YouTube Video URL:", placeholder="https://www.youtube.com/watch?v=...")
    
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'processed_url' not in st.session_state:
    st.session_state.processed_url = ""


if youtube_url and st.session_state.processed_url != youtube_url:
    video_id = extract_video_id(youtube_url)
    if video_id:
        with st.spinner("Fetching transcript..."):
            transcript = get_transcript(video_id)
        if transcript:
            st.success("Transcript fetched successfully!")
            st.session_state.vector_store = create_vector_store(transcript)
            st.session_state.chat_history = [] 
            st.session_state.processed_url = youtube_url
            st.rerun() 
        else:
            st.session_state.vector_store = None
            st.session_state.processed_url = ""
    else:
        st.error("Invalid YouTube URL. Please enter a valid URL.")
        st.session_state.vector_store = None
        st.session_state.processed_url = ""

if st.session_state.vector_store:
    
    if st.button("Clear"):
        st.session_state.chat_history = []
        st.rerun() 

    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            role = "Human" if isinstance(message, HumanMessage) else "AI"
            with st.chat_message(role):
                st.write(message.content)
    
    user_query = st.chat_input("Ask your query about the video...")
    if user_query:
        st.session_state.chat_history.append(HumanMessage(user_query))
        with st.chat_message("Human"):
            st.write(user_query)
        with st.chat_message("AI"):
            with st.spinner("Generating response..."):
                response = get_response(user_query, st.session_state.vector_store, st.session_state.chat_history)
                st.write(response)
                st.session_state.chat_history.append(AIMessage(response))

else:
    st.info("Welcome to VidQuery! Please enter a YouTube URL above to get started.")