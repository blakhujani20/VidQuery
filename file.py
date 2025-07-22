from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
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

@st.cache_resource(show_spinner="Generating response...")
def get_response(question: str, _vector_store) -> str:
    try:
        retriever = _vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        model = GoogleGenerativeAI(model="gemini-1.5-flash-latest")
        parser = StrOutputParser()
        prompt = PromptTemplate(
            template="""
            You are a helpful assistant. Use the context below to answer the question.
            if the context does not contain enough information, say "I don't know".
            
            Context:
            {context}           
            Question:
            {question}
            """,
            input_variables=["context", "question"]
        )
        
        def format_docs(retrieved_text):
            return "\n\n".join([doc.page_content for doc in retrieved_text])
        
        chain = RunnableParallel({
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        }) | prompt | model | parser

        result = chain.invoke(question)
        return result
    except Exception as e:
        st.error(f"Failed to get response: {e}")
        return "An error occurred while processing your request."
   
    
### Streamlit app 
st.set_page_config(page_title="VidQuery", page_icon="▶️", layout="centered")
st.title("VidQuery - Ask your Questions")

youtube_url = st.text_input("Enter YouTube Video URL:", placeholder="https://www.youtube.com/watch?v=VIDEO_ID")
if youtube_url:
    video_id = extract_video_id(youtube_url)
    if video_id:
        transcript = get_transcript(video_id)
        if transcript:
            st.write("Transcript fetched successfully.")
            vector_store = create_vector_store(transcript)
            if vector_store:
                question = st.text_input("Ask your query")
                if question:
                    response = get_response(question, vector_store)
                    st.write("Response:", response)
        else:
            st.error("No transcript available for this video.")
    else:
        st.error("Invalid YouTube URL. Please enter a valid URL.")
