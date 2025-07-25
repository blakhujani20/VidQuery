# VidQuery 
## Ask questions and get instant answers from any YouTube video.

**VidQuery** is a Streamlit web application that uses Google's **Gemini** models to let you chat with any YouTube video. Just paste a video URL — the app fetches the transcript and gives you context-aware, conversational responses.

## Key Features

**Conversational AI**: Chat with a YouTube video as if you were talking to a person.

**Transcript-Powered**: Automatically fetches the English transcript of the video.

**Context-Aware Memory**: Remembers the chat history for follow-up questions.

**Sleek UI**: A clean and intuitive user interface built with Streamlit.

**Easy to Use**: Just paste a URL and start asking!


## Technologies Used

**Backend**: Python

**Web Framework**: Streamlit

**LLM & Embeddings**: Google Generative AI (Gemini 1.5 Flash)

**Core Libraries**: LangChain, FAISS (for vector storage), youtube_transcript_api


## Setup and Installation
To run this project locally

1. Clone the repository

2. Create and activate a virtual environment

3. Install the required dependencies by running pip install -r requirement.txt

4. Set up your environment variables in .env file

5. Run the Streamlit application using streamlit run file.py


The application should now be running in your browser!