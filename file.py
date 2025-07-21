from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()


### Indexing - Document Ingestion

video_id = "Gfr50f6ZBvo"
try:
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
    transcripts = " ".join([t["text"] for t in transcript_list])
    # print(transcripts)

except TranscriptsDisabled:
    print("Transcripts are disabled for this video.")


### Indexing - Text Splitting


splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = splitter.create_documents([transcripts])
# print(f"No. of documents: {len(texts)}")


### Indexing - Embedding Geration and Vector Store 

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_store = FAISS.from_documents(texts, embeddings)
# print(vector_store.index_to_docstore_id)


### Retrieval 

retriever = vector_store.as_retriever(search_type = "similarity",search_kwargs={"k": 4})
# print(retriver.invoke("What is the main topic of the video?"))

### Augmentation - create a prompt (query + retrieved context)

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

question = "is the topic of aliens discussed in the video? if yes, what is the main point?"
retrieved_text = retriever.invoke(question)

context = "\n\n".join([doc.page_content for doc in retrieved_text])
final_prompt = prompt.invoke({"context": context, "question": question})

### Generation 

model = GoogleGenerativeAI(model="gemini-2.0-flash")
response = model.invoke(final_prompt)
# print(response)




### Using the Chain

from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

def format_docs(retrieved_text):
    return "\n\n".join([doc.page_content for doc in retrieved_text])

parallel_chain = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})

parser = StrOutputParser()

main_chain = parallel_chain | prompt | model | parser

result = main_chain.invoke("can you summarize the video?")
print(result)


