import streamlit as st
import tempfile
import os
import rag
import openai
from dotenv import load_dotenv
import pandas as pd
import json
import torch
from transformers import pipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.llms import OpenAI
import openai
from pinecone import Pinecone, ServerlessSpec

from langchain.vectorstores.base import VectorStoreRetriever
from langchain.schema import Document
from pydantic import Field
from typing import List

from langchain import PromptTemplate, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Pinecone as PineconeVectorStore
from langchain.chains import RetrievalQA
import numpy as np

from groq import Groq
from langchain_groq import ChatGroq
import datetime

st.set_page_config(layout="wide")

# Function to format the current time
def format_time():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Title of the app
# st.title('RAG Based VQA')
st.markdown('<h1 class="center-title">RAG Based VQA</h1>', unsafe_allow_html=True)
css = '''
<style>
    .scrollable-column {
        max-height: 70vh;
        overflow-y: auto;
        padding-right: 10px;
    }
    .scrollable-column::-webkit-scrollbar {
        width: 8px;
    }
    .scrollable-column::-webkit-scrollbar-thumb {
        background-color: #888;
        border-radius: 10px;
    }
    .scrollable-column::-webkit-scrollbar-thumb:hover {
        background-color: #555;
    }
</style>
'''
st.markdown(css, unsafe_allow_html=True)

left_col, right_col = st.columns(2)
# left_col, right_col = st.columns([1, 2]) 

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
# API_TOKEN = os.getenv("HF_API_TOKEN")

# openai.api_key = API_TOKEN
# openai.api_key = "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"
# openai.api_key = "sk-proj-GZ2uGXjtGqjgX21R4YIeT3BlbkFJUAzrqyOydJHTZzZWCKyB"

# client = Groq(
#     api_key=os.environ.get("GROQ_API_KEY"),
# )

pc_api_key = os.getenv("PINECONE_API_KEY ")
pc = Pinecone(api_key=pc_api_key)


print("Models loading")
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe_whisper = pipeline("automatic-speech-recognition", model="openai/whisper-base", device=device)
print("Whisper loaded")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print("Embeddings Models loaded")

# openai_llm = OpenAI(
#     api_key=openai.api_key, 
#     model_name="gpt-3.5-turbo-instruct",  
#     max_tokens=100,  
#     temperature=0.7,  
# )
# print("Openai_llm loaded")

llm = ChatGroq(
    model="mixtral-8x7b-32768",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key= groq_api_key,
)
print("GROQ LLM loaded")


with left_col:
    # File uploader for the video
    uploaded_video = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

    class CustomRetriever(VectorStoreRetriever):
        vectorstore: VectorStoreRetriever
        search_type: str = "similarity"
        search_kwargs: dict = Field(default_factory=dict)

        def get_relevant_documents(self, query: str) -> List[Document]:
            # Retrieve documents from the vdb
            results = self.vectorstore.get_relevant_documents(query=query)

            for r in results:
                print(r)

            return results

    if uploaded_video is not None:
        st.video(uploaded_video)
        st.success("Video uploaded and its processing now!")

        # Save the uploaded video to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            temp_file.write(uploaded_video.read())
            temp_video_path = temp_file.name

        # Process the uploaded video file
        transcription_df = rag.main(temp_video_path, pipe_whisper)
        print("transcription_df done")


    else:
        st.write("Please upload a video file to proceed.")

###############################################################################################

with right_col:
    # st.header("Right Column")
    with open('final_transcription.json', 'r') as f:
        transcription_data = json.load(f)

    documents_df = rag.concatenate_sentences(transcription_data)
    print("documents_df done")

    data = [{
        'start_time': doc.metadata['start_time'],
        'end_time': doc.metadata['end_time'],
        'text': doc.page_content
    } for doc in documents_df]

    df = pd.DataFrame(data)
    df['duration'] = df['end_time'] - df['start_time']

    df['embeddings'] = rag.extract_embeddings(df,embeddings)
    print("embeddings extracted")

    # Set up Pinecone and OpenAI LLM
    # created once
#     pc.create_index(
#     name="quickstart1",
#     dimension=384, # Replace with your model dimensions
#     metric="cosine", # Replace with your model metric
#     spec=ServerlessSpec(
#         cloud="aws",
#         region="us-east-1"
#     ) 
# )
    # connect
    index = pc.Index(name="quickstart1")

    # inserted once
    # inserting embeddings to VDB
    # for i, row in df.iterrows():
    #     vector = row['embeddings']
    #     metadata = {
    #         "text": row['text'],
    #         "start_time": row['start_time'],
    #         "end_time": row['end_time'],
    #         "duration" : row["duration"]
    #     }
    #     index.upsert(vectors=[(str(i), vector, metadata)])

    # 
    query_text = "So it's another Baltic Cup 22 runs already,  looted by Pakistan in the summer.  He's got a tingle and good call for the Empire to stay outside that line.  That's a slice and it's the catch.  So the Grolmys over and I will watch us strike from England just at the right time.  He was steaming forward and threatening to win the game on his own.  This is a slice in New Deal, a pressure catch and it's taken.  They'll put it in the catcher.  That's a golden wicked.  See what it means to Oymorgan.  of a SNF 24-148-7."
    # "What is the score?"
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    query_vector = embedding_model.embed_query(query_text)
    print("query_vector", len(query_vector))

    # Ensure the query vector is a flat list of floats and matches the index's dimensionality
    if isinstance(query_vector, np.ndarray):
        query_vector = query_vector.tolist()

    index_info = index.describe_index_stats()
    expected_dim = index_info['dimension']
    print("expected_dim" , expected_dim)
    actual_dim = len(query_vector)
    print("actual_dim" , actual_dim)

    if actual_dim != expected_dim:
        raise ValueError(f"Query vector dimensionality {actual_dim} does not match index dimensionality {expected_dim}.")

    # Query the index
    results = index.query(id="5", top_k=10)
    # print("results", results)

    # # Display results
    # for match in results['matches']:
    #     print(f"ID: {match['id']}, Score: {match['score']}")
    #     # Retrieve metadata
    #     result = index.fetch(ids=[match['id']])
    #     metadata = result['vectors'][match['id']]['metadata']
    #     print(f"Text: {metadata['text']}, Start Time: {metadata['start_time']}, End Time: {metadata['end_time']}")


    # Define a prompt template
    prompt_template = """The following is a conversation with an AI assistant.
    The assistant is helpful, creative, clever, and very friendly.

    Relevant Document: {context}

    User: {question}
    AI:"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )    

    memory = ConversationBufferMemory(memory_key="chat_history", output_key="answer")

    # Initialize the custom retriever
    vectordb = PineconeVectorStore(index=index, embedding=embedding_model.embed_query, text_key="text")

    custom_retriever = CustomRetriever(
        vectorstore=vectordb.as_retriever(search_type="similarity", search_kwargs={'k': 5})
    )

    qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=custom_retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT},
            memory=memory,
            output_key="answer"
        )

    # Initialize conversation state if not already initialized
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []

    # Display previous conversation
    st.write("## Your VQA Chat")
    
        # Custom CSS for the scrollable container within the right column
    css='''
    <style>
        section.main>div {
            padding-bottom: 1rem;
        }
        [data-testid="column"]>div>div {
            overflow: auto;
            height: 100vh;
        }
    </style>
    '''
    st.markdown(css, unsafe_allow_html=True)

    for i, (user_input, response) in enumerate(st.session_state.conversation):
        # User message (aligned right)
        st.markdown(
            f"""
            <div class="user-bubble-container">
                <div class="user-bubble">
                    <img src="user-icon.webp" class="avatar avatar-right"><strong></strong> {user_input}
                </div>
            </div>

            """,
            unsafe_allow_html=True
        )
        # AI message (aligned left)
        st.markdown(
            f"""
            <div class="ai-bubble-container">
                <div class="ai-bubble">
                    <img src="robot.png" class="avatar"><strong></strong>: {response}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        

    # Chatbot interface
    # user_input = st.text_input("Ask a question:", key="input_box")
    user_input = st.text_input("Ask a question:", key="input_box", placeholder="Type your message here...", label_visibility='collapsed')

    st.session_state.chat_history = []
    if st.button("Send"):
        if user_input:
            if user_input.lower() == "bye":
                st.write("Ending the conversation. Goodbye!")
                # Reset the conversation state and history
                st.session_state.conversation = []
                st.session_state.chat_history = []
            else:
                # Prepare the inputs for the conversation chain
                inputs = {
                    # "input": user_input,
                    "query": user_input,
                    "chat_history": st.session_state.chat_history
                }

                # Run the chain with the provided query
                response = qa_chain(inputs)
                answer = response.get("answer", "No response available.")
                source_documents = response.get("source_documents", [])

                # Update the conversation history with the new interaction
                st.session_state.chat_history.append({"input": user_input, "response": answer})

                # Store the interaction in the session state
                st.session_state.conversation.append((user_input, answer))

                # with st.container():
                    # st.markdown('<div class="scroll">', unsafe_allow_html=True) 
                # Display the answer
                st.write(f"###### **AI:** {answer}")
                
                if source_documents:
                    # st.write("**Source Documents:**")
                    timestamps = []
                    for i, doc in enumerate(source_documents):
                        # st.write(f"Document {i + 1}: {doc.page_content}")
                        metadata = doc.metadata
                        # st.write(f"Metadata: {metadata}")
                        start_time = metadata.get('start_time', 0)
                        timestamps.append((i, start_time))

                    # Display the timestamps and allow the user to select one
                    if timestamps:
                        # Format options properly for parsing
                        select_options = [f"Document {i + 1}: {start_time} seconds" for i, (doc_id, start_time) in enumerate(timestamps)]
                        selected_doc = st.selectbox("Select a timestamp to play from:", select_options)
                        # Extract the selected index from the select box output
                        selected_index = int(selected_doc.split()[1].replace(":", "")) - 1
                        selected_time = timestamps[selected_index][1]

                        st.write(f"Playing video from {selected_time/60}.")
                        st.video(uploaded_video, start_time=selected_time)
            # st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    # Option to ask another question
    st.write("#### You can ask another question or type 'bye' to end the conversation.")

#################################################################

st.markdown(
    """
    <style>
    .center-title {
        text-align: center;
        font-size: 64px;
        font-weight: bold;
        margin-bottom: 30px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>

    .user-bubble {
        background-color: #0000FF;
        padding: 10px;
        border-radius: 10px;
        max-width: 80%;
        text-align: right;
        margin-bottom: 10px;
        margin-top: 10px;
    }
    .ai-bubble {
        background-color: #212c33;
        padding: 10px;
        border-radius: 10px;
        max-width: 80%;
        text-align: left;
        margin-bottom: 10px;
        margin-top: 10px;
        
    }
    .avatar {
        vertical-align: middle;
        width: 40px;
        height: 40px;
        border-radius: 50%;
        margin-right: 10px;
    }
    .avatar-right {
        margin-left: 10px;
        margin-right: 0;
    }

    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    .chat-input {
        width: 80%;
        margin: 0 auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# st.markdown(
#     """
#     <style>
#     .chat-container {
#         max-height: 400px;
#         overflow-y: scoll;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )





    # .ai-bubble-container {
    #     display: flex;
    #     justify-content: flex-start;
    #     margin-bottom: 10px;
    # }


# import streamlit as st

# st.set_page_config(layout="wide")

# # Custom CSS for the scrollable container within the right column
# css='''
# <style>
#     section.main>div {
#         padding-bottom: 1rem;
#     }
#     [data-testid="column"]>div>div>div>div>div {
#         overflow: auto;
#         height: 70vh;
#     }
# </style>
# '''
# st.markdown(css, unsafe_allow_html=True)

# # Create two columns
# left_col, right_col = st.columns(2)
# right_col.write('Meow' + ' meow'*1000)

# # Static content in the left column
# with left_col:
#     st.header("Static Left Column")
#     st.write("This column does not scroll.")

# # Scrollable content in the right column
# with right_col:
#     st.header("Scrollable Right Column")
#     st.write("This column has a separate scrollable area.")
    
#     # Start of the scrollable container
#     st.markdown('<div class="scrollable-container">', unsafe_allow_html=True)
    
#     # Add a large amount of content to demonstrate scrolling
#     for i in range(1, 101):
#         st.write(f"Line {i}")

#     # End of the scrollable container
#     st.markdown('</div>', unsafe_allow_html=True)

