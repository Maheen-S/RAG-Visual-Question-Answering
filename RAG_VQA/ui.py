import streamlit as st
import tempfile
import os
import rag
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

from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from transformers import CLIPTokenizer
from sentence_transformers import SentenceTransformer

from langchain.schema import BaseRetriever
from typing import List
from langchain.schema import Document
from pydantic import Field
from langchain.chains.question_answering import load_qa_chain


st.set_page_config(layout="wide")


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

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

pc_api_key = os.getenv("PINECONE_API_KEY ")
pc = Pinecone(api_key=pc_api_key)


print("Models loading")
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe_whisper = pipeline("automatic-speech-recognition", model="openai/whisper-base", device=device)
print("Whisper loaded")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print("Embeddings Models loaded- sparse")

model_id = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)
print("CLIP loaded- dense")

model_dense = SentenceTransformer('sentence-transformers/clip-ViT-B-32', device=device)
print("SentenceTransformer loaded- dense")

tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
print("CLIPTokenizer loaded- dense")

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
        # frames_data = {}
    uploaded_video = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        st.video(uploaded_video)
        st.success("Video uploaded and its processing now!")
        # File uploader for the video
        # Save the uploaded video to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            temp_file.write(uploaded_video.read())
            temp_video_path = temp_file.name

        st.session_state['temp_video_path'] = temp_video_path

        # Process the uploaded video file
        transcription_df = rag.main(temp_video_path, pipe_whisper)
        print("transcription_df done")
        # frames_data = rag.extract_frames_and_compute_embeddings(temp_video_path, model,processor)
        # st.success("frames_data done!")
    else:
        st.write("Please upload a video file to proceed.")


###############################################################################################

with right_col:
    if 'temp_video_path' in st.session_state:
        temp_video_path = st.session_state['temp_video_path']
        print(temp_video_path)

    print(temp_video_path)

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

    df['txt_embeddings'] = rag.extract_embeddings(df,embeddings)
    print("txt_embeddings extracted")

    frames_data = rag.extract_frames_and_compute_embeddings(temp_video_path, model,processor)
    print(len(frames_data))

    chunks = rag.chunk_frames_by_time(frames_data, df)

    chunk_list = []
    for chunk in chunks:
        for frame in chunk["frames"]:
            chunk_list.append({
                "text": chunk["text"],
                "start_time": chunk["start_time"],
                "end_time": chunk["end_time"],
                "frame_no": frame["frame_no"],
                "timestamp": frame["timestamp"],
                "image_embedding": frame["image_embedding"]
            })

    df_chunks = pd.DataFrame(chunk_list)
    print(df_chunks)

    text_to_embedding = dict(zip(df['text'], df['txt_embeddings']))
    df_chunks['txt_embeddings'] = df_chunks['text'].map(text_to_embedding)


    # Set up Pinecone and OpenAI LLM
    # created once

    # pc.create_index(
    # name="rag-embeddings-dot",
    # dimension=len(df_chunks.iloc[0]['image_embedding']),  
    # metric="dotproduct", 
    # spec=ServerlessSpec(
    #     cloud="aws",
    #     region="us-east-1"
    #     )
    # )

    # connect
    index = pc.Index(name="rag-embeddings-dot")

    # inserting embeddings to VDB
    for i, row in df_chunks.iterrows():
        image_vector = row['image_embedding']
        text_vector = row['txt_embeddings']
        metadata = row.drop(['image_embedding', 'txt_embeddings']).to_dict()  
        
        # Convert the text embeddings to a sparse format dictionary
        sparse_text_vector = {
            "indices": list(range(len(text_vector))),
            "values": text_vector.tolist()  
        }
        
        # Convert sparse vector to a JSON string
        metadata['sparse_text_vector'] = json.dumps(sparse_text_vector)

        # Upsert image embedding as the primary vector
        index.upsert(vectors=[(str(i), image_vector, metadata)])

    # 
    query_text = "So it's another Baltic Cup 22 runs already, looted by Pakistan in the summer. He's got a tingle and good call for the Empire to stay outside that line. "
    text_query_vector = embeddings.embed_query(query_text)

    # Convert the dense text query vector to a sparse format dictionary
    sparse_text_query_vector = {
        "indices": list(range(len(text_query_vector))),
        "values": text_query_vector
    }

    dense = model_dense.encode(query_text).tolist()

    # query
    result = index.query(
        top_k=2,
        vector=dense,
        sparse_vector=sparse_text_query_vector,
        include_metadata=True
    )

    # print("Simple Query Results:")
    # for match in result['matches']:
    #     vector_id = match['id']
    #     score = match['score']
    #     metadata = match['metadata'] if 'metadata' in match else {}

    #     print(f"ID: {vector_id}, Score: {score}")
    #     if metadata:
    #         print("Metadata:", metadata)
    #     print("-" * 50)


    # Hybrid search query
    #Closer to 0==more sparse, closer to 1==more dense
    hdense, hsparse = rag.hybrid_scale(dense, sparse_text_query_vector, alpha=1)
    result = index.query(
        top_k=6,
        vector=hdense,
        sparse_vector=hsparse,
        include_metadata=True
    )

    # print("Hybrid search Query Results:")
    # for match in result['matches']:
    #     vector_id = match['id']
    #     score = match['score']
    #     metadata = match['metadata'] if 'metadata' in match else {}

    #     print(f"ID: {vector_id}, Score: {score}")
    #     if metadata:
    #         print("Metadata:", metadata)
    #     print("-" * 50)

        
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
    
    # Load the QA chain with the "stuff" method for combining documents
    combine_documents_chain = load_qa_chain(
        llm=llm,  # Your language model
        chain_type="stuff",  # The method to combine documents
        prompt=PROMPT  # The prompt template you're using
    )

    custom_hybrid_retriever = rag.CustomHybridRetriever(
        index=index,
        dense_embedder=model_dense,
        sparse_embedder=embeddings,
        alpha=1
    )

    qa_chain = RetrievalQA(
        combine_documents_chain=combine_documents_chain,  
        retriever=custom_hybrid_retriever, 
        return_source_documents=True,  # Whether to return source documents
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

                print(f"Query: {inputs['query']}")

                # Run the chain with the provided query
                # response = qa_chain.apply([{"query": inputs}])
                response = qa_chain.apply([{"query": inputs['query']}])

                
                # response = qa_chain(inputs)
                answer = response[0].get("answer", "No response available.")
                source_documents = response[0].get("source_documents", [])

                # answer = response.get("answer", "No response available.")
                # source_documents = response.get("source_documents", [])

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