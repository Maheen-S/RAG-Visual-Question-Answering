#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# In[2]:

import ffmpeg
from pydub import AudioSegment
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from pydub import AudioSegment
import torch
from transformers import pipeline
import json
import librosa
import noisereduce as nr
import soundfile as sf
import pandas as pd


# In[8]:


from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain import PromptTemplate, LLMChain
from langchain.chains import RetrievalQA
from langchain.vectorstores import Pinecone as PineconeVectorStore
from transformers import pipeline
from langchain.llms.base import LLM


# In[9]:

from langchain.memory import ConversationBufferMemory
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.schema import Document
from pydantic import Field
from typing import List


# In[10]:

import openai
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import subprocess
import os
import tempfile

# In[11]:


# device = "cuda" if torch.cuda.is_available() else "cpu"
# pipe_whisper = pipeline("automatic-speech-recognition", model="openai/whisper-base", device=device)


# In[12]:


# extract audio from video
def extract_audio_from_video(video_path, audio_output_path):
    try:
        # Command to extract audio using FFmpeg
        command = ['ffmpeg', '-i', video_path, '-vn', '-acodec', 'copy', audio_output_path]
        # Execute the command
        subprocess.run(command, check=True, stderr=subprocess.PIPE)
        print(f"Audio successfully extracted to {audio_output_path}")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e.stderr.decode()}")


    # print(f"Time taken to extract_audio_from_video {t2 - t1}")
    # ffmpeg.input(video_path).output(audio_output_path).run()


# In[13]:

def denoise(audio_path, output_path):
    y, sr = librosa.load(audio_path, sr=16000)
    reduced_noise = nr.reduce_noise(y=y, sr=sr)
    sf.write(output_path, reduced_noise, sr)


# In[14]:

# whisper functions
def split_audio(audio_path, chunk_length_ms=30000):
    audio = AudioSegment.from_wav(audio_path)
    chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
    return chunks

def audiosegment_to_np(audio_segment):
    samples = np.array(audio_segment.get_array_of_samples())
    if audio_segment.channels == 2:
        samples = samples.reshape((-1, 2))
    return samples.astype(np.float32) / np.iinfo(samples.dtype).max

def transcribe_chunk(chunk, start_time_s, pipe, sample_rate):
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_chunk_path = temp_file.name

    # Export the chunk to the temporary file
    chunk.export(temp_chunk_path, format="wav")
    
    # Convert the AudioSegment to a numpy array
    chunk_np = audiosegment_to_np(chunk)
    
    # Perform the transcription
    transcription = pipe({"array": chunk_np, "sampling_rate": sample_rate}, return_timestamps=True)

    # Adjust timestamps
    for item in transcription["chunks"]:
        item["timestamp"] = (
            item["timestamp"][0] + start_time_s,
            item["timestamp"][1] + start_time_s
        )

    # Clean up the temporary file
    os.remove(temp_chunk_path)

    return transcription["chunks"]

    # # Ensure the chunk is mono
    # if chunk.channels > 1:
    #     chunk = chunk.set_channels(1)
        
    # chunk.export("/tmp/temp_chunk.wav", format="wav")
    # chunk_np = audiosegment_to_np(chunk)
    # transcription = pipe({"array": chunk_np, "sampling_rate": sample_rate}, return_timestamps=True)
        
    # for item in transcription["chunks"]:
    #     if item["timestamp"] is not None:
    #         item["timestamp"] = (
    #             item["timestamp"][0] + start_time_s,
    #             item["timestamp"][1] + start_time_s
    #         )
    # return transcription["chunks"]

def whisper_main(pipe, audio_path):
    chunks = split_audio(audio_path)

    # Transcribe each chunk
    all_transcriptions = []
    chunk_length_s = 30
    sample_rate = 16000

    for i, chunk in enumerate(chunks):
        start_time_s = i * chunk_length_s
        transcriptions = transcribe_chunk(chunk, start_time_s, pipe, sample_rate)
        all_transcriptions.extend(transcriptions)

    # Compile the final transcription
    final_transcription = {
        "transcription": " ".join([t["text"] for t in all_transcriptions]),
        "timestamps": all_transcriptions
    }

    # Save the final transcription to a JSON file
    with open("final_transcription.json", "w") as f:
        json.dump(final_transcription, f, indent=4)

# In[15]:


def main(video_path, pipe):
    # extract audio from video 
    output_path = "output_audio.wav"
    extract_audio_from_video(video_path , output_path)
    audio_data, sample_rate = librosa.load(output_path, sr=16000, mono=True)
    print(audio_data)
    
    # denoise extractd audio from video
    output_path_denoised = "denoised_audio.wav"
    denoise(output_path, output_path_denoised)   
    
    # Whisper with 30 second divisions for timestamps
    whisper_main(pipe, output_path_denoised)


# In[ ]:

# # Making documents 

# In[17]:

# with open('final_transcription.json', 'r') as f:
#     transcription_data = json.load(f)

def concatenate_sentences(transcription_data, target_duration=55):
    documents = []
    current_text = ""
    current_start_time = None
    current_end_time = None
    current_duration = 0
    
    for item in transcription_data['timestamps']:
        doc_text = item['text']
        start_time = item['timestamp'][0]
        end_time = item['timestamp'][1]
        duration = end_time - start_time
        
        if current_start_time is None:
            current_start_time = start_time
        
        if current_duration + duration <= target_duration:
            # Concatenate the text and update the end time and duration
            if current_text:
                current_text += " "
            current_text += doc_text
            current_end_time = end_time
            current_duration += duration
        else:
            # Create a new Document 
            document = Document(page_content=current_text, metadata={'start_time': current_start_time, 'end_time': current_end_time})
            documents.append(document)
            
            # Reset for the next document
            current_text = doc_text
            current_start_time = start_time
            current_end_time = end_time
            current_duration = duration
    
    # last document 
    if current_text:
        document = Document(page_content=current_text, metadata={'start_time': current_start_time, 'end_time': current_end_time})
        documents.append(document)
    
    return documents


# In[18]:


# concatenated_documents = concatenate_sentences(transcription_data)


# In[19]:


# for doc in concatenated_documents:
#     print(f"Start Time: {doc.metadata['start_time']}, End Time: {doc.metadata['end_time']}")
#     print(f"Text: {doc.page_content}\n")


# In[20]:


# # dict -> df 
# data = [{
#     'start_time': doc.metadata['start_time'],
#     'end_time': doc.metadata['end_time'],
#     'text': doc.page_content
# } for doc in concatenated_documents]

# df = pd.DataFrame(data)
# df['duration'] = df['end_time'] - df['start_time']


# In[ ]:

# # Embeddings 

# In[23]:

# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# In[24]:


def extract_embeddings(df, embeddings):
    all_embeddings = []
    for text in df["text"]:
        embedding = embeddings.embed_query(text)
        all_embeddings.append(embedding)
    return all_embeddings

# df['embeddings'] = extract_embeddings(df)



# # VDB - Pinecone

# In[27]:


# pc = Pinecone(api_key="9bf699ce-092d-48a9-8526-a219a8e3d509")


# In[ ]:


# pc.create_index(
#     name="quickstart1",
#     dimension=384, # Replace with your model dimensions
#     metric="cosine", # Replace with your model metric
#     spec=ServerlessSpec(
#         cloud="aws",
#         region="us-east-1"
#     ) 
# )


# In[28]:


# connect
# index = pc.Index(name="quickstart1")


# In[29]:


# # inserting embeddings to VDB
# for i, row in df.iterrows():
#     vector = row['embeddings']
#     metadata = {
#         "text": row['text'],
#         "start_time": row['start_time'],
#         "end_time": row['end_time'],
#         "duration" : row["duration"]
#     }
#     index.upsert(vectors=[(str(i), vector, metadata)])


# In[ ]:


# print(index.describe_index_stats())


# In[ ]:


# # Query the index

# In[30]:


# query_text = "So it's another Baltic Cup 22 runs already,  looted by Pakistan in the summer.  He's got a tingle and good call for the Empire to stay outside that line.  That's a slice and it's the catch.  So the Grolmys over and I will watch us strike from England just at the right time.  He was steaming forward and threatening to win the game on his own.  This is a slice in New Deal, a pressure catch and it's taken.  They'll put it in the catcher.  That's a golden wicked.  See what it means to Oymorgan.  of a SNF 24-148-7."
# # "What is the score?"
# embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# query_vector = embedding_model.embed_query(query_text)
# print("query_vector", len(query_vector))

# # Ensure the query vector is a flat list of floats and matches the index's dimensionality
# if isinstance(query_vector, np.ndarray):
#     query_vector = query_vector.tolist()


# index_info = index.describe_index_stats()
# expected_dim = index_info['dimension']
# print("expected_dim" , expected_dim)
# actual_dim = len(query_vector)
# print("actual_dim" , actual_dim)


# if actual_dim != expected_dim:
#     raise ValueError(f"Query vector dimensionality {actual_dim} does not match index dimensionality {expected_dim}.")

# # Query the index

# results = index.query(id="5", top_k=10)
# print("results", results)

# # Display results
# for match in results['matches']:
#     print(f"ID: {match['id']}, Score: {match['score']}")
#     # Retrieve metadata
#     result = index.fetch(ids=[match['id']])
#     metadata = result['vectors'][match['id']]['metadata']
#     print(f"Text: {metadata['text']}, Start Time: {metadata['start_time']}, End Time: {metadata['end_time']}")

# In[ ]:

# # LLM - OpenAI

# In[33]:


# openai.api_key = "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"


# In[34]:


# openai_llm = OpenAI(
#     api_key=openai.api_key, 
#     model_name="gpt-3.5-turbo-instruct",  
#     max_tokens=100,  
#     temperature=0.7,  
# )


# In[ ]:
# ## Conversation Chain with docs 

# In[35]:


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


# In[36]:


# # Define a prompt template
# prompt_template = """The following is a conversation with an AI assistant.
# The assistant is helpful, creative, clever, and very friendly.

# Relevant Document: {context}

# User: {question}
# AI:"""

# PROMPT = PromptTemplate(
#     template=prompt_template,
#     input_variables=["context", "question"]
# )


# In[37]:


# memory = ConversationBufferMemory(memory_key="chat_history", output_key="answer")


# In[38]:


# Initialize the custom retriever
# vectordb = PineconeVectorStore(index=index, embedding=embedding_model.embed_query, text_key="text")

# custom_retriever = CustomRetriever(
#     vectorstore=vectordb.as_retriever(search_type="similarity", search_kwargs={'k': 5})
# )


# In[39]:


# from langchain.chains import RetrievalQA

# qa_chain = RetrievalQA.from_chain_type(
#         llm=openai_llm,
#         chain_type="stuff",
#         retriever=custom_retriever,
#         return_source_documents=True,
#         chain_type_kwargs={"prompt": PROMPT},
#         memory=memory,
#         output_key="answer"
#     )


# In[40]:


# qa_chain({"query":"what are the main points in the text"})


# # In[41]:


# qa_chain({"query":"who seems to win the match?"})
