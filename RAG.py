import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import T5Tokenizer, T5EncoderModel, GPT2Tokenizer, GPT2LMHeadModel

models_dir = "./models"
os.makedirs(models_dir, exist_ok=True)

BERT_q = T5Tokenizer.from_pretrained("google-t5/t5-small", cache_dir=models_dir, legacy=False) # for text processing
BERT_d = T5EncoderModel.from_pretrained("google-t5/t5-small", cache_dir=models_dir) # used for document encoding

LLM_tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2", cache_dir=models_dir)
LLM_tokenizer.pad_token = LLM_tokenizer.eos_token
LLM = GPT2LMHeadModel.from_pretrained("openai-community/gpt2", cache_dir=models_dir)

#%% Data preparation
def t5_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        encoder_outputs = model(**inputs)
    embeddings = encoder_outputs.last_hidden_state.mean(dim=1)
    return embeddings.squeeze().numpy()

dataset = ["Howard's birthday is 2003 August 11th",
           "Cindy's birthday is 2003 July 10th",
           "Howard is born in Taichung City",
           "Howard graduated from National Chung Hsing University",
           "Stanley teaches in National Taiwan University of Science and Technology"]

def create_vector_database(dataset):
    vector_db = []
    for chunk in dataset:
        embedding = t5_embedding(chunk, BERT_q, BERT_d)
        vector_db.append((chunk, embedding))
    return vector_db

VECTOR_DB = create_vector_database(dataset)

#%% Used for generating response with LLM 
def generate_response(query, RAG):
    input_text = f"Question: {query}\nContext: {RAG}\nAnswer:"
    inputs = LLM_tokenizer(
        input_text,
        return_tensors="pt",
        max_length=64,
        truncation=True,
        padding=True
    )

    with torch.no_grad():
        outputs = LLM.generate(
            inputs.input_ids,
            attention_mask=inputs['attention_mask'],
            max_length=32,
            num_beams=1,
            temperature=0.3,
            top_p=0.8,
            top_k=10,
            repetition_penalty=1.3,
            length_penalty=1.2,
            pad_token_id=LLM_tokenizer.eos_token_id,
            early_stopping=True
        )
    
    response = LLM_tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Answer:" in response:
        response = response.split("Answer:")[-1].strip()

    response = "\n".join([line for line in response.splitlines() if not line.strip().startswith("Source:")])
    
    return response

#%% Implementation
def rag_query(query):
    query_embedding = t5_embedding(query, BERT_q, BERT_d)
    
    similarities = []
    for chunk, embedding in VECTOR_DB:
        sim = np.exp(np.dot(query_embedding, embedding))
        similarities.append((chunk, sim))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_chunk = similarities[0][0]
    
    answer = generate_response(query, top_chunk)
    
    return answer, top_chunk

respond = rag_query("When is Howard's birthday")
print("Response:")
print(respond[0])
print("Retrieved Context:")
print(respond[1])