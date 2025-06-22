import os
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

models_dir = "./models"
os.makedirs(models_dir, exist_ok=True)

BERT_q = T5Tokenizer.from_pretrained("google-t5/t5-small", cache_dir=models_dir) # for text processing
BERT_d = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small", cache_dir=models_dir)
generation_tokenizer = gpt2

def t5_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        encoder_outputs = model.encoder(**inputs)
    embeddings = encoder_outputs.last_hidden_state.mean(dim=1)
    return embeddings.squeeze().numpy()

dataset = ["Howard's birthday is 2003 August 11th",
           "Howard is born in Taichung",
           "Howard studied in National Chung Hsing University",
           "Stanley teaches in National Taiwan University of Science and Technology"]

#%%
VECTOR_DB = []
def create_vector_database():
    for i, chunk in enumerate(dataset):
        embedding = t5_embedding(chunk, BERT_q, BERT_d)
        VECTOR_DB.append((chunk, embedding))
        print(f'Added chunk {i+1}/{len(dataset)} to the database')
    print("Vector database created successfully!")
create_vector_database()

#%%
def rag_query(query, top_k=2):
    query_embedding = LLM.encode(query)
    
    similarities = []
    for chunk, embedding in VECTOR_DB:
        similarity = cosine_similarity([query_embedding], [embedding])[0][0]
        similarities.append((similarity, chunk))
    
    similarities.sort(reverse=True)
    retrieved_context = " ".join([chunk for _, chunk in similarities[:top_k]])
    
    response = generate_response(query, retrieved_context)
    
    return response, retrieved_context
#%%
def generate_response(query, retrieved_context, max_length=100):
    input_text = f"Question: {query}\nContext: {retrieved_context}\nAnswer:"
    inputs = generation_tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True, padding=True)

    with torch.no_grad():
        outputs = BERT_d.generate(
            inputs.input_ids,
            max_length=max_length,
            num_beams=3,
            temperature=0.7,
            do_sample=True,
            pad_token_id=generation_tokenizer.eos_token_id,
            early_stopping=True
        )
    
    response = generation_tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Answer:" in response:
        response = response.split("Answer:")[-1].strip()
    
    return response