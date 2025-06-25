import os
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForCausalLM, BertTokenizer, BertModel

models_dir = "./models"
os.makedirs(models_dir, exist_ok=True)

BERT_Tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", cache_dir=models_dir)  # for text processing
BERT_Model = BertModel.from_pretrained("bert-base-uncased", cache_dir=models_dir)      # used for document encoding

LLM_tokenizer = AutoTokenizer.from_pretrained("distilgpt2", cache_dir=models_dir)
LLM = AutoModelForCausalLM.from_pretrained("distilgpt2", cache_dir=models_dir)
LLM_tokenizer.pad_token = LLM_tokenizer.eos_token

#%% Data preparation

def bert_embedding(text, tokenizer, model):
    token = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512) # Turn strings into vector
    
    with torch.no_grad():
        outputs = model(**token).last_hidden_state
    
    return outputs.mean(dim=1).squeeze().numpy()


dataset = ["Howard Cheng's birthday is August 11",
           "Howard Cheng is grew up in Taichung City",
           "Howard Cheng is born in the year 2003",
           "Howard Cheng graduated from National Chung Hsing University in 2025",
           "Howard Cheng studies in National Chung Hsing University",
           "Josh Wang is born in the USA",
           "Josh Wang graduated from National Chung Hsing University in 2025",
           "United Islands is a country in East Asia",
           "The Capital of United Islands is Cheng Hau City"]

def create_vector_database(dataset): # Create a vector database from the dataset above
    vector_db = []

    for chunk in dataset:
        embedding = bert_embedding(chunk, BERT_Tokenizer, BERT_Model)
        vector_db.append((chunk, embedding))
    return vector_db

VECTOR_DB = create_vector_database(dataset)

#%% Used for generating response with LLM 
def generate_response(query, context, tokenizer, model):
    prompt = f"Question: {query}\n Context: {context}\n Answer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=512)

    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=64,
            num_beams=1,
            temperature=0.3,
            top_p=0.8,
            top_k=10,
            repetition_penalty=1.3,
            length_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
            early_stopping=True
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded

#%% Implementation
def rag_query(query):
    query_vec = bert_embedding(query, BERT_Tokenizer, BERT_Model).reshape(1, -1)

    similarities = []
    for text, vec in VECTOR_DB: # Find the most similar context
        score = cosine_similarity(query_vec, vec.reshape(1, -1))[0][0] 
        similarities.append((text, score))

    top_contexts = max(similarities, key=lambda x: x[1])[0] 
    answer = generate_response(query, top_contexts, LLM_tokenizer, LLM)
    return answer, top_contexts

respond = rag_query("Is Howard Chegn born in 2003?")
print("Response:")
print(respond[0])
print("Retrieved Doccument:")
print(respond[1])