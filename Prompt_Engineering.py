import os
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel, T5Tokenizer, T5ForConditionalGeneration

models_dir = "./models"
os.makedirs(models_dir, exist_ok=True)

BERT_Tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", cache_dir=models_dir)  # for text processing
BERT_Model = BertModel.from_pretrained("bert-base-uncased", cache_dir=models_dir)      # used for document encoding

LLM_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large", cache_dir=models_dir, legacy=False)
LLM = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large", cache_dir=models_dir)
#LLM_tokenizer.pad_token = LLM_tokenizer.eos_token

#%% Data preparation

def bert_embedding(text, tokenizer, model):
    token = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512) # Turn strings into vector
    
    with torch.no_grad():
        outputs = model(**token).last_hidden_state
    
    return outputs.mean(dim=1).squeeze().numpy()

dataset = ["Howard Cheng's birthday is August 11st",
           "Howard Cheng is grew up in Taichung City",
           "Howard Cheng is born in the year 2003",
           "Howard Cheng graduated from National Chung Hsing University in 2025",
           "Howard Cheng studies in National Chung Hsing University",
           "United Islands is a country in East Asia",
           "The Capital of United Islands is Cheng Hau City"]

def read_dataset_from_file(filename="database.txt"):
    with open(filename, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines() if line.strip()]

dataset = read_dataset_from_file()

def create_vector_database(dataset): # Create a vector database from the dataset above
    vector_db = []

    for chunk in dataset:
        embedding = bert_embedding(chunk, BERT_Tokenizer, BERT_Model)
        vector_db.append((chunk, embedding))
    return vector_db

VECTOR_DB = create_vector_database(dataset)

#%% Used for generating response with LLM 
def generate_response(query, context, tokenizer, model):
    prompt = (
        "Answer the question ONLY using the provided context."
        f"Context: {context}\n"
        f"Question: {query}\n"
        "Answer:"
    )
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=256)

    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=64,
            num_beams=3,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded.strip()

#%% Implementation
def rag_query(query):
    query_vec = bert_embedding(query, BERT_Tokenizer, BERT_Model).reshape(1, -1)

    similarities = []
    for text, vec in VECTOR_DB: # Find the most similar context
        score = cosine_similarity(query_vec, vec.reshape(1, -1))[0][0] 
        similarities.append((text, score))

    top_contexts = max(similarities, key=lambda x: x[1])[0] 
    answer = generate_response(query, top_contexts, LLM_tokenizer, LLM)
    answer = answer.replace("A:\n", "").replace("A:", "").strip()
    return answer, top_contexts

respond = rag_query("When is Howard Cheng's birthday?")
print("Response:")
print(respond[0])
print("Retrieved Doccument:")
print(respond[1])