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
        outputs = model(**inputs).last_hidden_state
        embedding = outputs.mean(dim=1).squeeze()
    return embedding.numpy()


dataset = ["Howard's birthday is 2003 August 11th",
           "Howard is born in Taichung City",
           "Howard graduated from National Chung Hsing University",
           "Stanley teaches in National Taiwan University of Science and Technology",
           "Howard studies in National Chung Hsing University"]

def create_vector_database(dataset):
    vector_db = []
    for chunk in dataset:
        embedding = t5_embedding(chunk, BERT_q, BERT_d)
        vector_db.append((chunk, embedding))
    return vector_db

VECTOR_DB = create_vector_database(dataset)

#%% Used for generating response with LLM 
def generate_response(query, context, tokenizer, model):
    prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
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
    query_vec = t5_embedding(query, BERT_q, BERT_d).reshape(1, -1)
    ranked = sorted(
        ((text, cosine_similarity(query_vec, vec.reshape(1, -1))[0][0]) for text, vec in VECTOR_DB),
        key=lambda x: x[1], reverse=True
    )
    top_contexts = "\n".join(text for text, _ in ranked[:3])
    answer = generate_response(query, top_contexts, LLM_tokenizer, LLM)
    return answer, top_contexts

respond = rag_query("Is Howard born in 2003?")
print("Response:")
print(respond[0])
print("Retrieved Doccument:")
print(respond[1])