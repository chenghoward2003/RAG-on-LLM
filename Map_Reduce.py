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

#%% Data preparation

def bert_embedding(text, tokenizer, model):
    token = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512) # Turn strings into vector
    
    with torch.no_grad():
        outputs = model(**token).last_hidden_state
    
    return outputs.mean(dim=1).squeeze().numpy()

def read_dataset_from_file(filename="database.txt"):
    with open(filename, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines() if line.strip()]

def create_vector_database(dataset):
    vector_db = []

    for chunk in dataset:
        embedding = bert_embedding(chunk, BERT_Tokenizer, BERT_Model)
        vector_db.append((chunk, embedding))

    return vector_db

dataset = read_dataset_from_file()

#%% Map-Reduce

def retrieve_top_k(query, vector_db, k=3):
    query_emb = bert_embedding(query, BERT_Tokenizer, BERT_Model)
    similarities = [cosine_similarity([query_emb], [emb])[0][0] for _, emb in vector_db]
    top_k_indices = np.argsort(similarities)[-k:][::-1]
    return [vector_db[i][0] for i in top_k_indices]

def map_generate_answers(query, chunks): # Generate answer for each chunk
    answers = []
    for chunk in chunks:
        prompt = f"Context: {chunk}\n\n Question: {query}\n\n Answer:"
        input_ids = LLM_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).input_ids
        output = LLM.generate(input_ids, max_new_tokens=64)
        answer = LLM_tokenizer.decode(output[0], skip_special_tokens=True)
        answers.append(answer)
    return answers

def reduce_answer(query, partial_answers): # Find the final answer based on partial answers
    combined = ""
    for idx, ans in enumerate(partial_answers, 1):
        combined += f"Answer {idx}: {ans}\n"
    prompt = (
        f"synthesize a single, accurate, and concise final answer. "
        f"Use only the information provided in the answers. "
        f"\n\n Question: {query}\n\n {combined}\n Final Answer:"
    )
    input_ids = LLM_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).input_ids
    output = LLM.generate(input_ids, max_new_tokens=128)
    final_answer = LLM_tokenizer.decode(output[0], skip_special_tokens=True)
    return final_answer

def demo_map_reduce_rag(query, vector_db, k=3):
    print(f"Query: {query}\n")
    top_chunks = retrieve_top_k(query, vector_db, k)
    print(f"Top {k} Chunks:\n", top_chunks, "\n")
    partial_answers = map_generate_answers(query, top_chunks)
    print(f"Partial Answers:\n", partial_answers, "\n")
    final_answer = reduce_answer(query, partial_answers)
    print(f"Final Answer:\n", final_answer)
    return final_answer

demo_map_reduce_rag("Which city does Howard Cheng grew up in?", create_vector_database(dataset))