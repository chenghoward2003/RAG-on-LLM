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
vector_db = create_vector_database(dataset)

#%% Self-RAG Implementation

def self_critique(query, vector_db, k=5): # Retrieve and critique  
    query_emb = bert_embedding(query, BERT_Tokenizer, BERT_Model)
    similarities = [cosine_similarity([query_emb], [emb])[0][0] for _, emb in vector_db]
    top_k_indices = np.argsort(similarities)[-k:][::-1]
    retrieved_chunks = [vector_db[i][0] for i in top_k_indices]
    
    critique_prompt = f"""
    Critique the relevance of these retrieved documents for the query.
    Query: {query}
    Retrieved documents: {chr(10).join([f"{i+1}. {chunk}" for i, chunk in enumerate(retrieved_chunks)])}

    Rate each document's relevance from 1-5 (1 = irrelevant, 5 = highly relevant).
    Format: Document X: Score - Explanation
    """
    
    input_ids = LLM_tokenizer(critique_prompt, return_tensors="pt", truncation=True, max_length=1024).input_ids
    output = LLM.generate(input_ids, max_new_tokens=256)
    critique = LLM_tokenizer.decode(output[0], skip_special_tokens=True)
    
    return retrieved_chunks, critique

def generate_with_self_critique(query, chunks):
    context = "\n".join(chunks)
    generation_prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    
    input_ids = LLM_tokenizer(generation_prompt, return_tensors="pt", truncation=True, max_length=1024).input_ids
    output = LLM.generate(input_ids, max_new_tokens=128)
    initial_answer = LLM_tokenizer.decode(output[0], skip_special_tokens=True)
    
    critique_prompt = f"""
    Critique the quality of this answer based on the context and question.
        Question: {query}
        Context: {context}
        Answer: {initial_answer}

        Evaluate the answer on:
        1. Does it correctly answer the question?
        2. Does it stick to the provided context?

        Provide scores and brief explanations for each criterion.
        """
    
    input_ids = LLM_tokenizer(critique_prompt, return_tensors="pt", truncation=True, max_length=1024).input_ids
    output = LLM.generate(input_ids, max_new_tokens=256)
    generation_critique = LLM_tokenizer.decode(output[0], skip_special_tokens=True)
    
    return initial_answer, generation_critique

def self_improve_answer(query, chunks, initial_answer, critiques): # Generate final answer based on the critiques
    improvement_prompt = f"""
    Based on the critiques, improve the answer to make it better.

        Question: {query}
        Context: {chunks}
        Initial Answer: {initial_answer}
        Retrieval Critique: {critiques[0]}
        Generation Critique: {critiques[1]}

    Provide an improved answer that addresses the issues identified in the critiques:
    """
    
    input_ids = LLM_tokenizer(improvement_prompt, return_tensors="pt", truncation=True, max_length=1024).input_ids
    output = LLM.generate(input_ids, max_new_tokens=256)
    improved_answer = LLM_tokenizer.decode(output[0], skip_special_tokens=True)
    
    return improved_answer

def demo_self_rag(query, vector_db, k=3):
    print(f"Query: {query}\n")
    
    print("STEP 1: Retrieval with Self-Critique")
    retrieved_chunks, retrieval_critique = self_critique(query, vector_db, k)
    print(f"Retrieved {len(retrieved_chunks)} chunks:")
    for i, chunk in enumerate(retrieved_chunks, 1):
        print(f"{i}. {chunk[:100]}...")
    print(f"\nRetrieval Critique:\n{retrieval_critique}\n")
    
    print("STEP 2: Generation with Self-Critique")
    initial_answer, generation_critique = generate_with_self_critique(query, retrieved_chunks)
    print(f"Initial Answer: {initial_answer}")
    print(f"\nGeneration Critique:\n{generation_critique}\n")
    
    print("STEP 3: Self-Improvement")
    improved_answer = self_improve_answer(query, retrieved_chunks, initial_answer, [retrieval_critique, generation_critique])
    print(f"Improved Answer: {improved_answer}\n")
    
    return {
        'query': query,
        'retrieved_chunks': retrieved_chunks,
        'retrieval_critique': retrieval_critique,
        'initial_answer': initial_answer,
        'generation_critique': generation_critique,
        'improved_answer': improved_answer,
    }

#%% Run demo
result = demo_self_rag("What is the capital of United Islands?", vector_db, k=3)