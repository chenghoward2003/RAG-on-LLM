import os
import numpy as np
import torch
from itertools import combinations
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

#%% GraphRAG: entity graph construction, graph-aware retrieval, and answer generation

def llm_generate_text(prompt, max_new_tokens=128):
    inputs = LLM_tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=1024)
    with torch.no_grad():
        outputs = LLM.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=3,
            pad_token_id=LLM_tokenizer.eos_token_id,
        )
    return LLM_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

def extract_entities(text):

    instruction = (
        "Extract named entities and key concepts (people, organizations, locations, dates, core nouns) "
        "from the text below. Return only a comma-separated list without explanations.\n\n"
        f"Text: {text}\n\nEntities:"
    )
    raw = llm_generate_text(instruction, max_new_tokens=96)
    parts = [p.strip().strip('-').strip() for p in raw.replace('\n', ',').split(',')]
    entities = {p.lower() for p in parts if p}
    return entities

def build_entity_graph(chunks):

    chunk_entities = []
    entity_to_chunks = {}
    edges = {}

    for idx, chunk in enumerate(chunks):
        entities = extract_entities(chunk)
        chunk_entities.append(entities)
        for ent in entities:
            entity_to_chunks.setdefault(ent, set()).add(idx)
        # co-occurrence edges per chunk
        for a, b in combinations(sorted(entities), 2):
            key = (a, b)
            edges[key] = edges.get(key, 0) + 1

    # neighbors map
    neighbors = {}
    for (a, b), _w in edges.items():
        neighbors.setdefault(a, set()).add(b)
        neighbors.setdefault(b, set()).add(a)

    return {
        'chunks': list(chunks),
        'chunk_entities': chunk_entities,
        'entity_to_chunks': entity_to_chunks,
        'edges': edges,
        'neighbors': neighbors,
    }

def generate_response(query, context):
    prompt = (
        "Answer the question ONLY using the provided context.\n"
        f"Context: {context}\n"
        f"Question: {query}\n"
        "Answer:"
    )
    return llm_generate_text(prompt, max_new_tokens=128)

def normalize_scores(values):
    if not values:
        return []
    v_min, v_max = min(values), max(values)
    if v_max == v_min:
        return [1.0 for _ in values]
    return [(v - v_min) / (v_max - v_min) for v in values]

def graph_rag_retrieve(query, graph_index, vector_db, top_k=5, alpha=0.7):

    query_entities = extract_entities(query)
    if not query_entities:
        query_entities = set()

    entity_to_chunks = graph_index['entity_to_chunks']
    neighbors = graph_index['neighbors']
    chunk_entities = graph_index['chunk_entities']

    candidate_indices = set()
    for ent in query_entities:
        candidate_indices |= entity_to_chunks.get(ent, set())

    neighbor_entities = set()
    for ent in query_entities:
        neighbor_entities |= neighbors.get(ent, set())
    for ent in neighbor_entities:
        candidate_indices |= entity_to_chunks.get(ent, set())

    if not candidate_indices:
        query_vec = bert_embedding(query, BERT_Tokenizer, BERT_Model).reshape(1, -1)
        sims = [cosine_similarity(query_vec, emb.reshape(1, -1))[0][0] for _txt, emb in vector_db]
        top_idx = list(np.argsort(sims)[-top_k:][::-1])
        return [vector_db[i][0] for i in top_idx]

    query_vec = bert_embedding(query, BERT_Tokenizer, BERT_Model).reshape(1, -1)
    graph_scores = []
    sim_scores = []
    candidates = sorted(candidate_indices)

    for i in candidates:
        ents = chunk_entities[i]
        direct_overlap = len(ents & query_entities)
        neighbor_overlap = len(ents & neighbor_entities)
        graph_score = direct_overlap * 2.0 + neighbor_overlap * 1.0
        graph_scores.append(graph_score)

        # vector similarity using existing vector_db
        sim = cosine_similarity(query_vec, vector_db[i][1].reshape(1, -1))[0][0]
        sim_scores.append(sim)

    graph_scores_n = normalize_scores(graph_scores)
    sim_scores_n = normalize_scores(sim_scores)

    combined = [alpha * g + (1.0 - alpha) * s for g, s in zip(graph_scores_n, sim_scores_n)]
    ranked = [idx for _, idx in sorted(zip(combined, candidates), key=lambda x: x[0], reverse=True)]
    top_indices = ranked[:top_k]
    return [vector_db[i][0] for i in top_indices]

def graph_rag_query(query, graph_index, vector_db, top_k=5):
    retrieved = graph_rag_retrieve(query, graph_index, vector_db, top_k=top_k, alpha=0.7)
    context = "\n".join(retrieved)
    answer = generate_response(query, context)
    return answer, retrieved

#%% Build graph index once
GRAPH_INDEX = build_entity_graph(dataset)

#%% Demo
if __name__ == "__main__":
    q = "Which city did Howard Cheng grow up in and when is his birthday?"
    ans, ctx = graph_rag_query(q, GRAPH_INDEX, vector_db, top_k=4)
    print("Question:\n", q)
    print("\nAnswer:\n", ans)
    print("\nRetrieved Context:")
    for i, c in enumerate(ctx, 1):
        print(f"{i}. {c}")