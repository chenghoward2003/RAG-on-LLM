# Retrieval-Augmented Generation (RAG) on LLM

A demonstration about implementing Retrieval-Augmented Generation (RAG) to improve AI responses.

## 💡 Motivation

* **Supplement missing information** in LLMs without retraining
* **Reduce hallucinations** by grounding responses in external data
* **Enable domain-specific knowledge injection**
* **Improve transparency** in LLM reasoning


## 🧠 Architecture Overview

### 1. **Input**

* User query for tasks like:

  * Question Answering
  * Fact Verification
  * Jeopardy-style Generation

### 2. **Retriever (Dense Passage Retrieval)**

* **Query Encoder**: T5 (`google-t5/t5-small`)
* **Document Encoder**: T5 model encoder
* **Similarity**: Maximum Inner Product Search (MIPS) based on dot product
* **Index**: Custom in-memory vector store

### 3. **Generator**

* **Model**: GPT-2 (`gpt2-medium`)
* **Tokenization**: `Google/BART`

## 🛠️ Setup

```bash
pip install -r requirements.txt
```

## 📦 Dataset

A small hardcoded dataset simulates an external knowledge base:

```python
[
  "Howard's birthday is 2003 August 11th",
  "Howard is born in Taichung City",
  "Howard graduated from National Chung Hsing University",
  "Stanley teaches in National Taiwan University of Science and Technology"
]
```

---

## 🚀 How It Works

### 1. **Embedding & Indexing**

Each document in the dataset is encoded using T5 and stored with its vector representation.

### 2. **Query Handling**

When a user inputs a query:

* The query is encoded
* Top-k similar documents are retrieved based on exponential dot product similarity
* The context is concatenated and passed to the generator

### 3. **Generation**

The context-augmented prompt is fed to GPT-2 to generate an answer.

---

## 📌 Example

```python
respond = rag_query("When is Howard's birthday?")
print("Response: ", respond[0])
print("Retrieved Context: ", respond[1])
```

**Output:**

```
Response: 2003 August 11th
Retrieved Context: Howard's birthday is 2003 August 11th Howard is born in Taichung City
```

## 📚 References

* [RAG: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)