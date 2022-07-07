# **Haystack**

## **Pseudo Label Generator**

Use Pseudo Label Generator to create training data for dense retrievers without human annotation. Pseudo Label Generator uses Generative Pseudo Labelling (GPL), which is an unsupervised domain adaptation method for training dense retrievers. It generates labels for your Haystack Documents. When combined with a QuestionGenerator and a Retriever, it returns questions, labels, and negative passages that you can then use to train your EmbeddingRetriever.

[https://www.notion.so](https://www.notion.so)

# What we need compulsory

1. Evaluation system to evaluate the model accurcacy

## **Question Answer Generation Pipeline**

This pipeline takes a document as input, generates questions on it, and attempts to answer these questions using a Reader model

```python
reader = FARMReader("deepset/roberta-base-squad2")
qag_pipeline = QuestionAnswerGenerationPipeline(question_generator, reader)
for idx, document in enumerate(tqdm(document_store)):

    print(f"\n * Generating questions and answers for document {idx}: {document.content[:100]}...\n")
    result = qag_pipeline.run(documents=[document])
    print_questions(result)
```

---

## Difference between reader and generator

Hi **[@pratikkotian04](https://github.com/pratikkotian04)** , thanks for bringing up this issue. So I believe this is coming up because the Answers from the Generator are given None for their score. When JoinAnswers tries to sort these with the Answers coming from the Reader which do have a score, this Error is thrown:`TypeError: '<' not supported between instances of 'NoneType' and 'float'`

I've opened up a [PR](https://github.com/deepset-ai/haystack/pull/2436) which should fix this problem. It introduces a `sort` argument to the initialization of JoinAnswers. Could you try this out, and set `JoinAnswers(sort_by_score=False)`?

[Unable to use Generator instead of Reader in Text and Table QA Pipeline · Issue #2435 · deepset-ai/haystack](https://github.com/deepset-ai/haystack/issues/2435)

---

## Retriever

Retrievers help **narrowing down the scope for the Reader to smaller units of text** where a given question could be answered. 

With InMemoryDocumentStore or SQLDocumentStore, you can use the TfidfRetriever. For more retrievers, please refer to the tutorial-1.

```python
# An in-memory TfidfRetriever based on Pandas dataframes
from haystack.nodes import TfidfRetriever
retriever = TfidfRetriever(document_store=document_store)
```

The Retriever has a huge impact on the performance of our overall search pipeline.

### **Different types of Retrievers**

### **Sparse**

Family of algorithms based on counting the occurrences of words (bag-of-words) resulting in very sparse vectors with length = vocab size.

**Examples**: BM25, TF-IDF

**Pros**: Simple, fast, well explainable

**Cons**: Relies on exact keyword matches between query and text

### **Dense**

These retrievers use neural network models to create "dense" embedding vectors. Within this family there are two different approaches:

a) Single encoder: Use a **single model** to embed both query and passage.b) Dual-encoder: Use **two models**, one to embed the query and one to embed the passage

Recent work suggests that dual encoders work better, likely because they can deal better with the different nature of query and passage (length, style, syntax ...).

**Examples**: REALM, DPR, Sentence-Transformers

**Pros**: Captures semantinc similarity instead of "word matches" (e.g. synonyms, related topics ...)

**Cons**: Computationally more heavy, initial training of model

---

## **Reader**

A Reader scans the texts returned by retrievers in detail and extracts the k best answers. They are based on powerful, but slower deep learning models.

Haystack currently supports Readers based on the frameworks FARM and Transformers. With both you can either load a local model or one from Hugging Face's model hub (https://huggingface.co/models).

**Here:** a medium sized RoBERTa QA model using a Reader based on FARM (https://huggingface.co/deepset/roberta-base-squad2)

**Alternatives (Reader):** TransformersReader (leveraging the `pipeline` of the Transformers package)

**Alternatives (Models):** e.g. "distilbert-base-uncased-distilled-squad" (fast) or "deepset/bert-large-uncased-whole-word-masking-squad2" (good accuracy)

**Hint:** You can adjust the model to return "no answer possible" with the no_ans_boost. Higher values mean the model prefers "no answer possible"

### **FARMReader**

```python
# Load a local model or any of the QA models on
# Hugging Face's model hub (https://huggingface.co/models)

reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)
```

### **TransformersReader**

```python
# Alternative:
reader = TransformersReader(model_name_or_path="distilbert-base-uncased-distilled-squad", tokenizer="distilbert-base-uncased", use_gpu=-1)
```

The Reader takes a question and a set of Documents as input and returns an Answer by selecting a text span within the Documents. The Reader is also known as an Open-Domain QA system in Machine Learning speak.

**Position in a Pipeline - Generally after a Retriever**

**Input - Documents**

**Output - Answers**

**Classes -** FARMReader, TransformersReader, TableReader

**Pros**

- Built on the latest transformer-based language models
- Strong in their grasp of semantics
- Sensitive to syntactic structure
- State-of-the-art in QA tasks like SQuAD and Natural Questions

Haystack Readers contain all the components of end-to-end, open-domain QA systems, including:

- Loading of model weights
- Tokenization
- Embedding computation
- Span prediction
- Candidate aggregation

**Cons**

- Requires a GPU to run quickly

Haystack also has a close integration with FARM which means that you can further fine-tune your Readers on labelled data using a FARMReader. See our tutorials for an end-to-end example or below for a shortened example.

```python
from haystack.nodes import FARMReader
# Initialise Reader
model = "deepset/roberta-base-squad2"
reader = FARMReader(model)
# Perform fine-tuning
train_data = "PATH/TO_YOUR/TRAIN_DATA"
train_filename = "train.json"
save_dir = "finetuned_model"
reader.train(train_data, train_filename, save_dir=save_dir)
# Load
finetuned_reader = FARMReader(save_dir)

```

---

# Transformers

[🤗 Transformers](https://huggingface.co/docs/transformers/main/en/index)

# Pipelines

![Pipelines](/img/pipelines.png)

```python
class CustomQueryClassifier(BaseComponent):
    outgoing_edges = 2

    def run(self, query: str):
        if "?" in query:
            return {}, "output_2"
        else:
            return {}, "output_1"

    def run_batch(self, queries: List[str]):
        split = {"output_1": {"queries": []}, "output_2": {"queries": []}}
        for query in queries:
            if "?" in query:
                split["output_2"]["queries"].append(query)
            else:
                split["output_1"]["queries"].append(query)

        return split, "split"


# Here we build the pipeline
p_classifier = Pipeline()
p_classifier.add_node(component=CustomQueryClassifier(), name="QueryClassifier", inputs=["Query"])
p_classifier.add_node(component=bm25_retriever, name="ESRetriever", inputs=["QueryClassifier.output_1"])
p_classifier.add_node(component=embedding_retriever, name="EmbeddingRetriever", inputs=["QueryClassifier.output_2"])
p_classifier.add_node(component=reader, name="QAReader", inputs=["ESRetriever", "EmbeddingRetriever"])
p_classifier.draw("pipeline_classifier.png")

# Run only the dense retriever on the full sentence query
res_1 = p_classifier.run(query="Who is the father of Arya Stark?")
print("Embedding Retriever Results" + "\n" + "=" * 15)
print_answers(res_1)

# Run only the sparse retriever on a keyword based query
res_2 = p_classifier.run(query="Arya Stark father")
print("ES Results" + "\n" + "=" * 15)
print_answers(res_2)

```
# **Generator**

The Generator reads a set of documents and generates an answer to a question, word by word. While extractive QA highlights the span of text that answers a query, generative QA can return a novel text answer that it has composed.

The best current approaches, such as [Retriever-Augmented Generation](https://arxiv.org/abs/2005.11401) and [LFQA](https://yjernite.github.io/lfqa.html), can draw upon both the knowledge it gained during language model pretraining (parametric memory) and the passages provided to it with a Retriever (non-parametric memory). With the advent of transformer-based retrieval methods such as [Dense Passage Retrieval](https://arxiv.org/abs/2004.04906), Retriever and Generator can be trained concurrently from the one loss signal.

**Position in a Pipeline - After the Retriever, you can use it as a substitute to the Reader.**

**Input - Documents**

**Output - Answers**

**Classes - RAGenerator, Seq2SeqGenerator**

**Pros**

- More appropriately phrased answers.
- Able to synthesize information from different texts.
- Can draw on latent knowledge stored in language model.

**Cons**

- Not easy to track what piece of information the generator is basing its response off of.