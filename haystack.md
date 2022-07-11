# **Haystack**

# TODO
- Correct QA pairs created in this manner might not be so effective in retraining your Reader model. However, correcting wrong QA pairs creates training samples that your model found challenging. These examples are likely to be impactful when it comes to retraining. This is also a quicker workflow than having annotators generate both a question and an answer.
- 

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
---
# Evaluation

Haystack has all the tools needed to evaluate whole pipelines or individual Nodes, such as Retrievers, Readers, and Generators. Use evaluation and the metrics it to:

+ Judge how well your system is performing on a given domain
+ Compare the performance of different models
+ Identify underperforming Nodes in your pipeline

## Integrated and Isolated Node Evaluation
There are two evaluation modes for Pipelines: integrated and isolated Node evaluation.

In **integrated** evaluation, a Node receives the predictions from the preceding Node as input. It shows the performance users can expect when running the pipeline and it's the default mode when calling pipeline.eval().

In **isolated** evaluation, a Node is isolated from the predictions of the preceding node. Instead, it receives ground-truth labels as input. Isolated evaluation shows the maximum performance of a Node if it receives the perfect input from the preceding node. You can activate it by running pipeline.eval(add_isolated_node_eval=True).

For example, in an ExtractiveQAPipeline comprised of a Retriever and a Reader, isolated evaluation would measure the upper bound of the Reader's performance, that is the performance of the Reader assuming that the Retriever passes on all relevant documents.

If the isolated evaluation result of a Node differs significantly from its integrated evaluation result, you may need to improve the preceding Node to get the best results from your pipeline. If the difference is small, it means that you should improve the Node that you evaluated to improve the pipeline's overall result quality.

## Comparison to Open and Closed-Domain Question Answering
**Integrated** evaluation on an ExtractiveQAPipeline is equivalent to **open-domain** question answering. In this setting, QA is performed over multiple documents, typically an entire database, and the relevant documents first need to be identified.

In contrast, **isolated** evaluation of a Reader node is equivalent to **closed-domain** question answering. Here, the question is being asked on a single document. There is no retrieval step involved as the relevant document is already given.

When using pipeline.eval() in either **integrated** or isolated modes, Haystack evaluates the correctness of an extracted answer by looking for a match or overlap between the answer and prediction strings. Even if the predicted answer is extracted from a different position than the correct answer, that's fine, as long as the strings match

## Advanced Label Scopes
Answers are considered correct if the predicted answer matches the gold answer in the labels. Documents are considered correct if the predicted document ID matches the gold document ID in the labels. Sometimes, these simple definitions of "correctness" are not sufficient. There are cases where you want to further specify the "scope" within which an answer or a document is considered correct. For this reason, `EvaluationResult.calculate_metrics()` offers the parameters `answer_scope` and `document_scope`.

Say you want to ensure that an answer is only considered correct if it stems from a specific context of surrounding words. This is especially useful if your answer is very short, like a date (for example, "2011") or a place ("Berlin"). Such short answer might easily appear in multiple completely different contexts. Some of those contexts might perfectly fit the actual question and answer it. Some others might not: they don't relate to the question at all but still contain the answer string. In that case, you might want to ensure that only answers that stem from the correct context are considered correct. To do that, specify `answer_scope="context"` in `calculate_metrics()`. 

`answer_scope` takes the following values:
- `any` (default): Any matching answer is considered correct.
- `context`: The answer is only considered correct if its context matches as well. It uses fuzzy matching (see `context_matching` parameters of `pipeline.eval()`).
- `document_id`: The answer is only considered correct if its document ID matches as well. You can specify a custom document ID through the `custom_document_id_field` parameter of `pipeline.eval()`.
- `document_id_and_context`: The answer is only considered correct if its document ID and its context match as well.

In Question Answering, to enforce that the retrieved document is considered correct whenever the answer is correct, set `document_scope` to `answer` or `document_id_or_answer`.

`document_scope` takes the following values:
- `document_id`: Specifies that the document ID must match. You can specify a custom document ID through the `custom_document_id_field` parameter of `pipeline.eval()`.
- `context`: Specifies that the content of the document must match. It uses fuzzy matching (see the `context_matching` parameters of `pipeline.eval()`).
- `document_id_and_context`: A Boolean operation specifying that both `'document_id' AND 'context'` must match.
- `document_id_or_context`: A Boolean operation specifying that either `'document_id' OR 'context'` must match.
- `answer`: Specifies that the document contents must include the answer. The selected `answer_scope` is enforced.
- `document_id_or_answer` (default): A Boolean operation specifying that either `'document_id' OR 'answer'` must match.


---

# Generator

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

# Quick reads

- For a better understanding of the basic concepts underlying **nodes** and **pipelines**, I'd recommend checking out the [Haystack documentation](https://haystack.deepset.ai/overview/intro), especially (but not exclusively) the parts pertaining specifically to [nodes](https://haystack.deepset.ai/pipeline_nodes/overview) and [pipelines](https://haystack.deepset.ai/components/pipelines). In short, nodes can be thought of as cells that do very specific tasks: since you seem to be using an out-of-the-box QA pipeline you might have noticed common nodes like Retriever or Reader, which are built into Haystack. A pipeline connects nodes together to do larger tasks: this is the case with the very common Retriever-Reader pipeline, where the output of a Retriever node is fed into a Reader node. A custom node is exactly as it sounds: in addition to the myriad kinds of nodes that are built into Haystack, you can also build your own from scratch for task-specific purposes. One or multiple [custom nodes](https://haystack.deepset.ai/pipeline_nodes/custom-nodes) would be a great way to add the caching and retrieval of previous questions into your pipeline.

- Reducing Memory footprint 
    
    https://haystack-community.slack.com/archives/C01J3TZM9HT/p1643281566198200
    Hi @Stefano Fiorucci
    great to hear that you will share your application with the community! Regarding your first question, you could have a look at our recent model distillation features. For example, you could replace your reader model deepset/roberta-base-squad2 with deepset/roberta-base-squad2-distilled instead. It's the same size but better answers! :slightly_smiling_face: We're also working on smaller models.

    A smaller model is deepset/tinybert-6l-768d-squad2 It's about half the size so that should reduce RAM usage. However, the F1 score on the SQuAD dataset drops to 76% compared to 83% with your current deepset/roberta-large-squad2 model. You would need to check whether the answers are good enough for your application.

# Turning Cloud Notebooks(Colab, Kaggle) into web app
- Streamlit Cloud, [Huggingface spaces](https://huggingface.co/spaces), https://cloud.google.com/run (free for 1 year), AWS Sagemaker
