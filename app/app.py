# %% [code] {"execution":{"iopub.status.busy":"2022-07-10T14:11:18.930495Z","iopub.execute_input":"2022-07-10T14:11:18.931015Z","iopub.status.idle":"2022-07-10T14:11:18.967644Z","shell.execute_reply.started":"2022-07-10T14:11:18.930931Z","shell.execute_reply":"2022-07-10T14:11:18.966617Z"},"jupyter":{"outputs_hidden":false}}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

from haystack.pipelines import ExtractiveQAPipeline
from haystack.schema import EvaluationResult, MultiLabel
from haystack.document_stores import ElasticsearchDocumentStore
from subprocess import Popen, PIPE, STDOUT
import streamlit as st
from haystack.utils import print_questions
from haystack.pipelines import (
    QuestionGenerationPipeline,
    RetrieverQuestionGenerationPipeline,
    QuestionAnswerGenerationPipeline,  # what we need
)
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import DensePassageRetriever, BM25Retriever, QuestionGenerator, FARMReader, TextConverter, PDFToTextConverter, DocxToTextConverter, PreProcessor, TransformersReader, RAGenerator
from tqdm import tqdm
from pprint import pprint
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import time
import os.path

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [code] {"execution":{"iopub.status.busy":"2022-07-10T14:11:20.113586Z","iopub.execute_input":"2022-07-10T14:11:20.114203Z","iopub.status.idle":"2022-07-10T14:13:04.63055Z","shell.execute_reply.started":"2022-07-10T14:11:20.114166Z","shell.execute_reply":"2022-07-10T14:13:04.629434Z"},"jupyter":{"outputs_hidden":false}}
# Install the latest release of Haystack in your own environment
#! pip install farm-haystack

# Install the latest master of Haystack
# !pip install --upgrade pip
# !pip install git+https://github.com/deepset-ai/haystack.git#egg=farm-haystack[colab]
# !pip install farm-haystack[faiss]
# !pip install streamlit

# %% [code] {"execution":{"iopub.status.busy":"2022-07-10T14:13:18.212136Z","iopub.execute_input":"2022-07-10T14:13:18.214513Z","iopub.status.idle":"2022-07-10T14:13:18.222202Z","shell.execute_reply.started":"2022-07-10T14:13:18.214467Z","shell.execute_reply":"2022-07-10T14:13:18.221238Z"},"jupyter":{"outputs_hidden":false}}
# Imports needed to run this notebook


# Install these to allow pipeline visualization
# !apt install libgraphviz-dev
# !pip install pygraphviz

# %% [markdown]
# ### Document Store
#
# ### FAISS
#
# FAISS is a library for efficient similarity search on a cluster of dense vectors.
# The `FAISSDocumentStore` uses a SQL(SQLite in-memory be default) database under-the-hood
# to store the document text and other meta data. The vector embeddings of the text are
# indexed on a FAISS Index that later is queried for searching answers.
# The default flavour of FAISSDocumentStore is "Flat" but can also be set to "HNSW" for
# faster search at the expense of some accuracy. Just set the faiss_index_factor_str argument in the constructor.
# For more info on which suits your use case: https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index

# %% [code] {"execution":{"iopub.status.busy":"2022-07-10T14:13:18.22756Z","iopub.execute_input":"2022-07-10T14:13:18.227853Z","iopub.status.idle":"2022-07-10T14:13:19.171111Z","shell.execute_reply.started":"2022-07-10T14:13:18.227827Z","shell.execute_reply":"2022-07-10T14:13:19.170084Z"},"jupyter":{"outputs_hidden":false}}
if os.path.isfile("/home/ubuntu/SMRT_Trainee_Test_Question_Generator/app/faiss_document_store.db"):
    os.remove("/home/ubuntu/SMRT_Trainee_Test_Question_Generator/app/faiss_document_store.db")

document_store = FAISSDocumentStore(faiss_index_factory_str="Flat")
document_store

# %% [code] {"execution":{"iopub.status.busy":"2022-07-10T14:13:19.174366Z","iopub.execute_input":"2022-07-10T14:13:19.174704Z","iopub.status.idle":"2022-07-10T14:13:19.295552Z","shell.execute_reply.started":"2022-07-10T14:13:19.174673Z","shell.execute_reply":"2022-07-10T14:13:19.294464Z"},"jupyter":{"outputs_hidden":false}}
# converters exist for PDF and text files

converter = DocxToTextConverter(
    remove_numeric_tables=False, valid_languages=["en"])
doc_docx = converter.convert(
    file_path="/home/ubuntu/SMRT_Trainee_Test_Question_Generator/data/sanitized_manual.docx", meta=None)[0]

# %% [code] {"execution":{"iopub.status.busy":"2022-07-10T14:13:19.297073Z","iopub.execute_input":"2022-07-10T14:13:19.297609Z","iopub.status.idle":"2022-07-10T14:13:19.381291Z","shell.execute_reply.started":"2022-07-10T14:13:19.297573Z","shell.execute_reply":"2022-07-10T14:13:19.380392Z"},"jupyter":{"outputs_hidden":false}}
preprocessor = PreProcessor(
    clean_empty_lines=True,
    clean_whitespace=True,
    clean_header_footer=False,
    split_by="word",
    split_length=100,  # must split length for Dense Passage Retrieval
    split_respect_sentence_boundary=True,
)
docs_default = preprocessor.process([doc_docx])
print("\n n_docs_output : {}".format(len(docs_default)))

# %% [code] {"execution":{"iopub.status.busy":"2022-07-10T14:13:19.382802Z","iopub.execute_input":"2022-07-10T14:13:19.383397Z","iopub.status.idle":"2022-07-10T14:13:55.955064Z","shell.execute_reply.started":"2022-07-10T14:13:19.383359Z","shell.execute_reply":"2022-07-10T14:13:55.954077Z"},"jupyter":{"outputs_hidden":false}}
# Initialize document store and write in the documents
# Now, let's write the dicts containing documents to our DB.
document_store.write_documents(docs_default)

# Initialize Question Generator, ask questions
question_generator = QuestionGenerator()

# retriever = DensePassageRetriever(
#     document_store=document_store,
#     query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
#     passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
#     max_seq_len_query=64,
#     max_seq_len_passage=256,
#     batch_size=16,
#     use_gpu=True,
#     embed_title=True,
#     use_fast_tokenizers=True,
# )

# Important:
# Now that after we have the DPR initialized, we need to call update_embeddings() to iterate over all
# previously indexed documents and update their embedding representation.
# While this can be a time consuming operation (depending on corpus size), it only needs to be done once.
# At query time, we only need to embed the query and compare it the existing doc embeddings which is very fast.
# document_store.update_embeddings(retriever)

# %% [markdown]
# ## **Question Answer Generation Pipeline, IMPT**
#
# This pipeline takes a document as input, generates questions on it, and attempts to answer these questions using
# a Reader model

# %% [code] {"execution":{"iopub.status.busy":"2022-07-10T14:13:55.956493Z","iopub.execute_input":"2022-07-10T14:13:55.956863Z"},"jupyter":{"outputs_hidden":false}}

reader = FARMReader("deepset/roberta-base-squad2", use_gpu=True)
data_dir = "../data"
reader.train(data_dir=data_dir, train_filename="answers.json",
             use_gpu=True, n_epochs=1, save_dir="./my_model")

# generator = Seq2SeqGenerator(model_name_or_path="vblagoje/bart_lfqa") # results were not nice, so do not use

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Saving the model happens automatically at the end of training into the `save_dir` you specified
# However, you could also save a reader manually again via:
reader.save(directory="./my_model")

# %% [code] {"execution":{"iopub.status.busy":"2022-07-10T07:51:18.448337Z","iopub.execute_input":"2022-07-10T07:51:18.449755Z","iopub.status.idle":"2022-07-10T07:54:14.412809Z","shell.execute_reply.started":"2022-07-10T07:51:18.449713Z","shell.execute_reply":"2022-07-10T07:54:14.411827Z"},"jupyter":{"outputs_hidden":false}}
qag_pipeline = QuestionAnswerGenerationPipeline(
    question_generator, reader)  # Can also replace generator with reader

for idx, document in enumerate(tqdm(document_store)):
    print(
        f"\n * Generating questions and answers for document {idx}: {document.content[:100]}...\n")
    result = qag_pipeline.run(documents=[document])
    print_questions(result)
    print()

# %% [markdown]
# # Evaluation

# %% [markdown]
# To be able to make a statement about the **quality of results a question-answering pipeline** or any other pipeline in haystack produces, it is important to evaluate it. Furthermore, evaluation allows determining which components of the pipeline can be improved. The results of the evaluation can be saved as CSV files, which contain all the information to calculate additional metrics later on or inspect individual predictions

# %% [markdown]
# ## Fetch, Store And Preprocess the Evaluation Dataset

# %% [code] {"jupyter":{"outputs_hidden":false}}
# make sure these indices do not collide with existing ones, the indices will be wiped clean before data is inserted
doc_index = "docs"
label_index = "labels"

# %% [code] {"jupyter":{"outputs_hidden":false}}
# If Docker is available: Start Elasticsearch as docker container
# from haystack.utils import launch_es
# launch_es()

# Alternative in Colab / No Docker environments: Start Elasticsearch from source
# ! wget https: // artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.9.2-linux-x86_64.tar.gz - q
# ! tar -xzf elasticsearch-7.9.2-linux-x86_64.tar.gz
# ! chown -R daemon: daemon elasticsearch-7.9.2


es_server = Popen(
    # as daemon
    ["elasticsearch-7.9.2/bin/elasticsearch"], stdout=PIPE, stderr=STDOUT, preexec_fn=lambda: os.setuid(1)
)
# wait until ES has started
time.sleep(30)

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Connect to Elasticsearch

# Connect to Elasticsearch
document_store = ElasticsearchDocumentStore(
    host="localhost",
    username="",
    password="",
    index=doc_index,
    label_index=label_index,
    embedding_field="emb",
    embedding_dim=768,
    excluded_meta_data=["emb"],
)

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Add evaluation data to FIASS Document Store
# We first delete the custom tutorial indices to not have duplicate elements
# and also split our documents into shorter passages using the PreProcessor

# Add evaluation data to Elasticsearch Document Store
# We first delete the custom tutorial indices to not have duplicate elements
# and also split our documents into shorter passages using the PreProcessor

preprocessor_eval = PreProcessor(
    split_by="word",
    split_length=200,
    split_overlap=0,
    split_respect_sentence_boundary=False,
    clean_empty_lines=False,
    clean_whitespace=False,
)
document_store.delete_documents(index=doc_index)
document_store.delete_documents(index=label_index)

# The add_eval_data() method converts the given dataset in json format into Haystack document and label objects. Those objects are then indexed in their respective document and label index in the document store. The method can be used with any dataset in SQuAD format.
document_store.add_eval_data(
    filename="/home/ubuntu/SMRT_Trainee_Test_Question_Generator/data/answers_eval.json",
    doc_index=doc_index,
    label_index=label_index,
    preprocessor=preprocessor_eval,
)

# %% [markdown]
# ## Initialize the Two Components of an ExtractiveQAPipeline: Retriever and Reader

# %% [code] {"jupyter":{"outputs_hidden":false}}
retriever = BM25Retriever(document_store=document_store)

# %% [code] {"jupyter":{"outputs_hidden":false}}
# reader = FARMReader("deepset/roberta-base-squad2", use_gpu=True)
# data_dir = "/kaggle/input/dataset"
# reader.train(data_dir=data_dir, train_filename="answers.json",
#              use_gpu=True, n_epochs=1, save_dir="my_model")

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Define a pipeline consisting of the initialized retriever and reader
pipeline = ExtractiveQAPipeline(reader=reader, retriever=retriever)

# %% [markdown]
# ## Evaluation of an ExtractiveQAPipeline
# Here we evaluate retriever and reader in open domain fashion on the full corpus of documents i.e. a document is considered
# correctly retrieved if it contains the gold answer string within it. The reader is evaluated based purely on the
# predicted answer string, regardless of which document this came from and the position of the extracted span.
#
# The generation of predictions is separated from the calculation of metrics. This allows you to run the computation-heavy model predictions only once and then iterate flexibly on the metrics or reports you want to generate.

# %% [code] {"jupyter":{"outputs_hidden":false}}

# We can load evaluation labels from the document store
# We are also opting to filter out no_answer samples
eval_labels = document_store.get_all_labels_aggregated(
    drop_negative_labels=True, drop_no_answers=True)

# Alternative: Define queries and labels directly

# eval_labels = [
#    MultiLabel(
#        labels=[
#            Label(
#                query="who is written in the book of life",
#                answer=Answer(
#                    answer="every person who is destined for Heaven or the World to Come",
#                    offsets_in_context=[Span(374, 434)]
#                ),
#                document=Document(
#                    id='1b090aec7dbd1af6739c4c80f8995877-0',
#                    content_type="text",
#                    content='Book of Life - wikipedia Book of Life Jump to: navigation, search This article is
#                       about the book mentioned in Christian and Jewish religious teachings...'
#                ),
#                is_correct_answer=True,
#                is_correct_document=True,
#                origin="gold-label"
#            )
#        ]
#    )
# ]

# Similar to pipeline.run() we can execute pipeline.eval()
eval_result = pipeline.eval(labels=eval_labels, params={
                            "Retriever": {"top_k": 5}})

# %% [code] {"jupyter":{"outputs_hidden":false}}
# The EvaluationResult contains a pandas dataframe for each pipeline node.
# That's why there are two dataframes in the EvaluationResult of an ExtractiveQAPipeline.

retriever_result = eval_result["Retriever"]
retriever_result.head()

# %% [code] {"jupyter":{"outputs_hidden":false}}
reader_result = eval_result["Reader"]
reader_result.head()

# %% [code] {"jupyter":{"outputs_hidden":false}}
pipeline.print_eval_report(eval_result)

# %% [code] {"jupyter":{"outputs_hidden":false}}
metrics = eval_result.calculate_metrics()
print(
    f'Retriever - Recall (single relevant document): {metrics["Retriever"]["recall_single_hit"]}')
print(
    f'Retriever - Recall (multiple relevant documents): {metrics["Retriever"]["recall_multi_hit"]}')
print(f'Retriever - Mean Reciprocal Rank: {metrics["Retriever"]["mrr"]}')
print(f'Retriever - Precision: {metrics["Retriever"]["precision"]}')
print(f'Retriever - Mean Average Precision: {metrics["Retriever"]["map"]}')

print(f'Reader - F1-Score: {metrics["Reader"]["f1"]}')
print(f'Reader - Exact Match: {metrics["Reader"]["exact_match"]}')

# %% [markdown]
# Before training the FARMmodel (reader):
# - Retriever - Recall (single relevant document): 0.8
# - Retriever - Recall (multiple relevant documents): 0.8
# - Retriever - Mean Reciprocal Rank: 0.6247619047619047
# - Retriever - Precision: 0.16571428571428576
# - Retriever - Mean Average Precision: 0.6257142857142857
# - Reader - F1-Score: 0.35396155963876497
# - Reader - Exact Match: 0.05714285714285714
#
# After training the FARM model (reader)
# - Retriever - Recall (single relevant document): 0.8
# - Retriever - Recall (multiple relevant documents): 0.8
# - Retriever - Mean Reciprocal Rank: 0.6247619047619047
# - Retriever - Precision: 0.16571428571428576
# - Retriever - Mean Average Precision: 0.6257142857142857
# - Reader - F1-Score: 0.5805476872540989
# - Reader - Exact Match: 0.4

# %% [markdown]
# ## Advanced Evaluation Metrics
#
# As an advanced evaluation metric, semantic answer similarity (SAS) can be calculated. This metric takes into account whether the meaning of a predicted answer is similar to the annotated gold answer rather than just doing string comparison

# %% [code] {"jupyter":{"outputs_hidden":false}}
advanced_eval_result = pipeline.eval(
    labels=eval_labels, params={"Retriever": {"top_k": 5}}, sas_model_name_or_path="cross-encoder/stsb-roberta-large"
)

metrics = advanced_eval_result.calculate_metrics()
print(metrics["Reader"]["sas"])

# %% [markdown]
# The **metrics["Reader"]["sas"] = 0.75167686**

# %% [markdown]
# ## Advanced Label Scopes
#
#
# Answers are considered correct if the predicted answer matches the gold answer in the labels. Documents are considered correct if the predicted document ID matches the gold document ID in the labels. Sometimes, these simple definitions of "correctness" are not sufficient. There are cases where you want to further specify the "scope" within which an answer or a document is considered correct. For this reason, EvaluationResult.calculate_metrics() offers the parameters `answer_scope` and `document_scope`.

# %% [code] {"jupyter":{"outputs_hidden":false}}
metrics = eval_result.calculate_metrics(answer_scope="context")
print(
    f'Retriever - Recall (single relevant document): {metrics["Retriever"]["recall_single_hit"]}')
print(
    f'Retriever - Recall (multiple relevant documents): {metrics["Retriever"]["recall_multi_hit"]}')
print(f'Retriever - Mean Reciprocal Rank: {metrics["Retriever"]["mrr"]}')
print(f'Retriever - Precision: {metrics["Retriever"]["precision"]}')
print(f'Retriever - Mean Average Precision: {metrics["Retriever"]["map"]}')

print(f'Reader - F1-Score: {metrics["Reader"]["f1"]}')
print(f'Reader - Exact Match: {metrics["Reader"]["exact_match"]}')

# %% [markdown]
# - Retriever - Recall (single relevant document): 0.8
# - Retriever - Recall (multiple relevant documents): 0.8
# - Retriever - Mean Reciprocal Rank: 0.6247619047619047
# - Retriever - Precision: 0.16000000000000003
# - Retriever - Mean Average Precision: 0.6247619047619047
# ---
# - Reader - F1-Score: 0.5342846794824483
# - Reader - Exact Match: 0.4

# %% [code] {"jupyter":{"outputs_hidden":false}}
document_store.get_all_documents()[0]

# %% [code] {"jupyter":{"outputs_hidden":false}}
len(document_store.get_all_documents())

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Let's try Document Retrieval on a file level (it's sufficient if the correct file identified by its name (for example, 'Book of Life') was retrieved).
eval_result_custom_doc_id = pipeline.eval(
    labels=eval_labels, params={"Retriever": {"top_k": 5}}, custom_document_id_field="name"
)
metrics = eval_result_custom_doc_id.calculate_metrics(
    document_scope="document_id")
print(
    f'Retriever - Recall (single relevant document): {metrics["Retriever"]["recall_single_hit"]}')
print(
    f'Retriever - Recall (multiple relevant documents): {metrics["Retriever"]["recall_multi_hit"]}')
print(f'Retriever - Mean Reciprocal Rank: {metrics["Retriever"]["mrr"]}')
print(f'Retriever - Precision: {metrics["Retriever"]["precision"]}')
print(f'Retriever - Mean Average Precision: {metrics["Retriever"]["map"]}')

# %% [markdown]
# - Retriever - Recall (single relevant document): 0.0
# - Retriever - Recall (multiple relevant documents): 0.0
# - Retriever - Mean Reciprocal Rank: 0.0
# - Retriever - Precision: 0.0
# - Retriever - Mean Average Precision: 0.0

# %% [markdown]
# ## Evaluation of Individual Components: Reader

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Evaluate Reader on its own
reader_eval_results = reader.eval(
    document_store=document_store, label_index=label_index, doc_index=doc_index)

top_n = reader_eval_results["top_n"]
# Evaluation of Reader can also be done directly on a SQuAD-formatted file without passing the data to Elasticsearch
# reader_eval_results = reader.eval_on_file("../data/nq", "nq_dev_subset_v2.json", device=device)

# Reader Top-N-Accuracy is the proportion of predicted answers that match with their corresponding correct answer including no_answers
print(f"Reader Top-{top_n}-Accuracy:", reader_eval_results["top_n_accuracy"])

# Reader Top-1-Exact Match is the proportion of questions where the first predicted answer is exactly the same as the correct answer including no_answers
print("Reader Top-1-Exact Match:", reader_eval_results["EM"])

# Reader Top-1-F1-Score is the average overlap between the first predicted answers and the correct answers including no_answers
print("Reader Top-1-F1-Score:", reader_eval_results["f1"])

# Reader Top-N-Accuracy is the proportion of predicted answers that match with their corresponding correct answer excluding no_answers
print(f"Reader Top-{top_n}-Accuracy (without no_answers):",
      reader_eval_results["top_n_accuracy_text_answer"])

# Reader Top-N-Exact Match is the proportion of questions where the predicted answer within the first n results is exactly the same as the correct answer excluding no_answers (no_answers are always present within top n).
print(f"Reader Top-{top_n}-Exact Match (without no_answers):",
      reader_eval_results["top_n_EM_text_answer"])

# Reader Top-N-F1-Score is the average overlap between the top n predicted answers and the correct answers excluding no_answers (no_answers are always present within top n).
print(f"Reader Top-{top_n}-F1-Score (without no_answers):",
      reader_eval_results["top_n_f1_text_answer"])

# %% [markdown]
# * Reader Top-4-Accuracy: 85.71428571428571
# * Reader Top-1-Exact Match: 0.0
# * Reader Top-1-F1-Score: 0.0
# * Reader Top-4-Accuracy (without no_answers): 85.71428571428571
# * Reader Top-4-Exact Match (without no_answers): 40.0
# * Reader Top-4-F1-Score (without no_answers): 63.821344286830964

# %% [markdown]
# # The End

# ssh -i "C:\Users\shankar\Downloads\internship.pem" ubuntu@
# ssh -i "/secrets/internship.pem" ubuntu@