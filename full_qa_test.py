import json
import pickle
import pyinputplus
import pandas as pd
import functions as fn


# Constants
LINE = 100*'-'
DOUBLE_LINE = 100*'='

# Importing base
df = pd.read_csv("tokenized_base.csv")

# Importing parameters
with open("parameters.pickle", "rb") as infile:
    parameters = pickle.load(infile)

embeddings_list = parameters["embeddings_list"]
vector_length = parameters["vector_length"]

# Importing test questions
with open("test_questions.json", "r") as infile:
    questions_data = json.load(infile)

questions = questions_data["questions"]
real_answers = questions_data["real_answers"]
real_laws_and_articles = questions_data["real_articles_and_laws"]

# Recording results
results = pd.DataFrame(columns=["question",
                                "related articles found",
                                "obtained answer",
                                "obtained answer's law and article",
                                "real answer",
                                "real answer's law and article",
                                "metric",
                                "confidence"])

for i, question in enumerate(questions):
    print(DOUBLE_LINE)
    print(f" {i+1}. | QUESTION: {question}")
    print(DOUBLE_LINE)

    for metric in ["euclidean", "manhattan", "cosine"]:
        embedded_question = fn.question_embedding(input_question=question, vector_length=vector_length)

        similar_chunks, similar_articles, similar_laws = fn.find_similar_chunks(input_embedded=embedded_question, df=df,
                                                                               embeddings_list=embeddings_list,
                                                                               metric=metric)

        obtained_answer, confidence, chunk_index = fn.deeppavlov_bert_qa_chatbot(question=question,
                                                                                 similar_chunks=similar_chunks)

        obtained_article = similar_articles[chunk_index]
        obtained_law = similar_laws[chunk_index]

        print(LINE)
        print(f"Question: {question} | Answer: {obtained_answer}")
        print(LINE)

        print(f"Metric: {metric}")
        print(LINE)

        print(f"Confidence: {round(confidence, 2)}")
        print(LINE)

        print("Similar texts: ")
        for j, similar_text in enumerate(similar_chunks):
            print(f" {j+1}. {similar_text}")

        print(LINE)

        articles_articles_found = pyinputplus.inputYesNo(prompt="Do related chunks have an answer? ")

        data = {"question": question,
                "related articles found": articles_articles_found,
                "obtained answer": obtained_answer,
                "obtained answer's law and article": f"{similar_laws[chunk_index]}, {similar_articles[chunk_index]}",
                "real answer": real_answers[i],
                "real answer's law and article": real_laws_and_articles[i],
                "metric": metric,
                "confidence": confidence}

        results = pd.concat([results, pd.DataFrame([data])], ignore_index=True)

# Results saving
results.to_csv("full qa test - results.csv", index=False)



