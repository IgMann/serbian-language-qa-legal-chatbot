import os
import re
import ast
import glob
import PyPDF2
import os
import pandas as pd
import nltk.data
from tqdm import tqdm
import transformers
from transformers import AutoTokenizer
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances, cosine_distances
from deeppavlov import build_model, configs
import numpy as np


def pdf_finder(path):
    # Constants
    LINE = 100 * '-'
    DOUBLE_LINE = 100 * '='

    # Use glob to find all pdf files in directory
    pdf_files = glob.glob(os.path.join(path, '*.pdf'))

    print(DOUBLE_LINE)
    print("Found pdfs: ")
    print(DOUBLE_LINE)

    # Loop through each pdf file
    for i, pdf_file in enumerate(pdf_files):
        # Print the name of the file
        file_name = os.path.basename(pdf_file)

        print(str(i + 1) + '. ', file_name)
        print(LINE)

    # print(DOUBLE_LINE)

    return pdf_files


def pdf_to_txt(paths):
    # Constants
    LINE = 100 * '-'
    DOUBLE_LINE = 100 * '='

    print(DOUBLE_LINE)
    print("Converted pdfs: ")
    print(DOUBLE_LINE)

    paths_number = len(paths)
    for i in tqdm(range(1, paths_number + 1)):
        path = paths[i - 1]
        with open(path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)

            # Get the first page of the PDF
            first_page = pdf_reader.pages[0]

            # Extract the text of the first page
            first_page_text = first_page.extract_text()

            if not first_page_text:
                pass
                # print("Page is unreadable.")
            else:
                # print(os.path.basename(path))
                # print(LINE)

                text = ''
                for page_num in range(len(pdf_reader.pages)):
                    pages = pdf_reader.pages[page_num]
                    text += pages.extract_text()

                    txt_filename = os.path.splitext(path)[0] + '.txt'
                    with open(txt_filename, 'w', encoding='utf-8') as txt_file:
                        txt_file.write(text)


def chunking_laws(path, new_base_name):
    # Constants
    LINE = 100 * '-'
    DOUBLE_LINE = 100 * '='

    print(DOUBLE_LINE)
    print("Chunked laws: ")
    print(DOUBLE_LINE)

    # Finding all txt files in designated path
    law_files = [f for f in os.listdir(path) if f.endswith('.txt')]
    laws_number = len(law_files)

    # Initialization of base
    df = pd.DataFrame(columns=["chunk",
                               "article",
                               "law"])

    for law_counter in tqdm(range(1, laws_number + 1)):
        law_file = law_files[law_counter - 1]
        with open(os.path.join(path, law_file), 'r') as f:
            # Finding law's name
            law = f.read()
            start_tag_name = "poslednju verziju možete naći OVDE . "
            end_tag_name = "("

            start_index_name = law.find(start_tag_name) + len(start_tag_name)
            end_index_name = law.find(end_tag_name, start_index_name)

            law_name = law[start_index_name:end_index_name].strip().replace("\n", "")
            law_name = law_name.capitalize()

            # Split law into lines
            law_lines = law.split("\n")

            # Clean up law lines
            cleaned_law_lines = []
            glava_flag = False
            for line in law_lines:
                if glava_flag == True:
                    glava_flag = False
                else:
                    if line.startswith("Član"):
                        split_line = line.split()
                        if len(split_line) > 2:
                            cleaned_law_lines.append(split_line[0] + " " + split_line[1])
                            cleaned_law_lines.append(" ".join(split_line[2:]))
                        else:
                            cleaned_law_lines.append(line)

                    elif line.startswith("Glava"):
                        if len(line.split()) > 3:
                            pass
                        else:
                            glava_flag = True

                    elif re.search(r"(.*)(\bČlan\s\d+\b)", line):
                        match = re.search(r"(.*)(\bČlan\s\d+\b)", line)
                        left_half = match.group(1).strip()
                        right_half = match.group(2).strip()
                        cleaned_law_lines.append(left_half)
                        cleaned_law_lines.append(right_half)

                    else:
                        cleaned_law_lines.append(line)

            # Finding all article signs and their indices
            article_signs = []
            article_sign_indices = []
            for i, line in enumerate(cleaned_law_lines):
                if line.startswith("Član"):
                    match = re.search(r"Član\s+(\d+)", line)
                    if match:
                        article_signs.append(line)
                        article_sign_indices.append(i)

            # Finding heads
            article_heads = []
            for i in article_sign_indices:
                potential_head = cleaned_law_lines[i - 1]

                if "Sl." in potential_head or "br." in potential_head:
                    head = cleaned_law_lines[i - 3] + cleaned_law_lines[i - 2]
                    cleaned_law_lines[i - 1] = ''
                    cleaned_law_lines[i - 2] = ''
                    cleaned_law_lines[i - 3] = ''
                elif len(potential_head.rsplit(".", maxsplit=1)) > 1:
                    split_head = potential_head.rsplit(".", maxsplit=1)
                    cleaned_law_lines[i - 1] = split_head[0]
                    head = split_head[1]
                elif len(potential_head.rsplit(";", maxsplit=1)) > 1:
                    split_head = potential_head.rsplit(";", maxsplit=1)
                    cleaned_law_lines[i - 1] = split_head[0]
                    head = split_head[1]
                else:
                    head = potential_head
                    cleaned_law_lines[i - 1] = ''

                article_heads.append(head)

            # Finding all bodies of articles
            article_bodies = []
            for i, _ in enumerate(article_sign_indices):
                body = ''
                if article_sign_indices[i] != article_sign_indices[-1]:
                    for j in range(article_sign_indices[i] + 1, article_sign_indices[i + 1]):
                        line = cleaned_law_lines[j]
                        body = " ".join([body, line])
                else:
                    for j in range(article_sign_indices[i] + 1, len(cleaned_law_lines)):
                        line = cleaned_law_lines[j]
                        body = " ".join([body, line])
                article_bodies.append(body)

            # Splitting text for chunking
            raw_chunks = []
            chunk_article_signs = []
            chunk_article_heads = []
            for i, body in enumerate(article_bodies):
                body_tokens = body.split()
                if len(body_tokens) > 150:
                    body_segments = re.split(r"\(\d+\)", body)[1:]
                    for _, body_segment in enumerate(body_segments):
                        segment_tokens = body_segment.split()
                        if len(segment_tokens) > 150:
                            segment_parts = body_segment.strip().split(':')

                            primary_part = segment_parts[0]
                            secondary_parts = segment_parts[1].split(';')

                            segment_parts_recombined = [f"{primary_part.strip()} {secondary_part.strip()}"
                                                        for secondary_part in secondary_parts]

                            for _, raw_chunk in enumerate(segment_parts_recombined):
                                if "(Brisano)" not in raw_chunk:
                                    raw_chunks.append(raw_chunk)
                                    chunk_article_signs.append(article_signs[i])
                                    chunk_article_heads.append(article_heads[i])

                        else:
                            if "(Brisano)" not in body_segment:
                                raw_chunks.append(body_segment)
                                chunk_article_signs.append(article_signs[i])
                                chunk_article_heads.append(article_heads[i])

                else:
                    if "(Brisano)" not in body:
                        raw_chunks.append(body)
                        chunk_article_signs.append(article_signs[i])
                        chunk_article_heads.append(article_heads[i])

            # Finishing chunks and base
            for i, raw_chunk in enumerate(raw_chunks):
                chunk = law_name + ' | ' + chunk_article_heads[i] + ' | ' + raw_chunk
                chunk = re.sub(' +', ' ', chunk)

                data = {"chunk": chunk,
                        "article": chunk_article_signs[i],
                        "law": law_name}

                df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)

    # Save the DataFrame as a CSV file with the specified base name
    df.to_csv(f"{new_base_name}.csv", index=False)

    return df

def embedding_text_with_bert(df, new_base_name):
    # Initialize the BERT tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-uncased")

    # Making list of chunks
    chunks = df["chunk"].tolist()

    # Embed the text using the BERT tokenizer
    # df["embeddings"] = df["chunk"].apply(lambda x: tokenizer.encode(x, add_special_tokens=True))
    embeddings = [tokenizer.encode(chunk, add_special_tokens=True) for chunk in chunks]

    # Perform zero padding on the embeddings
    max_length = max(len(x) for x in embeddings)
    padded_embeddings = [embedding + [0] * (max_length - len(embedding)) for embedding in embeddings]
    df["embedding"] = [frozenset(padded_embedding) for padded_embedding in padded_embeddings]

    # Reorder the columns in the dataframe
    df = df[["embedding", "chunk", "article", "law"]]

    # Save the updated dataframe to a new CSV file
    df.to_csv(f"{new_base_name}.csv", index=False)

    # Return the updated dataframe
    return df, padded_embeddings, max_length


# def get_embeddings_list(df):
#     embeddings_old_list = df['embeddings'].tolist()
#     embeddings_list = []
#     # for embedded_string in embeddings_old_list:
#     #     # Remove the brackets and split the string by commas
#     #     string = embedded_string.strip('[]')
#     #     items = string.split()
#     #     # Convert each item to its appropriate data type
#     #     items = [eval(item) for item in items]
#     #     # Append the list of items to the result list
#     #     embeddings_list.append(items)
#
#     for embedded_array in embeddings_old_list:
#         embedding = embedded_array.tolist()
#         embeddings_list.append(embedding)
#
#     vector_length = len(embeddings_list[0])
#
#     return embeddings_list, vector_length


def question_embedding(input_question, vector_length):
    # Initialize the BERT tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-uncased")

    # Perform BERT encoding on input question
    input_embedded = tokenizer.encode(input_question, add_special_tokens=True, return_tensors="np")[0]
    input_embedded = input_embedded.tolist()

    # Zero-pad embedded input question to match length of vectors in DataFrame
    input_embedded = input_embedded + [0] * (vector_length - len(input_embedded))

    return input_embedded


def find_similar_chunks(input_embedded, df, embeddings_list, metric, N=10):
    # Calculate distances between input_embedded and embeddings_list
    if metric == 'euclidean':
        distances = euclidean_distances([input_embedded], embeddings_list)[0]
    elif metric == 'manhattan':
        distances = manhattan_distances([input_embedded], embeddings_list)[0]
    elif metric == 'cosine':
        distances = cosine_distances([input_embedded], embeddings_list)[0]
    else:
        raise ValueError("Metric must be 'euclidean', 'manhattan', or 'cosine'.")

    # Get the indices of the N most similar embeddings
    similar_indices = np.argsort(distances)[:N]

    # Get the texts corresponding to the similar embeddings
    similar_chunks = df.loc[similar_indices, "chunk"].tolist()
    similar_articles = df.loc[similar_indices, "article"].tolist()
    similar_laws = df.loc[similar_indices, "law"].tolist()

    return similar_chunks, similar_articles, similar_laws


def deeppavlov_bert_qa_chatbot(question, similar_chunks):
    model_qa = build_model(configs.squad.qa_multisberquad_bert, download=True)

    context = ' '.join(similar_chunks)

    result = model_qa([context], [question])

    answer = result[0][0]
    answer_index = result[1][0]
    confidence = result[2][0]

    # Find the chunk index from which the answer is extracted
    chunk_index = -1
    for i, chunk in enumerate(similar_chunks):
        if answer_index >= context.find(chunk) and answer_index < context.find(chunk) + len(chunk):
            chunk_index = i
            break

    return answer, confidence, chunk_index

