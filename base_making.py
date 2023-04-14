import json
import pickle
import functions as fn

# Constants
LINE = 100*'-'
DOUBLE_LINE = 100*'='
PATH = "./Zakoni"

# Finding all pdf files on designated path
pdf_files = fn.pdf_finder(PATH)

# Conversation all pdf files to txt files on designated path
fn.pdf_to_txt(pdf_files)

# Making raw database from chunks of txt files
df_raw_base = fn.chunking_laws(path=PATH, new_base_name="raw_base")

# Making new database of raw database with BERT
df_tokenized_base, embeddings_list, vector_length = fn.embedding_text_with_bert(df=df_raw_base, new_base_name="tokenized_base")

# Saving parameters
data_dictionary = {"embeddings_list": embeddings_list, "vector_length": vector_length}

with open("parameters.pickle", "wb") as outfile:
    pickle.dump(data_dictionary, outfile)


