# -------------------------------------------------------------------------
# AUTHOR: Milosz Kryzia
# FILENAME: similarity.py
# SPECIFICATION: This program calculates the cosine similarity between text documents.
# FOR: CS 5990 (Advanced Data Mining) - Assignment #1
# TIME SPENT: 
# -----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy or pandas.
#You have to work here only with standard dictionaries, lists, and arrays

# Importing some Python libraries
import csv
from sklearn.metrics.pairwise import cosine_similarity

documents = []

#reading the documents in a csv file
with open('cleaned_documents.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         documents.append (row)

#Building the document-term matrix by using binary encoding.
#You must identify each distinct word in the collection using the white space as your character delimiter.
#--> add your Python code here
vocab = set()
for doc in documents:
    words = doc[1].split()
    vocab.update(words)

docTermMatrix = []

for doc in documents:
    doc_vector = [1 if word in doc[1].split() else 0 for word in vocab]
    docTermMatrix.append(doc_vector)

# Compare the pairwise cosine similarities and store the highest one
# Use cosine_similarity([X], [Y]) to calculate the similarities between 2 vectors
# --> Add your Python code here
similarities = {}
max_similarity = 0
most_similar_pair = None

for i in range(len(docTermMatrix)):
    for j in range(i + 1, len(docTermMatrix)):
        sim = cosine_similarity([docTermMatrix[i]], [docTermMatrix[j]])[0][0]
        similarities[(i+1, j+1)] = sim
        if sim > max_similarity:
            max_similarity = sim
            most_similar_pair = (i+1, j+1)


# Print the highest cosine similarity following the information below
# The most similar documents are document 10 and document 100 with cosine similarity = x
# --> Add your Python code here
if most_similar_pair:
    print(f"The most similar documents are document {most_similar_pair[0]} and document {most_similar_pair[1]} with cosine similarity = {max_similarity}")