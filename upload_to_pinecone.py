from pinecone import Pinecone
import openai
import pandas as pd
import os

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
INDEX_NAME = 'japanese-lessons'
OPENAI_API_KEY = 'os.environ.get("OPENAI_API_KEY")'
CSV_FILENAME = 'iml-course-lessons.csv'
SOURCE = 'iml-course-lessons'

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
client = openai.OpenAI(api_key=OPENAI_API_KEY)


df = pd.read_csv(CSV_FILENAME)

# Print available column names for debugging
print("CSV columns:", df.columns)


def get_embedding(text):

    response = client.embeddings.create(
        input=[text],
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

# Removed erroneous usage of 'row' outside the loop

vectors = []
for i, row in df.iterrows():
    id_value = str(row['id'])
    if not id_value or id_value.lower() == "nan":
        print(f"Skipping row {i} due to missing ID")
        continue

vectors = []
for i, row in df.iterrows():
    id_value = str(row['id'])
    if not id_value or id_value.lower() == "nan":
        print(f"Skipping row {i} due to missing ID")

vectors = []

for i, row in df.iterrows():
    id_value = str(row['id'])
    if not id_value or id_value.lower() == "nan":
        print(f"Skipping row {i} due to missing ID")
        continue
    embed_text = str(row['jp_text'])
    embedding = get_embedding(embed_text)
    metadata = row.to_dict()
    metadata['source'] = SOURCE
    vectors.append((id_value, embedding, metadata))

    if (i + 1) % 100 == 0 or (i + 1) == len(df):
        index.upsert(vectors)
        print(f"Upserted {i+1} vectors...")
        vectors = []



print("✅ All rows uploaded to Pinecone with embeddings!")





