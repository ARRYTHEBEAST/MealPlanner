import os
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from dotenv import load_dotenv
import json
from tqdm import tqdm  # Importing tqdm for progress bar
import config.py
# Path to your JSON file
FILE_PATH = r"/Users/arjun/Documents/StreamlitChatbot/dubhacks24-RAG-workshop/data/recipes.json"

print(f"Uploading {FILE_PATH} to ChromaDB")

load_dotenv()

# Set up the embedding function
google_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
    api_key=(config.GOOGLE_API_KEY)
)
client = chromadb.PersistentClient(path="./data/vectorDB")

# Create collection if it doesn't exist already
collection = client.create_collection(
    name="my_collection",
    embedding_function=google_ef,
    metadata={"hnsw:space": "cosine"},
)

# Load the JSON data
with open(FILE_PATH, 'r') as json_file:
    data = json.load(json_file)

documents = []
ids = []

# Process each recipe with a progress bar
for index, recipe in tqdm(enumerate(data["recipes"]), total=len(data["recipes"]), desc="Processing recipes"):
    # Create a formatted string for the recipe
    ingredients_list = ', '.join([ingredient['name_raw'] for ingredient in recipe["ingredients"]])
    
    formatted_recipe = (
        f"Recipe Name: {recipe['name']}\n"
        f"Time (mins): {recipe['time(mins)']}\n"
        f"Rating: {recipe['rating']}\n"
        f"Ingredients: {ingredients_list}\n"
    )
    
    documents.append(formatted_recipe)
    ids.append(f"recipe_{index + 1}")

# Add documents to the collection in ChromaDB
print("Recipes indexed, uploading to ChromaDB...")
print("This may take a minute...")

# Upload the documents in batches with a progress bar
batch_size = 10  # Set the batch size for uploading (you can adjust this)
for i in tqdm(range(0, len(documents), batch_size), desc="Uploading to ChromaDB"):
    collection.add(documents=documents[i:i+batch_size], ids=ids[i:i+batch_size])

print("Upload complete!")
print(f"{collection.count()} documents in collection.")
