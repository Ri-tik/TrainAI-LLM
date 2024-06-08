from sentence_transformers import SentenceTransformer
import numpy as np
from trainaillm import GeminiModel

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def cosine_similarity(a, b):
    return np.dot(a, b.T) / (np.linalg.norm(a, axis=1)[:, np.newaxis] * np.linalg.norm(b, axis=1))

from trainaillm import UnstructuredDataLoader, split_into_segments

loader = UnstructuredDataLoader()
external_database = loader.get_data(folder_name='ICI')

external_database = loader.get_data(folder_name='ICI', file_name='ICI.txt')
chunks = split_into_segments(external_database[0]['data'])


embedded_data = model.encode(chunks)

# Open the text file
# file_path = "file.txt"  # Replace "file.txt" with the path to your text file
# with open(file_path, "r") as file:
#     # Read the lines of the file and store them in a list
#     external_data = file.readlines()
# embedded_data = model.encode(external_data)


queries = ["What is ICI?"]

embedded_queries = model.encode(queries)


Gemini = GeminiModel(api_key = "API_KEY", model_name = "gemini-1.0-pro")


for i, query_vec in enumerate(embedded_queries):
    # Compute similarities
    similarities = cosine_similarity(query_vec[np.newaxis, :], embedded_data)

    # Get top 3 indices based on similarities
    top_indices = np.argsort(similarities[0])[::-1][:3]
    top_doct = [chunks[index] for index in top_indices]

    # Print the top 3 similar sentences
    argumented_prompt = f"You are an expert question answering system, I'll give you question and context and you'll return the answer. Query : {queries[i]} Contexts : {top_doct[0]}"
    model_output = Gemini.generate_content(argumented_prompt)
    print(model_output)
    # If you're not importing the Gemini model from the trainaillm package
    # Use the following code to get the output
    # print(model_output.text)