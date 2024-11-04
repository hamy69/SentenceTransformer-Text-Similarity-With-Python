import warnings
from flask import Flask, request, jsonify
from werkzeug.exceptions import RequestTimeout
from threading import Thread
import time
from sentence_transformers import SentenceTransformer
import numpy as np
import Levenshtein
import pyodbc  # For SQL Server connection
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class TimeoutMiddleware:
    def __init__(self, app, timeout=60):
        self.app = app
        self.timeout = timeout

    def __call__(self, environ, start_response):
        def handle_request():
            self.result = self.app(environ, start_response)

        thread = Thread(target=handle_request)
        thread.start()
        thread.join(self.timeout)
        
        if thread.is_alive():
            raise RequestTimeout(f'Request timed out after {self.timeout} seconds')
        
        return self.result

warnings.simplefilter(action='ignore', category=FutureWarning)

app = Flask(__name__)
app.wsgi_app = TimeoutMiddleware(app.wsgi_app, timeout=360)

class SentenceSimilarity:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def calculate_SentenceTransformer_similarity(self, sentence1, sentence2):
        # Encode both sentences
        embeddings = self.model.encode([sentence1, sentence2])

        # Calculate cosine similarity between the two sentences
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )

        return float(similarity)
    
    def calculate_Levenshtein_distance(self, sentence1, sentence2):
        distance = Levenshtein.distance(sentence1, sentence2)
        max_len = max(len(sentence1), len(sentence2))
        return 1.0 - (distance / max_len)

    def calculate_Jaccard_similarity(self, sentence1, sentence2):
        set1 = set(sentence1.split())
        set2 = set(sentence2.split())
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        return len(intersection) / len(union) if union else 0.0
    
    def calculate_similarities(self, sentences, single_sentence):
        similarities = [
            {
                'id': sentence['id'],
                'sentence': sentence['sentence'],
                'similarity':
                    {
                    'SentenceTransformer': self.calculate_SentenceTransformer_similarity(sentence['sentence'], single_sentence),
                    'LevenshteinDistance': self.calculate_Levenshtein_distance(sentence['sentence'], single_sentence),
                    'Jaccard': self.calculate_Jaccard_similarity(sentence['sentence'], single_sentence),
                    }
            }
            for sentence in sentences
        ]
        return similarities
    
    def fetch_similar_sentences_from_db(self, target_sentence, connection_string, query, levenshteinThreshold=0.3, similarityThreshold=0.9):
        """
        Connects to the SQL Server, retrieves sentences from the specified table,
        and returns those with a similarity above the given threshold.
        
        :param target_sentence: The sentence to compare against.
        :param connection_string: The connection string for the SQL Server.
        :param levenshteinThreshold: The levenshtein distance threshold for cut data from and reduce time prosses in the transformer similarity (default is 0.3).
        :param similarityThreshold: The transformer similarity threshold (default is 0.9).
        :return: List of sentences with similarity > similarityThreshold.
        """
        try:
            # Connect to the SQL Server
            conn = pyodbc.connect(connection_string)
            cursor = conn.cursor()
            
            # Fetch all sentences from the specified table
            cursor.execute(query)
            rows = cursor.fetchall()

            # Filter sentences based on the similarity threshold
            similar_sentences = []
            for row in rows:
                id, sentence = row.Id, row.Sentence
                Levenshtein_distance = self.calculate_Levenshtein_distance(target_sentence, sentence)
                if Levenshtein_distance > levenshteinThreshold:
                    similarity = self.calculate_SentenceTransformer_similarity(target_sentence, sentence)
                    if similarity > similarityThreshold:
                        similar_sentences.append({'Id': id, 'Sentence': sentence})

            # Close the connection
            cursor.close()
            conn.close()

            return similar_sentences
        
        except Exception as e:
            return jsonify({'error': str(e)}), 500

similarity_calculator = SentenceSimilarity()

@app.route('/similarity', methods=['POST'])
def calculate_similarity():
    data = request.get_json()
    sentence1 = data['sentence1']
    sentence2 = data['sentence2']
    similarity = similarity_calculator.calculate_SentenceTransformer_similarity(sentence1, sentence2)
    return jsonify({'similarity': similarity})

@app.route('/similarities', methods=['POST'])
def calculate_similarities():
    data = request.get_json()
    sentences = data['sentences']
    single_sentence = data['single_sentence']
    SimilarityResponse = similarity_calculator.calculate_similarities(sentences, single_sentence)
    return jsonify(SimilarityResponse)

@app.route('/ProductSimilarities', methods=['Get'])
def calculate_ProductSimilarities():
    data = request.get_json()
    target_sentence = data['target_sentence']
    similarityThreshold = data['similarityThreshold']
    levenshteinThreshold = data['levenshteinThreshold']
    connection_string = os.getenv('DB_CONNECTION_STRING')
    query = os.getenv('DB_Product_QUERY')  # Adjust 'sentence' to actual column name
    SimilarityResponse = similarity_calculator.fetch_similar_sentences_from_db(target_sentence, connection_string, query, levenshteinThreshold, similarityThreshold)
    # Map the response to include 'ID' and 'Name_PRD' keys
    formatted_response = [{'ID': item['Id'], 'Name_PRD': item['Sentence']} for item in SimilarityResponse]
    return jsonify(formatted_response)

if __name__ == '__main__':
     app.run(debug=True, host='0.0.0.0', port=5000)
