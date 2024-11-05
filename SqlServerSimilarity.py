"""
This module implements a Flask web application that provides endpoints for calculating sentence similarities
using various methods, including Sentence Transformers, Levenshtein distance, and Jaccard similarity. 
It also includes functionality to fetch similar sentences from a SQL Server database.

Dependencies:
- Flask: A web framework for building web applications.
- Sentence Transformers: A library for sentence embeddings.
- NumPy: A library for numerical operations.
- Levenshtein: A library for calculating Levenshtein distance.
- PyODBC: A library for connecting to SQL Server databases.
- dotenv: A library for loading environment variables from a .env file.

Classes:
- TimeoutMiddleware: Middleware to handle request timeouts.
- SentenceSimilarity: Class for calculating sentence similarities.

Endpoints:
- /similarity (POST): Calculates the similarity between two sentences.
- /similarities (POST): Computes similarity scores for a list of sentences compared to a single sentence.
- /ProductSimilarities (GET): Fetches similar sentences from the database based on a target sentence and specified thresholds.

Usage:
1. Set up the environment variables in a .env file for database connection.
2. Run the Flask application.
3. Use the provided endpoints to calculate similarities or fetch similar sentences.

Example:
- To calculate similarity between two sentences, send a POST request to /similarity with JSON body:
  {
      "sentence1": "Your first sentence.",
      "sentence2": "Your second sentence."
  }

- To fetch similar sentences from the database, send a GET request to /ProductSimilarities with JSON body:
  {
      "target_sentence": "Your target sentence.",
      "similarityThreshold": 0.9,
      "levenshteinThreshold": 0.3
  }
"""
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
        """
        Initializes the TimeoutMiddleware with a specified timeout.
        
        Args:
            app (Flask): The Flask application to wrap.
            timeout (int): The timeout duration in seconds (default is 60).
        """
        self.app = app
        self.timeout = timeout

    def __call__(self, environ, start_response):
        """
        Handles the request and applies the timeout logic.
        
        Args:
            environ (dict): The WSGI environment.
            start_response (callable): The WSGI start response callable.
        
        Returns:
            The result of the application call or raises a RequestTimeout exception if the timeout is exceeded.
        """
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
        """
        Initializes the SentenceSimilarity class and loads the Sentence Transformer model.
        """
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def calculate_SentenceTransformer_similarity(self, sentence1, sentence2):
        """
        Calculates the cosine similarity between two sentences using a Sentence Transformer model.
            
        Args:
            sentence1 (str): The first sentence to compare.
            sentence2 (str): The second sentence to compare.
            
        Returns:
            float: A float representing the cosine similarity score between the two sentences.
        """
        # Encode both sentences
        embeddings = self.model.encode([sentence1, sentence2])

        # Calculate cosine similarity between the two sentences
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )

        return float(similarity)
    
    def calculate_Levenshtein_distance(self, sentence1, sentence2):
        """
        Computes the normalized Levenshtein distance between two sentences.
            
        Args:
            sentence1 (str): The first sentence to compare.
            sentence2 (str): The second sentence to compare.
            
        Returns:
            float: A float representing the similarity score, where 1.0 indicates identical sentences.
        """
        distance = Levenshtein.distance(sentence1, sentence2)
        max_len = max(len(sentence1), len(sentence2))
        return 1.0 - (distance / max_len)

    def calculate_Jaccard_similarity(self, sentence1, sentence2):
        """
        Calculates the Jaccard similarity coefficient between two sentences.
            
        Args:
            sentence1 (str): The first sentence to compare.
            sentence2 (str): The second sentence to compare.
            
        Returns:
            float: A float representing the Jaccard similarity score, where 1.0 indicates identical sets of words.
        """
        set1 = set(sentence1.split())
        set2 = set(sentence2.split())
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        return len(intersection) / len(union) if union else 0.0
    
    def calculate_similarities(self, sentences, single_sentence):
        """
        Computes similarity scores for a list of sentences compared to a single sentence.
            
        Args:
            sentences (list[dict]): A list of dictionaries containing sentences and their IDs.
            single_sentence (str): The sentence to compare against.
            
        Returns:
            list[dict]: A list of dictionaries with IDs, sentences, and their similarity scores.
        """
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
            
        Args:
            target_sentence (str): The sentence to compare against.
            connection_string (str): The connection string for the SQL Server.
            query (str): The SQL query to fetch sentences.
            levenshteinThreshold (float): The threshold for Levenshtein distance, where that reduce time prosses in the transformer similarity by cut data (default is 0.3).
            similarityThreshold (float): The threshold for Sentence Transformer similarity (default is 0.9).
            
        Returns:
            list[dict]: A list of sentences with similarity above the specified threshold.
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
    """
    Endpoint to calculate the similarity between two sentences.
    
    This function retrieves two sentences from the request body, calculates their similarity
    using the Sentence Transformer model, and returns the similarity score as a JSON response.
    
    Returns:
        JSON response containing the similarity score.
    """
    data = request.get_json()
    sentence1 = data['sentence1']
    sentence2 = data['sentence2']
    similarity = similarity_calculator.calculate_SentenceTransformer_similarity(sentence1, sentence2)
    return jsonify({'similarity': similarity})

@app.route('/similarities', methods=['POST'])
def calculate_similarities():
    """
    Endpoint to compute similarity scores for a list of sentences compared to a single sentence.
    
    This function retrieves a list of sentences and a single sentence from the request body,
    calculates their similarity scores using various methods, and returns the results as a JSON response.
    
    Returns:
        JSON response containing a list of similarity scores for each sentence.
    """
    data = request.get_json()
    sentences = data['sentences']
    single_sentence = data['single_sentence']
    SimilarityResponse = similarity_calculator.calculate_similarities(sentences, single_sentence)
    return jsonify(SimilarityResponse)

@app.route('/ProductSimilarities', methods=['Get'])
def calculate_ProductSimilarities():
    """
    Endpoint to fetch similar sentences from the database based on a target sentence and specified thresholds.
    
    This function retrieves the target sentence and similarity thresholds from the request body,
    connects to the database, fetches similar sentences, and returns them as a JSON response.
    
    Returns:
        JSON response containing a list of similar sentences from the database, formatted with 'ID' and 'Name_PRD' keys.
    """
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
    """
    Runs the Flask application.
    
    This function starts the Flask web server, allowing it to listen for incoming requests
    on the specified host and port.
    """
    app.run(debug=True, host='0.0.0.0', port=5000)
