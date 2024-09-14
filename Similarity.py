import warnings
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import numpy as np
import Levenshtein

warnings.simplefilter(action='ignore', category=FutureWarning)

app = Flask(__name__)

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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
