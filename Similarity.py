import warnings
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import numpy as np

warnings.simplefilter(action='ignore', category=FutureWarning)

app = Flask(__name__)

class SentenceSimilarity:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def calculate_similarity(self, sentence1, sentence2):
        # Encode both sentences
        embeddings = self.model.encode([sentence1, sentence2])

        # Calculate cosine similarity between the two sentences
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )

        return float(similarity)

similarity_calculator = SentenceSimilarity()

@app.route('/similarity', methods=['POST'])
def calculate_similarity():
    data = request.get_json()
    sentence1 = data['sentence1']
    sentence2 = data['sentence2']
    similarity = similarity_calculator.calculate_similarity(sentence1, sentence2)
    return jsonify({'similarity': similarity})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
