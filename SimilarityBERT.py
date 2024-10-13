from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertModel
import torch
import torch.nn.functional as F

app = Flask(__name__)

# Load pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Ensure the model is in evaluation mode
model.eval()

# Utility function to get BERT embeddings for a sentence
def get_bert_embeddings(text):
    # Tokenize input text and convert to tensor
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    
    # Get the embeddings (the output is a tuple, we need the first element)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # The embeddings are in the last hidden state (first output in the tuple)
    # We'll average the token embeddings to get a single embedding for the sentence
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Average pooling across the tokens
    
    return embeddings

# Function to compute cosine similarity between two sentence embeddings
def cosine_similarity(embedding1, embedding2):
    # Cosine similarity using PyTorch's F.cosine_similarity function
    similarity = F.cosine_similarity(embedding1, embedding2)
    
    # Convert to a percentage similarity between 0 and 1
    return (similarity.item() + 1) / 2  # Scaling [-1, 1] to [0, 1]

# Define API route for calculating sentence similarity
@app.route('/sentence_similarity', methods=['POST'])
def sentence_similarity():
    try:
        # Get the JSON input
        data = request.json
        sentence1 = data.get('sentence1', '')
        sentence2 = data.get('sentence2', '')

        if not sentence1 or not sentence2:
            return jsonify({'error': 'Both sentence1 and sentence2 are required'}), 400

        # Get BERT embeddings for both sentences
        embeddings1 = get_bert_embeddings(sentence1)
        embeddings2 = get_bert_embeddings(sentence2)

        # Compute cosine similarity between the two embeddings
        similarity = cosine_similarity(embeddings1, embeddings2)

        # Return the similarity as a JSON response
        return jsonify({'similarity_percentage': similarity})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
