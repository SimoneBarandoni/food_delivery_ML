
from flask import Flask, request, jsonify
import random

app = Flask(__name__)

# Fake model function to predict the ratings and approval status
def fake_model(text):
    # Randomize food and delivery ratings (1 to 5)
    food_rating = random.randint(1, 5)
    delivery_rating = random.randint(1, 5)
    
    # Randomize approval status (0 or 1)
    approved = random.randint(0, 1)
    
    # Return predictions
    return {
        'food_rating': food_rating,
        'delivery_rating': delivery_rating,
        'approved': approved
    }

# API route to accept a test set (text) and return predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON data from the request
        data = request.json
        print(data)
        # Ensure that the request contains 'text'
        if 'text' not in data:
            return jsonify({'error': 'Invalid input, text field is missing'}), 400
        
        # Get the input text
        text = data['text']
        
        # Use the fake model to generate predictions
        prediction = fake_model(text)
        
        # Return the prediction as a JSON response
        return jsonify(prediction), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Main block to run the Flask app
if __name__ == '__main__':
    app.run(debug=True)

