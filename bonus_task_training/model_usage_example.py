from keras.models import load_model
import transformers
import numpy as np
from transformers import BertTokenizer

# Load the model from the file
loaded_model = load_model('food_delivery_model.h5', custom_objects={"TFBertModel": transformers.TFBertModel})

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Now you can use loaded_model to make predictions
def predict_ratings_and_approval(comment):
    # Tokenize and preprocess the comment
    inputs = tokenizer(comment, 
                       padding='max_length', 
                       truncation=True, 
                       max_length=128,  # Same as MAX_LEN used during training
                       return_tensors='tf')

    # Prepare the input data for the model
    input_ids = np.array(inputs['input_ids'])
    attention_mask = np.array(inputs['attention_mask'])

    # Make predictions
    predictions = loaded_model.predict({'input_ids': input_ids, 'attention_mask': attention_mask})

    # Extract predictions
    food_rating = np.clip(round(predictions[0][0][0]), 1, 5)  # Round and clip to range 1-5
    delivery_rating = np.clip(round(predictions[1][0][0]), 1, 5)  # Round and clip to range 1-5
    approval_probability = predictions[2][0][0]  # This is the approval output

    # Convert approval_probability to a binary prediction
    approval = 1 if approval_probability >= 0.5 else 0

    # Print results
    print("Predicted Food Rating: ", food_rating)
    print("Predicted Delivery Rating: ", delivery_rating)
    print(f"Predicted Approval: {'Approved' if approval == 1 else 'Rejected'}")
    print("Confidence in Approval: ", approval_probability)  # Confidence score

# Example usage
new_comment = "The delivery was good and the food was absolutely awesome!"
predict_ratings_and_approval(new_comment)
