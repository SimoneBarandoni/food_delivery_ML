import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset (comments, food_rating, delivery_rating, approval)
data = pd.read_csv(r'data\training_data.csv') 

# Define parameters
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 2e-5

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenization
def preprocess_texts(texts):
    # Tokenize the texts and pad/truncate to MAX_LEN
    encoded = tokenizer(list(texts), 
                        padding='max_length', 
                        truncation=True, 
                        max_length=MAX_LEN, 
                        return_tensors='tf')
    return {
        'input_ids': np.array(encoded['input_ids']), 
        'attention_mask': np.array(encoded['attention_mask'])
    }

# Tokenize the comments
X = preprocess_texts(data['comment'])

# Prepare the target variables
y_food = data['food_rating'].astype(np.float32)
y_delivery = data['delivery_rating'].astype(np.float32)
y_approval = data['approval'].astype(np.float32)

# Ensure that the number of samples matches the dataset rows
assert len(X['input_ids']) == len(data), "Mismatch in the number of samples after preprocessing."

# Split data into train and test sets
X_train_ids, X_test_ids, X_train_mask, X_test_mask, y_food_train, y_food_test, y_delivery_train, y_delivery_test, y_approval_train, y_approval_test = train_test_split(
    X['input_ids'], X['attention_mask'], y_food, y_delivery, y_approval, test_size=0.2, random_state=42
)

# Load the pre-trained BERT model
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

# Define the multi-task model
def create_multi_task_model():
    # Input layer for tokenized text
    input_ids = tf.keras.Input(shape=(MAX_LEN,), dtype=tf.int32, name='input_ids')
    attention_mask = tf.keras.Input(shape=(MAX_LEN,), dtype=tf.int32, name='attention_mask')

    # BERT encoder (shared representation)
    bert_output = bert_model(input_ids, attention_mask=attention_mask)
    pooled_output = bert_output.pooler_output  # [CLS] token representation

    # Shared dropout layer for regularization
    x = Dropout(0.3)(pooled_output)

    # Food rating prediction head (regression)
    food_output = Dense(64, activation='relu')(x)
    food_output = Dense(1, name='food_rating')(food_output)

    # Delivery rating prediction head (regression)
    delivery_output = Dense(64, activation='relu')(x)
    delivery_output = Dense(1, name='delivery_rating')(delivery_output)

    # Approval prediction head (binary classification)
    approval_output = Dense(64, activation='relu')(x)
    approval_output = Dense(1, activation='sigmoid', name='approval')(approval_output)

    # Multi-task model
    model = Model(inputs=[input_ids, attention_mask], 
                  outputs=[food_output, delivery_output, approval_output])

    return model

# Create the model
model = create_multi_task_model()

# Compile the model
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
              loss={'food_rating': 'mse', 
                    'delivery_rating': 'mse', 
                    'approval': 'binary_crossentropy'},
              metrics={'food_rating': 'mae', 
                       'delivery_rating': 'mae', 
                       'approval': 'accuracy'})

# Prepare input dictionaries for training and testing
train_inputs = {'input_ids': X_train_ids, 'attention_mask': X_train_mask}
test_inputs = {'input_ids': X_test_ids, 'attention_mask': X_test_mask}

# Train the model
history = model.fit(train_inputs, 
                    {'food_rating': y_food_train, 
                     'delivery_rating': y_delivery_train, 
                     'approval': y_approval_train},
                    validation_data=(test_inputs, 
                                     {'food_rating': y_food_test, 
                                      'delivery_rating': y_delivery_test, 
                                      'approval': y_approval_test}),
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS)

results = model.evaluate(test_inputs, 
                         {'food_rating': y_food_test, 
                          'delivery_rating': y_delivery_test,
                          'approval': y_approval_test})

# Print the evaluation results
print(f"Evaluation Results: {results}")

model.save('food_delivery_model.h5')
