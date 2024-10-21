# Food Delivery (Geckosoft Challenge)

### Assignment 1. ML model design and description.
The ML model should predict the food and delivery ratings based on the content of the textual comments shared by customers. Furthermore, it should
propose an approval or rejection, providing a numerical reliability estimate.
The customer reviews dataset presents a _Comment_ column, containing the text written by the user, _FoodRating_ and _DeliveryRating_ columns, containing the
ratings (1-5), and an _Approved_ column, containing a flag to indicate whether the comment has been approved (1) or rejected (0) by the moderation.

Since three different variables should be predicted, a multi-task learning model is needed. The model should rely on a shared text representation obtained through a pre-trained encoder (e.g., a BERT model, like _bert-based-uncased_), which tokenizes the textual comments and then generates the embeddings to capture the semantic meaning. 
Then, the model should have three heads, one for each specific task:
1) For the Food Rating, it could be a regression layer to predict the value of the rating (1-5), with Mean Squared Error as a Loss function.
2) For the Delivery Rating, the layer could be the same of the Food Rating one;
3) For the Approval/Rejection task, a classification head might be included to predict the approval (1) or rejection (0) of the comment, with Binary Cross-Entropy as Loss function. This layer would provide the sigmoid output too, which can be interpreted as the model's confidence in the approval/rejection decision, as an estimation of the numerical reliability.

### Assignment 2. Mock training dataset.
To generate a mock dataset to train the model we could rely on Large Language Models and on Prompt Engineering.
We could start by writing a series of comments _manually_. These should mimic the actual texts and ratings written and shared by customers, and also consider the approval/rejection feature. In order to consider all the cases, we could ideate 25 comments, and assign them appropriate ratings, organised in five categories:
1) Fully positive: the comment is approved and the customer apreciated both the food and the delivery (the assigned ratings are 3, 4, or 5 for both of them). Example: "Amazing delivery speed and delicious food!". 
2) Positive food, negative delivery: the comment is approved and the customer apreciated the food, but not the delivery (the assigned ratings are 3, 4, or 5 for the food, 1 or 2 for the delivery). Example: "The food was great, but the delivery was late."
3) Negative food, positive delivery: the comment is approved and the customer apreciated the delivery, but not the food (the assigned ratings are 3, 4, or 5 for the delivery, 1 or 2 for the food). Example: "The delivery was perfectly on time, but I didn't like the food."
4) Fully negative: the comment is approved and the customer apreciated neither the food and the delivery (the assigned ratings are 1 or 2 for both of them). Example: "Terrible service, the delivery was delayed and food was bad."
5) Rejected: the comment contains elements which make them unacceptable for publication, such as inapropriate language (bad words) or spam (URLs). For these comments the ratings are not important, so they can be assigned randomly. Example: "Check out my stuff at http://www.example.com/channel".

Once these samples have been manually written, we could define a few-shot prompt to request the generation of more comments to a Large Language Models. The prompt should contain the description of the task (e.g., "generate 1000 customer reviews of the service of a food delivery company. Each review is composed of a textual comment, a rating 1-5 for the food, a rating 1-5 for the delivery, and a flag 0-1 to indicate whether the comment has been approved or rejected by a moderator. Rejected comments contain offensive or inapropriate language, or spam. The output should follow the format of a JSON...") and the 25 example (e.g., "the following are five examples of fully positive reviews: 'example1, example2, etc.', ...").
Regarding the choice of the LLM: assuming that we meet the technical requirements, we could deploy an open-source model locally, such as Llama 3 or Mixtral; otherwise, assuming that we have the budget, we could employ GPT-4 via APIs; for limited budget and computer infrasctructure we could rely on smaller models such as Gemma, Phi, Mistral, or on Quantization of larger models.

### Assignment 3. Model integration.
The service could be integrated into several applications by deploying the trained model and allow other systems interact with it via API calls. A solution could be using Flask, a Python framework for creating web applications. This would require the definition of API endpoints and of preprocessing functions (to prepare the input text into the format required by the model).
Another solution could be containerizing the model with Docker: this would allow the model to be deployed and used in any environment and without API calls.

### Assignment 4. Evaluation App Implementation.
The application to interact with the API of the machine learning model could be implemented in JavaScript (Node.js). The app should use a testset containing a series of comments along with ratings and approval to let the model predict these labels and then evaluate the prediction through Mean Absolute Error (for the regression tasks, the two ratings), and through Accuracy (for the classification task, the approval/reject).

The file _evaluate_model.js_ contains the code of the application. I used the _axios_ library to send POST requests to the model API, and the _csv-parser_ library to read the testset (test_reviews.csv). The two functions calculateMAE and calculateAccuracy are used to take the true values (from the testset) and the predicted values (received from the model via API call) and calculate Mean Absolute Error and Accuracy, respectively. The function evaluateModel reads the testset (test_reviews.csv) and, for each row of the dataset, takes the comment and make a POST request to the API. Then extracts the content of the received API response, storing the predicted values. After, these values are used to call the calculateMAE and calculateAccuracy functions and perform the evaluation.

In order to test the application, I created a Flask app, in the _model.py_ file. This contains a fake version of the machine learning model, which does not actually perform the regression and classification tasks. It takes the input text and returns three values, which are not the predictions but random numbers: two random number between 1 and 5, to be returned as predicted food rating and predicted delivery rating, and a random number between 0 and 1, to be returned as predicted approved/rejected. The file contains the function **predict()** to receive the JSON data from the request, use the fake model to generate fake predictions, and return them as JSON response. Once this app is launched through **python model.py**, a local environment is open on http://127.0.0.1:5000/. This is why in the Node.js file the apiUrl is **'http://localhost:5000/predict'**. The folder contains also the test_reviews.csv file containing the testset. 

node evaluate_model.js
aggiungere altre metriche oltre a accuracy, cercare anche qualcosaltro oltre a MAE
