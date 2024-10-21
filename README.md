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
5) Rejected: the comment contains elements which make them unacceptable for publication, such as inapropriate language (bad words) or spam (URLs). For these comments the ratings are not important, so they can be assigned randomly. Example: "Check out my stuff at http://www.example.com/channel"
Once these samples have been manually written, we could define a few-shot prompt to request the generation of more comments to a Large Language Models. The prompt should contain the description of the task (e.g., "generate 1000 customer reviews of the service of a food delivery company. Each review is composed of a textual comment, a rating 1-5 for the food, a rating 1-5 for the delivery, and a flag 0-1 to indicate whether the comment has been approved or rejected by a moderator. Rejected comments contain offensive or inapropriate language, or spam. The output should follow the format of a JSON...") and the 25 example (e.g., "the following are five examples of fully positive reviews: 'example1, example2, etc.', ...").
Regarding the choice of the LLM: assuming that we meet the technical requirements, we could deploy an open-source model locally, such as Llama 3 or Mixtral; otherwise, assuming that we have the budget, we could employ GPT-4 via APIs; for limited budget and computer infrasctructure we could rely on smaller models such as Gemma, Phi, Mistral, or on Quantization of larger models.

### Assignment 3. Model integration.
