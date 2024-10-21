# Food Delivery (Geckosoft Challenge)

### Assignment 1. ML model design and description.
The ML model should predict the food and delivery ratings based on the content of the textual comments shared by customers. Furthermore, it should
propose an approval or rejection, providing a numerical reliability estimate.
The customer reviews dataset presents a _Comment_ column, containing the text written by the user, _FoodRating_ and _DeliveryRating_ columns, containing the
ratings (1-5), and an _Approved_ column, containing a flag to indicate whether the comment has been approved (1) or rejected (0) by the moderation.

Since three different variables should be predicted, a multi-task learning model is needed. The model should rely on a shared text representation obtained through a pre-trained encoder (e.g., a BERT model, like bert-based-uncased), which tokenizes the textual comments and then generates the embeddings to capture the semantic meaning. Then, the model should have three heads, one for each specific task: for the Food Rating it could be a regression layer to predict the value of the rating (1-5), with Mean Squared Error as a Loss function; the same could be done for the Delivery Rating; for the Approval/Rejection task, a classification head might be included to predict the approval (1) or rejection (0) of the comment, with Binary Cross-Entropy as Loss function; this layer would provide the sigmoid output too, which can be interpreted as the model's confidence in the approval/rejection decision, as an estimation of the numerical reliability. 
