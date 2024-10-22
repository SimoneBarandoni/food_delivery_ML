const axios = require('axios');
const fs = require('fs');
const csv = require('csv-parser');

// Define API endpoint (Flask service running locally at port 5000)
const apiUrl = 'http://localhost:5000/predict';

// Arrays to store true and predicted values
let trueFoodRatings = [];
let predictedFoodRatings = [];
let trueDeliveryRatings = [];
let predictedDeliveryRatings = [];
let trueApprovals = [];
let predictedApprovals = [];

// Function to calculate Mean Absolute Error (MAE)
function calculateMAE(trueValues, predictedValues) {
    let totalError = 0;
    for (let i = 0; i < trueValues.length; i++) {
        totalError += Math.abs(trueValues[i] - predictedValues[i]);
    }
    return totalError / trueValues.length;
}

// Function to calculate accuracy for binary classification
function calculateAccuracy(trueLabels, predictedLabels) {
    let correct = 0;
    for (let i = 0; i < trueLabels.length; i++) {
        if (trueLabels[i] === predictedLabels[i]) {
            correct++;
        }
    }
    return correct / trueLabels.length;
}

// Function to process the CSV test dataset and evaluate the model
function evaluateModel() {
    const promises = []; // Array to store axios promises

    fs.createReadStream('data//test_reviews.csv')
        .pipe(csv())
        .on('data', (row) => {
            const comment = row.comment;

            // Push axios POST requests (promises) into the array
            const promise = axios.post(apiUrl, { text: comment })
                .then((response) => {
                    const prediction = response.data;

                    // Store true and predicted values for food rating, delivery rating, and approval
                    trueFoodRatings.push(parseFloat(row.true_food_rating));
                    predictedFoodRatings.push(prediction.food_rating);

                    trueDeliveryRatings.push(parseFloat(row.true_delivery_rating));
                    predictedDeliveryRatings.push(prediction.delivery_rating);

                    trueApprovals.push(parseInt(row.true_approval));
                    predictedApprovals.push(prediction.approved);
                })
                .catch((error) => {
                    console.error(`Error processing review: ${comment}`);
                    console.error(error.message);
                });

            // Add the promise to the array
            promises.push(promise);
        })
        .on('end', () => {
            // Once all rows have been processed, wait for all axios requests to complete
            Promise.all(promises)
                .then(() => {
                    // Calculate and log performance metrics
                    console.log('Finished processing test data. Calculating performance metrics...');
                    // console.log(trueFoodRatings);
                    // console.log(predictedFoodRatings);
                    // console.log(trueDeliveryRatings);
                    // console.log(predictedDeliveryRatings);
                    // console.log(trueApprovals);
                    // console.log(predictedApprovals);
                    const foodRatingMAE = calculateMAE(trueFoodRatings, predictedFoodRatings);
                    const deliveryRatingMAE = calculateMAE(trueDeliveryRatings, predictedDeliveryRatings);
                    const approvalAccuracy = calculateAccuracy(trueApprovals, predictedApprovals);

                    console.log(`Mean Absolute Error (Food Rating): ${foodRatingMAE}`);
                    console.log(`Mean Absolute Error (Delivery Rating): ${deliveryRatingMAE}`);
                    console.log(`Approval Accuracy: ${(approvalAccuracy * 100).toFixed(2)}%`);

                    // Write the results to a .txt file
                    const output = `Mean Absolute Error (Food Rating): ${foodRatingMAE}
                    Mean Absolute Error (Delivery Rating): ${deliveryRatingMAE}
                    Approval Accuracy: ${(approvalAccuracy * 100).toFixed(2)}%`;

                    fs.writeFile('performance_metrics.txt', output, (err) => {
                        if (err) {
                            console.error('Error writing to file', err);
                        } else {
                            console.log('Performance metrics saved to performance_metrics.txt');
                        }
                    });
                })
                .catch((error) => {
                    console.error('Error completing requests:', error.message);
                });
        });
}

// Start evaluation
evaluateModel();
