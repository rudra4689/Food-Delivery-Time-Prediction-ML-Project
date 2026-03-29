# Food Delivery Time Prediction ML Project

## Problem Statement

Predict the time taken (in minutes) to deliver a food order using delivery partner details, order type, vehicle type, and restaurant–customer location data.

## Data Used

https://www.kaggle.com/datasets/bhanupratapbiswas/food-delivery-time-prediction-case-study

To build the prediction model, the company collects historical delivery data, including the following features:

1. ID : Order Id 
2. Delivery_person_ID, Delivery_person_Age,	Delivery_person_Ratings
3. Resturant and Delivery coordinates:
   Restaurant_latitude,Restaurant_longitude,Delivery_location_latitude, Delivery_location_longitude	
4. Type_of_order: Drink, Meals, Buffet, Snacks
5. Type_of_vehicle: Scotter, Motorcycle, Electric Scotter, Bicycle
6. Time_taken(min): Predicted value

## EDA Insights

1. Data Cleaning: Removed outliers present 
2. Extracted importanted features : Distance, Cooking time (waiting time for each
   resturant based on type of order)
3. Distance vs Time Taken Graph
4. Rating vs Time Taken Graph
5. Age Distribution
6. Plotted box plot for type of order vs time taken to find any outliers are present
7. Plotted box plot for type of vehicle vs time taken to find any outliers are present
8. Found Distance and Time are linearly correlated
9. Plotted heatmap to drop highly coorelated varaibles
10. Dropped Unimportant features

## Model Building

1. Linear Regression
2. Random Forest Regressor
3. Support Vector Regressor
4. XgBoost

Data is splitted in to train and test samples and Trained the data by using above models.
Calculated R2 score, MSE, RMSE, MAE
Scaling of data is done while applying SVR for better results. 
Out of all the models mentioned XgBoost is giving better results. 

## Deployment

1. Downloaded pickle file
2. Written code for app.py to deploy the model in Hugging Face
3. Created requirements.txt file
