# Ascend-Analytics DataScience Take-Home Assignment

## Background
In the California energy market (CAISO), we can sell energy produced by our solar farm into  
either the real-time (RT) market at spot prices or in the day-ahead (DA) market at prices determined
the day before. In order to decide which market to participate in, we want to make predictions for 
what the DA and RT prices will be. The DA market for tomorrow closes at 8am of the current day. 
We want to submit our bids at 7am to be sure we are on time. 
As such, by 7am we need predictions for what the DA and RT prices will be the 
following day. If we predict that the average RT price will be higher than DA in a given hour, we would like
to sell into the RT market. If we predict that DA will be greater than RT, we would like to sell into DA. 
This is a very challenging problem because we can only fit a model based on the data
we have available at 7am, and need to predict up to 11pm of the following day. 
That's (24 - 7) + 24 = 41 hours in advance. Yikes! 

## NOTE!!
You will not be evaluated on the performance of your ML model. Do not spend ANY time doing additional
feature engineering or tuning the model. You do NOT need to have a separate validation data-set. 
We simply want to expose you to the types of problems we solve at Ascend so that we can ask follow-up 
questions during the interview. 

DO NOT SPEND MORE THAN 1 HOUR ON THIS ASSIGNMENT.

## Task
It is 7am on 8/19/2023. "**raw_data.csv**" has the data that was available at this runtime. 
We want to predict the DA prices for each hour of 8/20. (don't worry about predicting the RT prices for now)

1. Add an appropriate lag to each column in "**raw_data.csv**" so that it can be to used predict tomorrow's DA prices WITHOUT DATA LEAKAGE.
2. Use a basic `RandomForestRegressor` to predict DA prices on 8/20
3. Visualize the results. The true values are in "**y_true.csv**"

The file "**take_home_assignment.py**" has code to get you started. You can use any IDE (or a Jupyter Notebook) to add your code. 

## During the interview
Be prepared to share your screen with your code and results. Be ready to run your code, we'll be making updates during the interview.
 - 5 minutes: Introductions
 - 30-45 minutes: Live Coding -> additional questions about this problem, and potentially other small problems depending on time. 
 - remaining time: Software Design -> non-coding ML software design/infrastructure questions (no need to study, we'll adjust these based on your experience level)

