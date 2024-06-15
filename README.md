# recsystemdraft
Draft for run specialty recommendation system

Code flow:

app.py --> initializes local web page
import logic --> this custom module contains all of the thinking/machinelearning/recommendation system

<strong> logic.get --> </strong> this returns a list of all customer data, which includes product recommendations. 
logic.postprocess --> this appends relevant customer data to a local csv file for later processing (in deployment, there would be a separate function to go through and evaluate the accuracy of the model's predictions throughout the sales day) 
