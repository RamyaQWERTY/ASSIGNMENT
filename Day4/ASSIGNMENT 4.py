#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import logging
from typing import List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from fastapi import FastAPI

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load the weather data
weather = pd.read_csv('weather_forecast (1).csv')

# Normalize the data
cols = ['precipitation', 'temp_max', 'temp_min', 'wind']
for col in cols:
    weather[col] = weather[col] / weather[col].max()

# Split the data into X and y
y = weather.pop('weather')
weather.pop('date')
X = weather

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Train the Random Forest Classifier
rf = RandomForestClassifier(bootstrap=False)
rf.fit(x_train, y_train)

# Initialize the FastAPI app
app = FastAPI()


@app.post("/predict_weather")
def predict_weather(inputs: List[float]):
    """
    Predict the weather using a trained Random Forest Classifier model.

    Args:
        inputs (List[float]): List of input values for precipitation, max temperature,
                              min temperature, and wind.

    Returns:
        str: Predicted weather category.

    Example:
        >>> predict_weather([0.0, 10.0, 2.8, 2.0])
        'sunny'
    """
    input_data = np.asarray(inputs)
    input_data_reshape = input_data.reshape(1, -1)
    prediction = rf.predict(input_data_reshape)
    return prediction[0]


if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


# In[ ]:


#this is the deployment link
http://localhost:8000/docs#/default/predict_weather_predict_weather_post

