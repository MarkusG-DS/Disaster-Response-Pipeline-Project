# Disaster Response Pipeline Project

In this project I build a Natural Language Processing model (NLP) to categorize messages related to Disasters on real time basis. The Dataset for this project is provided by FigureEight and Udacity. It contains Twitter and messaging data from real-world disaster events.

## 1. Installations

The necessary libraries to run the code beyond the Anaconda distribution of Python Version 3.* are:
Machine Learning Libraries: pandas, numpy, scipy, sklearn, matplotlib
NLP: nltk
Database: sqlalchemy (for SQLlite Database)
Web App: Flast, plotly
Other: pickle

## 2. Project Motivation

This is the second project of the Udacity Data Science Nanodegree Program.
In this project I build a Natural Language Processing model (NLP) to categorize messages related to Disasters on real time basis. The Dataset for this project is provided by FigureEight and Udacity. It contains Twitter and messaging data from real-world disaster events.

## 4. How to Interact with your project

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## 5. Licensing, Authors, Acknowledgements, etc.

FigureEight - providing relevant dataset for this project
UDACITY
