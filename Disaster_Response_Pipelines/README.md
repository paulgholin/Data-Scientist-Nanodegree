Built ETL pipeline to clean data, ML pipeline to train a classifier, and deployed a web that can categorize text messages from a sender using the trained model.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data (disaster_messages.csv) and stores database (disaster_response.db)
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/disaster_response.db`
        
    - To run ML pipeline that trains classifier and saves model (classifier.pkl)
        `python models/train_classifier.py data/disaster_response.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
