# **Disaster Response Message Classification**
[link to github repo](https://github.com/gradinarualex/disaster-response-model)

Application that classifies messages related to disaster response from [appen](https://appen.com/datasets/combined-disaster-response-data/) and classifies them based on the type of information provided. This helps route these messages to the appropriate NGO handling every aspect of the crisis response. The model is trained on around 30,000 real messages during disasters such as floods in Pakistan in 2010, super-storm Sandy in U.S.A. in 2012, and many others. The online dashboard takes in messages and classifies them into multiple categories of the total 35.  


## **Getting Started**
Follow the instructions below to get this app up and running on your local machine.

#### **Prerequisites**
Python 3.6 or later needed on your local machine for the code to run properly.

#### **Installing**
Clone or fork this repository to your local machine.
Create a virtual environment to install dependencies in `requirements.txt` file:

```cli
pip install -r requirements.txt
```

After requirements are installed, you can just run the following command from the project root folder:

```cli
python app/run.py
```

To see the application, go to http://localhost:3001/ and you should see the app up and running.

## **Classification Approach**
Disaster messages fall under multiple categories, ranging from type of help requested (food, water, shelter, etc.), type of situation (refugee, missing people, death, etc.) to disaster type (storm, flood, earthquake, etc.), each with varying number of instances. This creates a data imbalance where some categories have high rate of occurance whereas some have very low occurance rate (200 out of 30.000 on fire-related messages). But occurance rate doesn't imply importance, with some rare occuring categories requiring fast action if true.
Disaster response is a sensitive endeavor, and it's better to have false alarms than to miss important information that could save lives. That's why we measure performance using both Precision and Recall (and therefore F1 Score). Recall penalizes false negatives and helps make sure that as few messages are not properly categorised as possible. Our final model has the highest score on Recall specifically for this reason.

#### **Model Description**
For this task we've tried multiple models, from [Random Forest Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html?highlight=random%20forest%20classifier#sklearn.ensemble.RandomForestClassifier), [AdaBoost Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html?highlight=adaboost#sklearn.ensemble.AdaBoostClassifier) and finally a [Gradient Boosting Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html?highlight=gradientboost#sklearn.ensemble.GradientBoostingClassifier) which proved to have the best overall performance.
In order to make it work on a multi-output model, we use the [MultiOutputClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html) class from [sklearn](https://scikit-learn.org/stable/).  

The best model (Gradient Boosting Classifier) had the following specifications:
- estimators: 100
- max depth: 8
- learning rate: 0.07

The features used to train the model are (with weight in brackets):
- transformed text data using tf-idf approach (90%)
- question mark count (2.5%)
- exclamation point count (2.5%)
- capital count (2.5%)
- word count (2.5%)


#### **Performance**
Best model reached a weighted average (across classes) Precision of 0.72, Recall of 0.61 (F1 = 0.64).

| Model | Fitting | Precision | Recall | F1 Precision |
| :---- | :---- | :-------: | :----: | :----------: |
| Random Forrest Classifier | grid search cross-validation | 0.78 | 0.47 | 0.53 |
| Random Forrest Classifier | previous + 4 extra features | 0.75 | 0.43 | 0.49 |
| AdaBoost Classifier | as previous | 0.73 | 0.59 | 0.64 |
| Gradient Boosting Classifier | as previous | 0.72 | 0.61 | 0.64 |


## **Licensing**
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details