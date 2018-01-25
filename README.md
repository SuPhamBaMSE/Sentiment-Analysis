# Sentiment-Analysis
Group 4 - Su Pham Ba + Papa

I. SOFTWARE: Movie Reviews - Sentiment Analysis

Sentiment analysis or opinion mining is the computational study of people’s
opinions, appraisals, attitudes, and emotions toward entities, individuals,
issues, events, topics and their attributes. The task is technically challenging
and practically very useful. For example, businesses always want to find
public or consumer opinions about their products and services. Potential
customers also want to know the opinions of existing users before they use a
service or purchase a product.
A main topic in sentiment analysis is sentiment classification, which has been
studied extensively in the literature. In this task, we focus on classifying
an opinion document (e.g., a product review) as expressing a positive or
negative sentiment.

- Python classification of movie reviews (positive or negative) using NLTK-3 and sklearn.
- Data: use your own movie_review data set or download it in the nltk corpus

II. What is in our project
- Using sklearn
    + Naive Bayes: MultinomialNB
    + SVM: LinearSVC
    + Linear Model: LogisticRegression

we implemented in SentimentClassifier.py

III. REQUIREMENTS
1. Install packages and the dependencies
- scikit-learn
If you already have a working installation of numpy and scipy, the easiest way to install scikit-learn is using pip
pip install -U scikit-learn
or conda:
conda install scikit-learn

- nltk
sudo pip install -U nltk

- punkt
>>> import nltk
>>> nltk.download('punkt')

2. Downloading the dataset
The dataset used in this package is bundled along with the nltk package.
>>> import nltk
>>> nltk.download('movie_reviews')


IV. RUNNING IT

1. Enter the path of your data set (ex: /Users/susu/Desktop/SAGroup4/movie_reviews): your path
2. Enter movie reviews: enter a review in here
