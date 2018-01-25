import nltk
import os
import time
from sklearn.datasets import load_files
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import cross_val_score

import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


def cross_validation(classifier, docs_train, y_train):
    accuracies = cross_val_score(estimator=classifier, X=docs_train, y=y_train, cv=10, n_jobs=-1)
    return accuracies;


def accuracies_display(accuracies):
    average_accuracy = accuracies.mean()
    for a in accuracies:
        print a
    print ('Average: %r' % (average_accuracy * 100))


def preprocess(review_list):
    corpus = []
    for i in review_list:
        review = re.sub('[^a-zA-Z]', ' ', i)
        review = review.split()
        ps = PorterStemmer()
        review = [ps.stem(word) for word in review]  # if not word in set(stopwords.words('english'))
        review = ' '.join(review)
        corpus.append(review)

    return corpus


def main():
    movie_dir = str(raw_input("Enter the path of your data set (ex: /Users/susu/Desktop/SAGroup4/movie_reviews): "))

    if os.path.isdir(movie_dir):
        time.clock()
        # Importing the data set
        movie_train = load_files(movie_dir, shuffle=True)

        # Cleaning the texts
        corpus = preprocess(movie_train.data)

        # Creating the Bag of Words model
        # movie_vec = CountVectorizer(min_df=2, tokenizer=nltk.word_tokenize)
        movie_vec = CountVectorizer(min_df=2)
        # data set turned into sparse vector of word frequency counts
        movie_counts = movie_vec.fit_transform(corpus)
        # print movie_vec.get_feature_names()
        # print movie_counts.shape

        # Convert raw frequency counts into TF-IDF (Term Frequency -- Inverse Document Frequency) values
        tfidf_transformer = TfidfTransformer()
        movie_tfidf = tfidf_transformer.fit_transform(movie_counts)

        # Splitting the data set into Training set and Test set
        from sklearn.model_selection import train_test_split
        docs_train, docs_test, y_train, y_test = train_test_split(movie_tfidf, movie_train.target, test_size=0.20,
                                                                  random_state=12)
        # from sklearn.preprocessing import Normalizer
        # normalizer = Normalizer()
        # docs_train = normalizer.fit_transform(docs_train)
        # docs_test = normalizer.transform(docs_test)

        # Fitting Naive Bayes to the Training set
        from sklearn.naive_bayes import MultinomialNB
        naive_bayes_classifier = MultinomialNB().fit(docs_train, y_train)

        # Fitting LinearSVC to the Training set
        from sklearn.svm import SVC, LinearSVC, NuSVC
        svc_classifier = LinearSVC().fit(docs_train, y_train)

        # Fitting Logistic Regression to the Training set
        from sklearn.linear_model import LogisticRegression
        log_reg_classifier = LogisticRegression(random_state=0)
        log_reg_classifier.fit(docs_train, y_train)

        # Predicting the Test set results
        y_pred_naive_bayes = naive_bayes_classifier.predict(docs_test)
        y_pred_svc = svc_classifier.predict(docs_test)
        y_pred_log_reg = log_reg_classifier.predict(docs_test)

        # Making the Confusion Matrix
        naive_bayes_cm = confusion_matrix(y_test, y_pred_naive_bayes)
        svc_cm = confusion_matrix(y_test, y_pred_svc)
        log_reg_cm = confusion_matrix(y_test, y_pred_log_reg)

        print('- Confusion matrix: \n1. Multinominal Naive Bayes: \n%r, \n2. SVC: \n%r, \n3. Logistic Regression: \n%r'
              % (naive_bayes_cm, svc_cm, log_reg_cm))

        # Applying k-Fold Cross Validation
        print '\n- k-Fold Cross Validation:'
        print '1. Accuracy of Multinominal Naive Bayes'
        accuracies_display(cross_validation(naive_bayes_classifier, docs_train, y_train))

        print '\n2. Accuracy of SVM'
        accuracies_display(cross_validation(svc_classifier, docs_train, y_train))

        print '\n3. Accuracy of Logistic Regression'
        accuracies_display(cross_validation(log_reg_classifier, docs_train, y_train))

        print 'Time: ' + str(time.clock()) + ' sec'

        # Test
        reviews_new = []
        print '\n===FAKE MOVIE REVIEWS==='

        while True:
            review_new = str(raw_input("Enter movie reviews: "))
            if not review_new:
                continue
            else:
                break

        reviews_new.append(review_new)
        reviews_new_counts = movie_vec.transform(reviews_new)
        reviews_new_tfidf = tfidf_transformer.transform(reviews_new_counts)

        pred = svc_classifier.predict(reviews_new_tfidf)

        for review, category in zip(reviews_new, pred):
            print('%r => %s' % (review, movie_train.target_names[category]))

    else:
        print "Directory not exists."


if __name__ == '__main__':
    main()
