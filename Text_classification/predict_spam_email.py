import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import string
import os
import pickle

from nltk.stem import SnowballStemmer, PorterStemmer
from nltk.corpus import stopwords
from textblob import TextBlob, Word

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm



file_path =  "/Users/macpro2elevenlab/Desktop/Data/spam.csv"
email_data = pd.read_csv(file_path, encoding='latin1')
email_data = email_data[['v1', 'v2']]
email_data = email_data.rename(columns={"v1": "target", "v2":"email"})
NAIVE_BAYES = "naive_bayes.checkpoint"
LINEAR = "linear_model.checkpoint"

def convert_to_dataframe(arr, name):
    """Convert raw data to dataframe type"""
    return pd.DataFrame({name:arr})

def processing_data(data):
    """Processing data"""
    # Define stopwords list
    stopwords_list = stopwords.words('english')
    # Define stemmer transform
    st = PorterStemmer()
    # Transform Up case to Low case
    data = data.apply(lambda x : " ".join(x.lower() for x in x.split()))
    # Remove stopwords
    data = data.apply(lambda x : " ".join(x for x in x.split() if x not in stopwords_list))
    # Stemming data
    data = data.apply(lambda x: " ".join([st.stem(word) for word in x.split()]))
    # Lemmatization
    data = data.apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

    return data

def convert_to_tfidf(data, convert_data):

    tfidf_vector = TfidfVectorizer(analyzer='word', token_pattern=r"\w{1,}", max_features=5000)
    tfidf_vector.fit(data)
    return tfidf_vector.transform(convert_data)


def train_model(model, name_model, tfidf_trainx, train_y, tfidf_vaildx, valid_y):

    # Encoder labels
    encoder = preprocessing.LabelEncoder()
    train_label = encoder.fit_transform(train_y)
    valid_label = encoder.fit_transform(valid_y)

    # train model
    model.fit(tfidf_trainx, train_label)

    # save model
    pickle.dump(model, open(name_model, "wb"))

    # predict model
    predictions = model.predict(tfidf_vaildx)

    # validation model
    score = metrics.accuracy_score(predictions, valid_label)

    return score

def predict_model(name_model, tfidf_testx):
    # Loading model
    model = pickle.load(open(name_model, "rb"))
    # Get predictions
    predictions = model.predict(tfidf_testx)
    # Setup results
    for prediction in predictions:
        if str(prediction) == "1":
            print("This Email is Spam Email")
        else:
            print(" This Email is Normal Email")



# processing dataset
email_data_processed = processing_data(email_data["email"])
# split dataset to train and vaildation
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(email_data_processed, email_data["target"])
# feature data
tfidf_trainx = convert_to_tfidf(email_data_processed, train_x)
tfidf_validx = convert_to_tfidf(email_data_processed, valid_x)

# test data
test_data = ["Get your FREE* $1500, Pottery Barn Gift Card!Please see here to claim your prize!",
             "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's",
             "Even my brother is not like to speak with me. They treat me like aids patent."]
# convert to dataframe
test_data_frame = convert_to_dataframe(test_data,"email")
# processing test data
test_data_processed = processing_data(test_data_frame["email"])
#  feature test data
tfidf_testx = convert_to_tfidf(email_data_processed, test_data_processed)

# naive bayes model
print("\nNaive bayes results : \n")
if os.path.exists(NAIVE_BAYES):
    prediction = predict_model(NAIVE_BAYES, tfidf_testx)
else:
    naive_bayes_model = train_model(naive_bayes.MultinomialNB(alpha=0.2), NAIVE_BAYES, tfidf_trainx,
                                    train_y, tfidf_validx, valid_y)
    print(naive_bayes_model)
    prediction = predict_model(NAIVE_BAYES, tfidf_testx)


# linear model
print("\nLinear model results : \n")
if os.path.exists(LINEAR):
    prediction = predict_model(LINEAR, tfidf_testx)
else:
    linear_classification_model = train_model(linear_model.LogisticRegression(), LINEAR,
                                              tfidf_trainx, train_y, tfidf_validx, valid_y)
    print(linear_classification_model)
    prediction = predict_model(LINEAR, tfidf_testx)
