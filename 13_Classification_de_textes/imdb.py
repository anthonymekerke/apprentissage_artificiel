import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from keras import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix, classification_report

def load_data(filename):
    dataframe = pd.read_csv(filename)

    return dataframe

def analyze_data(data):
    print('shape of dataframe:')
    print(data.shape)
    data.info()
    print(data.describe())
    print(data.head())
    print(data['sentiment'].value_counts())
    print(data['review'].value_counts())

def split_data(data, ratio):
    train, test = train_test_split(data, test_size=ratio)
    
    x_train = train['review']
    y_train = train['sentiment']

    x_test = test['review']
    y_test = test['sentiment']

    le = LabelEncoder()
    le.fit(y_train)
    y_train = le.transform(y_train)
    y_test = le.transform(y_test)

    return x_train, y_train, x_test, y_test

def create_vectors(x_train, x_test, vectorizer):
    x_train_vect = vectorizer.transform(x_train)
    x_test_vect = vectorizer.transform(x_test)

    return x_train_vect, x_test_vect

def create_model(x_train):
    model = Sequential()
    model.add(Dense(16, input_dim=x_train.shape[1], activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def show_results():
    #TODO
    return

def main():
    dataframe = load_data('films.csv')
    analyze_data(dataframe)

    x_train, y_train, x_test, y_test = split_data(dataframe, 0.3)

    vectorizer = CountVectorizer(min_df=0.01, max_df=1.0, 
                                stop_words='english', ngram_range=(1,1), 
                                analyzer='word')
    vectorizer.fit(x_train)
    x_train_vect, x_test_vect = create_vectors(x_train, x_test, vectorizer)
    #print(vectorizer.get_features_names()) # better with ngram_range=(2,2)

    model=create_model(x_train_vect)
    model.fit(x_train_vect, y_train, epochs=5,
            verbose=True, validation_data=(x_test_vect, y_test),
            batch_size=256)

    y_pred = model.predict(x_test_vect)
    y_pred = np.round(y_pred)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    print('TF-IDF')
    tf_idf_vectorizer = TfidfVectorizer(min_df=0.01, max_df=1.0, 
                                stop_words='english', ngram_range=(1,1), 
                                norm='l2', sublinear_tf=True)

    X_train_tfidf = tf_idf_vectorizer.fit_transform(x_train)
    X_test_tfidf = tf_idf_vectorizer.transform(x_test)
    model_tfidf = create_model(X_train_tfidf)

    model.fit(X_train_tfidf, y_train, epochs=5,
            verbose=True, validation_data=(X_test_tfidf, y_test),
            batch_size=256)

    y_pred = model_tfidf.predict(x_test_tfidf)
    y_pred = np.round(y_pred)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))


if __name__ == '__main__':
    main()