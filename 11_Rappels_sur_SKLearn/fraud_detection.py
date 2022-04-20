import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

# Question 1: print(dataframe['Class'].value_counts())
# Il y a deux classes dans la database

# Question 2: print(dataframe.info())
# Il y a 31 caractéristique descriptive, toute de type 'float' ou 'int'

# Question 3: print(dataframe.shape)
# Il y a 284807 exemples dans la base

#Question 4: print(dataframe['Class'].value_counts())
# Il y a 284315 exemple de la classe '0', et 492 exemple de la classe '1'

#Question 5: 

def load_data(filename):
    dataframe = pd.read_csv(filename)
    return dataframe

def analyze_data(data):
    print(data.shape)
    data.info()
    print(data.describe())
    print(data.head())
    print(data['Class'].value_counts())

def show_matrix(data):
    plt.figure(figsize=(51,51))
    sns.heatmap(data.corr(), vmax=1, annot=false, square=true, cmap='RdY1Gn')
    plt.title('Correlation matrix')
    plt.show()

def split_data(data, ratio):
    train, test = train_test_split(data, test_size=ratio)
    
    x_train = train
    y_train = train['Class']
    del x_train['Class']

    x_test = test
    y_test = x_test['Class']
    del x_test['Class']

    return x_train, y_train, x_test, y_test

def create_model(classifier, x_train, y_train):
    classifier.fit(x_train, y_train)
    return classifier

def display_score(classifier, x_train, y_train, x_test, y_test):
    print("train score {}, test score {}".format(classifier.score(x_train, y_train), classifier.score(x_test, y_test)))
    y_pred = classifier.predict(x_test)
    target_names = ['Correction', 'Fraud']
    print(classification_report(y_test, y_pred, target_names=target_names))
    print(confusion_matrix(y_test, y_pred))

def remove_data(data, ratio):
    subset = data.loc[data.Class == 0]
    n = int(len(subset.index) * ratio)
    drop_index = np.random.choice(subset.index, n, replace=False)
    return data.drop(drop_index)

def duplicate_data(data, ratio):
        duplicate_rows = []
        subset = data.loc[data.Class == 1]
        subset_len = len(subset.index)
        n = int(subset_len * ratio)
        for _ in range(subset_len, n):
            duplicate_index = np.random.choice(subset.index, 1)
            duplicate_rows.append(subset.loc[duplicate_index])
        return data.append(duplicate_rows, ignore_index=True)

def main():
    data = load_data('creditcard.csv')

    data = duplicate_data(data, 40)
    # analyze_data(data)
    # show_matrix(data)
    x_train, y_train, x_test, y_test = split_data(data, 0.3)

    print('k plus proche voisins')
    classifier = KNeighborsClassifier()
    create_model(classifier, x_train, y_train)
    display_score(classifier, x_train, y_train, x_test, y_test)

if __name__ == '__main__':
    main()