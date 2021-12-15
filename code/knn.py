import os
from collections import Counter
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import imblearn
import sklearn
from imblearn import under_sampling, over_sampling
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(data):
    # Create a Dict containing List of Values
    data_frame = defaultdict(list)
    # Go through all the classes in the data dictionary
    for key in data.keys():
        # Read all the files associated with the class
        for file in os.listdir(data[key]):
            with open(data[key]+file) as fr:
                # Read the text and covert it into list of integers
                wave = [int(i) for i in fr.read().split('\n') if i not in ['']]
                num=int(len(wave)-1)
                for j in range(len(wave)-1):
                    name='z'+str(j)
                    val=wave[j]
                    data_frame[name].append(val)
            data_frame['target'].append(key)
    # return data_frame
    return pd.DataFrame(data_frame)

if __name__ == '__main__':

    data = {0:"C:/Users/ASUS/Desktop/EEG Data/set a/", 1:"C:/Users/ASUS/Desktop/EEG Data/set e/"}
    df = load_data(data)
    print(df.shape)
    final = sklearn.utils.shuffle(df, random_state=1)
    x = df.iloc[:, 0:4096].values
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
    oversample = SMOTE()
    X_train, y_train = oversample.fit_resample(X_train, y_train)
    counter = Counter(y_train)
    print(counter)
    print(y_test)
    print("xtrain ",X_train.shape)
    print("ytrain ",y_train.shape)
    print("xtest ",X_test.shape)
    print("ytest ",y_test.shape)

    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    from sklearn.neighbors import KNeighborsClassifier

    clf = KNeighborsClassifier(n_neighbors=4)
    clf.fit(X_train, y_train)
    y_pred_knn = clf.predict(X_test)
    acc_knn = round(clf.score(X_train, y_train) * 100, 2)

    print("KNN ", str(acc_knn) + '%')
    print(confusion_matrix(y_test, y_pred_knn, labels=[1, 0]))
    CM = confusion_matrix(y_test, y_pred_knn, labels=[1, 0])
    SENSITIVITY = CM[1, 1] / (CM[1, 1] + CM[1, 0])
    specificity1 = CM[0, 0] / (CM[0, 0] + CM[0, 1])
    PRECISION = CM[1, 1] / (CM[1, 1] + CM[0, 1])
    recall = CM[1, 1] / (CM[1, 1] + CM[1, 0])
    from sklearn.metrics import f1_score

    f1 = f1_score(y_test, y_pred_knn, zero_division=1)
    print("F1 SCORE ", f1)
    print("sensitivity : ", SENSITIVITY)
    print("specificity : ", specificity1)
    print("precision", PRECISION)
    print("recall", recall)
    from sklearn.metrics import f1_score

    f1 = f1_score(y_test, y_pred_knn, zero_division=1)
    print(f1)

    from sklearn.metrics import plot_roc_curve
    from sklearn import metrics

    metrics.plot_roc_curve(clf, X_test, y_test)
    plt.show()
