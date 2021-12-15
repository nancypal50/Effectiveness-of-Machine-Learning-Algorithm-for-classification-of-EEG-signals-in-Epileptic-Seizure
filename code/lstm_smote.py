import pandas as pd
from imblearn.over_sampling import SMOTE
from collections import Counter
import matplotlib.pyplot as plt
import os
import numpy as np
import sklearn.utils
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from sklearn.metrics import confusion_matrix
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



    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1, shuffle=True)
    oversample = SMOTE()
    X_train, y_train = oversample.fit_resample(X_train, y_train)
    counter = Counter(y_train)
    print(counter)
    print("xtrain ", X_train.shape)
    print("ytrain ", y_train.shape)
    print("xtest ", X_test.shape)
    print("ytest ", y_test.shape)
    x_train = np.reshape(X_train, (X_train.shape[0], 1, x.shape[1]))
    x_test = np.reshape(X_test, (X_test.shape[0], 1, x.shape[1]))
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.layers import Embedding
    from tensorflow.keras.layers import LSTM

    tf.keras.backend.clear_session()
    model = Sequential()
    model.add(LSTM(64, input_shape=(1, 4096), activation="softmax", return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(32, activation="sigmoid"))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='sigmoid'))
    from keras.optimizers import SGD

    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    model.summary()
    history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))
    score, acc = model.evaluate(X_test, y_test)
    from sklearn.metrics import accuracy_score

    pred = model.predict(X_test)
    predict_classes = np.argmax(pred, axis=1)
    expected_classes = np.argmax(y_test, axis=1)
    correct = (accuracy_score(expected_classes, predict_classes) * 100)
    print(f"LSTM Accuracy: {correct}%")
    pred = np.round(pred)

    from sklearn.metrics import f1_score
    CM = confusion_matrix(y_test.argmax(axis=1), pred.argmax(axis=1), labels=[1, 0])
    SENSITIVITY = CM[1, 1] / (CM[1, 1] + CM[1, 0])
    specificity1 = CM[0, 0] / (CM[0, 0] + CM[0, 1])
    PRECISION = CM[1, 1] / (CM[1, 1] + CM[0, 1])
    recall = CM[1, 1] / (CM[1, 1] + CM[1, 0])
    f1 = 200 * PRECISION * recall / (PRECISION + recall)
    print(CM)
    print("F1 Score = ", '%.2f' % f1, "%")
    print("sensitivity : ", SENSITIVITY)
    print("specificity : ", specificity1)
    print("precision", PRECISION)
    print("recall", recall)

    print(history.history.keys())
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()





