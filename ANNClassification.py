from data_prep import X_train, y_train, X_test, y_test
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.metrics import confusion_matrix
# baseline model
def create_baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(191, input_dim=191, activation='relu'))
    model.add(Dense(256, activation='sigmoid'))
    model.add(Dense(128, activation='sigmoid'))
    model.add(Dense(64, activation='sigmoid'))
    model.add(Dense(32, activation='sigmoid'))
    model.add(Dense(16, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

m = create_baseline_model()
m.fit(X_train, y_train, batch_size=1024, epochs=50, validation_split=0.1)

print(accuracy_score(pred, y_test))

pred = np.rint(m.predict(X_test))


print(confusion_matrix(pred, y_test))
