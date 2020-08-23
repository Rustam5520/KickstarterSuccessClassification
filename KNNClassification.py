from data_prep import X_train, y_train, X_test, y_test
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

KNN = KNeighborsClassifier(n_neighbors=25)
KNN.fit(X_train, y_train)

KNN_pred = KNN.predict(X_test)
accuracy_score(KNN_pred, y_test)
