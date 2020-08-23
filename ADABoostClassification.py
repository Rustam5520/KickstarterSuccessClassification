from data_prep import X_train, y_train, X_test, y_test
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score

ADB = AdaBoostClassifier()
ADB.fit(X_train, y_train)

pred = ADB.predict(X_test)

accuracy_score(pred, y_test)
