from data_prep import X_train, y_train, X_test, y_test
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score

RR = RidgeClassifier(alpha=0.1)
RR.fit(X_test, y_test)

RR_test_pred = RR.predict(X_test)
accuracy_score(RR_test_pred, y_test)
