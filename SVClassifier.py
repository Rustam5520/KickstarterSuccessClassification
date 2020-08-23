from data_prep import X_train, y_train, X_test, y_test
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
import numpy as np

clf_svc = LinearSVC()
clf_svc.fit(X_train, y_train)
print(clf_svc.score(X_test, y_test))

predsc = np.rint(clf_svc.predict(X_test))
SVC_cf = confusion_matrix(predsc, y_test)

accuracy = np.trace(SVC_cf) / SVC_cf.sum()
