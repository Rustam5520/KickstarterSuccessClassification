from data_prep import X_train, y_train, X_test, y_test
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import numpy as np

RF = RandomForestClassifier(max_depth=32, n_estimators=100)
RF.fit(X_train, y_train)

print(RF.score(X_test, y_test))

pred_for = np.rint(RF.predict(X_test))

RF_cf = confusion_matrix(pred_for, y_test)

accuracy = np.trace(RF_cf) / RF_cf.sum()
