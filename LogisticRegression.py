import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from data_prep import X_train, y_train, X_test, y_test
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold

LR = LogisticRegression()
LR.fit(X_train, y_train)

pred_log = np.rint(LR.predict(X_test))
pred_prob = LR.predict_proba(X_test)[:, 1]

LR_Con_Mat = confusion_matrix(pred_log, y_test)
accuracy = np.trace(LR_Con_Mat) / LR_Con_Mat.sum()

print(pd.DataFrame(LR_Con_Mat, columns=['pred_s', 'pred_f'], index= ['act_s', 'act_f']))
print(accuracy)

fpr, tpr, _ = roc_curve(y_test, pred_prob)
auc = roc_auc_score(y_test, pred_prob)

plt.plot(fpr, tpr, label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()

def evaluate_model(model, X, y):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
    scores = cross_val_score(model, X, y,
                             scoring='accuracy',
                             cv=cv,
                             n_jobs=-1,
                             error_score='raise')
    return scores

# evaluate_model((), X_train, y_train)
