from data_prep import X_train, y_train, X_test, y_test
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold

XGB_K_CV = XGBClassifier()
XGB_K_CV.fit(X_train, y_train)

XGB_pred_log = XGB_K_CV.predict(X_test)
XGB_pred_prob = XGB_K_CV.predict_proba(X_test)[:, 1]

#XGB_test_score = accuracy_score(XGB_pred_log, y_test)

XGB_Con_Mat = confusion_matrix(XGB_pred_log, y_test)
accuracy = np.trace(XGB_Con_Mat) / XGB_Con_Mat.sum()

print(pd.DataFrame(XGB_Con_Mat, columns=['pred_s', 'pred_f'], index= ['act_s', 'act_f']))
print(accuracy)

fpr, tpr, _ = roc_curve(y_test, XGB_pred_prob)
auc = roc_auc_score(y_test, XGB_pred_prob)

plt.plot(fpr, tpr, label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()

# K-Fold Cross Validation
def evaluate_model(model, X, y):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
    scores = cross_val_score(model, X, y,
                             scoring='accuracy',
                             cv=cv,
                             n_jobs=-1,
                             error_score='raise')
    return scores

XGB_K_CV = XGBClassifier()
XGB_K_CV_scores = cross_val_score(XGB_K_CV, X_train, y_train)



# Some results:
# Random state 42
0.9391916337887428
0.9341929389063647
# Fully random CV
0.9385412538979722
0.9347657859562023

