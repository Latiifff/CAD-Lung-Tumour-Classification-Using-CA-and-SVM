from sklearn.metrics import classification_report
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut
from sklearn.metrics import make_scorer, recall_score, accuracy_score, precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
# Graphic
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.model_selection import learning_curve
import seaborn as sns
from sklearn import svm
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
import joblib
import pickle

df = pd.read_csv('model/tumourfeature.csv')
X = df.drop('CLASS', axis=1)
y = df['CLASS']
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))

print
model = svm.SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,
                decision_function_shape='ovo', gamma='scale', kernel='linear', degree=3,
                max_iter=-1, probability=False, random_state=None, shrinking=True,
                tol=0.001, verbose=False)

print("\t####---Split Data (80%)---####\n")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred, labels=[0, 1])
print(report)

accuracy = accuracy_score(y_pred, y_test)
recall = recall_score(y_test, y_pred, average='macro')
precision = precision_score(y_test, y_pred, average='macro')

print(f"Train set Accuracy: {accuracy_score(y_train, model.predict(X_train)) * 100:.2f}%" )
print(f"Train set Precision: {precision_score(y_train, model.predict(X_train)) * 100:.2f}%" )
print(f"Train set Recall: {recall_score(y_train, model.predict(X_train)) * 100:.2f}%" )

print(f"Test set Accuracy: {accuracy * 100:.2f}%" )
print(f"Test set Precision: {precision * 100:.2f}%" )
print(f"Test set Recall: {recall * 100:.2f}%" )
cohen_score = cohen_kappa_score(y_test, y_pred)
print(f"Kappa Score:{cohen_score * 100:.2f} \n")
matrix = confusion_matrix(y_test, y_pred)
print(matrix,"\n")

# Buat scorer untuk accuracy, precision, dan recall
accuracy_scorer = make_scorer(accuracy_score)
precision_scorer = make_scorer(precision_score, average='macro')
recall_scorer = make_scorer(recall_score, average='macro') 

# Hitung cross-validation scores untuk accuracy
cross_accuracy_scores = cross_val_score(model,X,y,cv=3,scoring=accuracy_scorer)
mean_accuracy = cross_accuracy_scores.mean()
std_accuracy = cross_accuracy_scores.std()

# Hitung cross-validation scores untuk precision
cross_precision_scores = cross_val_score(model, X, y, cv=3, scoring=precision_scorer)
mean_precision = cross_precision_scores.mean()
std_precision = cross_precision_scores.std() 

# Hitung cross-validation scores untuk recall
cross_recall_scores = cross_val_score(model, X, y, cv=3, scoring=recall_scorer)
mean_recall = cross_recall_scores.mean()
std_recall = cross_recall_scores.std()

# print Accuracy, Precision, Recall
print(f"Mean Accuracy: {mean_accuracy * 100:.2f}%")
print(f"Standard Deviation of Accuracy: {std_accuracy * 100:.3f}%")
print(f"Mean Precision: {mean_precision * 100:.2f}%")
print(f"Standard Deviation of Precision: {std_precision * 100:.3f}%")
print(f"Mean Recall: {mean_recall * 100:.2f}%")
print(f"Standard Deviation of Recall: {std_recall * 100:.3f}%")


# For graphic
report = classification_report(y_test, y_pred, labels=[0, 1], output_dict=True)
df_report = pd.DataFrame(report).transpose()

fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

precision, recall, _ = precision_recall_curve(y_test, y_pred)
average_precision = average_precision_score(y_test, y_pred)

train_sizes, train_scores, test_scores = learning_curve(model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)

train_scores_mean = train_scores.mean(axis=1)
train_scores_std = train_scores.std(axis=1)
test_scores_mean = test_scores.mean(axis=1)
test_scores_std = test_scores.std(axis=1)

flow_choice = input("Enter 1 to Generate Model or else To Decline: ")
flow_choice = int(flow_choice)

if flow_choice == 1:
    model = svm.SVC(kernel='linear', gamma='auto')
    model.fit(X, y)
    filename = 'modelBaru2'
    joblib.dump(model, filename)
    loaded_model = joblib.load(filename)
    result = loaded_model.score(X, y)
    with open(f'{filename}.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)
    print(result)
else:
    plt.figure(figsize=(10, 6))
    sns.heatmap(df_report.iloc[:-1, :].T, annot=True, cmap="flare")
    plt.title('Classification Report')
    plt.show()

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()


    plt.figure()
    plt.plot(recall, precision, color='b', lw=2, label='Precision-Recall curve (area = %0.2f)' % average_precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.show()

    plt.figure()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.xlabel('Training examples')
    plt.ylabel('Score')
    plt.title('Learning Curve')
    plt.legend(loc="best")
    plt.show()