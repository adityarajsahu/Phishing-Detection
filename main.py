import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import pickle

DataFrame = pd.read_csv('Dataset/Phising_Training_Dataset.csv')
# print(DataFrame.columns)

features = DataFrame.drop(columns=['key', 'Result'])
label = DataFrame['Result']

X_train, X_val, Y_train, Y_val = train_test_split(features, label, test_size=0.2, shuffle=True)

logistic_regression = LogisticRegression()
logistic_regression.fit(X_train, Y_train)
log_reg_accuracy = logistic_regression.score(X_val, Y_val)
print("Logistic Regression Accuracy: {:.2f} %".format(log_reg_accuracy * 100))
pickle.dump(logistic_regression, open('Models/logistic_regression_model.sav', 'wb'))

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
dec_tree_accuracy = decision_tree.score(X_val, Y_val)
print("Decision Tree Accuracy: {:.2f} %".format(dec_tree_accuracy * 100))
pickle.dump(decision_tree, open('Models/decision_tree_model.sav', 'wb'))

random_forest = RandomForestClassifier()
random_forest.fit(X_train, Y_train)
ran_for_accuracy = random_forest.score(X_val, Y_val)
print("Random Forest Accuracy: {:.2f} %".format(ran_for_accuracy * 100))
pickle.dump(random_forest, open('Models/random_forest_model.sav', 'wb'))

Test_df = pd.read_csv('Dataset/Phising_Testing_Dataset.csv')
key_column = Test_df['key']
X = Test_df.drop(columns=['key'])
predictions = random_forest.predict(X)
predictions = predictions.reshape(predictions.shape[0])

submission = pd.DataFrame({'key': key_column, 'Result': predictions})
submission['Result'] = submission['Result'].astype(int)
submission.to_csv('submission_random_forest.csv', index=False)

svm_clf = SVC()
svm_clf.fit(X_train, Y_train)
svm_accuracy = svm_clf.score(X_val, Y_val)
print("SVM Accuracy: {:.2f} %".format(svm_accuracy * 100))
pickle.dump(svm_clf, open('Models/svm_clf_model.sav', 'wb'))

knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
knn_accuracy = knn.score(X_val, Y_val)
print("KNN Accuracy: {:.2f} %".format(knn_accuracy * 100))
pickle.dump(knn, open('Models/knn_model.sav', 'wb'))

nb_clf = GaussianNB()
nb_clf.fit(X_train, Y_train)
nb_accuracy = nb_clf.score(X_val, Y_val)
print("Naive Bayes Accuracy: {:.2f} %".format(nb_accuracy * 100))
pickle.dump(nb_clf, open('Models/nb_clf_model.sav', 'wb'))
