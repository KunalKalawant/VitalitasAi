import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.stats import mode
import statistics
import warnings

# Read the training CSV file and remove last column if it is null
data = pd.read_csv("/Users/kunalkalawant/Desktop/vss2/disease_prediction_project/data/Disease_Training.csv").dropna(axis=1)

# Check if the dataset is balanced using a bar plot
disease_counts = data['prognosis'].value_counts()
temp_df = pd.DataFrame({'Disease': disease_counts.index, 'Counts': disease_counts.values})

plt.figure(figsize=(18, 8))
sns.barplot(x='Disease', y='Counts', data=temp_df)
plt.xticks(rotation=90)
plt.show()

# Label encode the 'prognosis' column
encoder = LabelEncoder()
y = encoder.fit_transform(data['prognosis'])

# Create predictions classes from the label-encoded 'prognosis'
prediction_classes = {index: disease for index, disease in enumerate(encoder.classes_)}

# Create symptom index from columns of the dataset (excluding 'prognosis')
symptom_index = {symptom: idx for idx, symptom in enumerate(data.columns[:-1])}

# Populate the data_dict
data_dict = {
    "symptom_index": symptom_index,
    "predictions_classes": prediction_classes
}

# Encode 'prognosis' column
data['prognosis'] = encoder.fit_transform(data['prognosis'])

# Split dataset into training and testing sets
x = data.iloc[:, :-1]
y = data.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=24)

# K-Fold cross-validation for model selection
def cv_scoring(estimator, x, y):
    return accuracy_score(y, estimator.predict(x))

# Initialize the models
models = {
    "SVC": SVC(),
    "Gaussian NB": GaussianNB(),
    "Random Forest": RandomForestClassifier(random_state=18)
}

# Cross-validation scores for each model
for model_name in models:
    model = models[model_name]
    scores = cross_val_score(model, x_train, y_train, cv=10, n_jobs=-1, scoring=cv_scoring)
    print("=" * 60)
    print(model_name)
    print(f"Scores: {scores}")
    print(f"Mean Score: {np.mean(scores)}")

# Train and test the models
# SVM Model
svm_model = SVC()
svm_model.fit(x_train, y_train)
print("Accuracy of SVM model on training data:", accuracy_score(y_train, svm_model.predict(x_train)) * 100)
preds = svm_model.predict(x_test)
print("Accuracy of SVM model on testing data:", accuracy_score(y_test, preds) * 100)
cf_matrix = confusion_matrix(y_test, preds)
plt.figure(figsize=(12, 8))
sns.heatmap(cf_matrix, annot=True)
plt.title("Confusion Matrix for SVM Model")
plt.show()

# Naive Bayes Model
nb_model = GaussianNB()
nb_model.fit(x_train, y_train)
print("Accuracy of NB model on training data:", accuracy_score(y_train, nb_model.predict(x_train)) * 100)
preds = nb_model.predict(x_test)
print("Accuracy of NB model on testing data:", accuracy_score(y_test, preds) * 100)
cf_matrix = confusion_matrix(y_test, preds)
plt.figure(figsize=(12, 8))
sns.heatmap(cf_matrix, annot=True)
plt.title("Confusion Matrix for Naive Bayes Model")
plt.show()

# Random Forest Model
rf_model = RandomForestClassifier(random_state=18)
rf_model.fit(x_train, y_train)
print("Accuracy of RF model on training data:", accuracy_score(y_train, rf_model.predict(x_train)) * 100)
preds = rf_model.predict(x_test)
print("Accuracy of RF model on testing data:", accuracy_score(y_test, preds) * 100)
cf_matrix = confusion_matrix(y_test, preds)
plt.figure(figsize=(12, 8))
sns.heatmap(cf_matrix, annot=True)
plt.title("Confusion Matrix for Random Forest Model")
plt.show()

# Train final models on full dataset
final_SVM_model = SVC()
final_NB_model = GaussianNB()
final_RF_model = RandomForestClassifier(random_state=18)

final_SVM_model.fit(x, y)
final_NB_model.fit(x, y)
final_RF_model.fit(x, y)

# Load the test dataset
test_data = pd.read_csv("/Users/kunalkalawant/Desktop/vss2/disease_prediction_project/data/Disease_Testing.csv").dropna(axis=1)
test_x = test_data.iloc[:, :-1]
test_y = encoder.transform(test_data.iloc[:, -1])

# Make predictions with each model
svm_preds = final_SVM_model.predict(test_x)
nb_preds = final_NB_model.predict(test_x)
rf_preds = final_RF_model.predict(test_x)

# Combine predictions using majority vote
combined_preds = np.array([svm_preds, nb_preds, rf_preds])
final_preds = mode(combined_preds, axis=0).mode.ravel()

# Calculate and print accuracy
print("Accuracy of combined models on unseen data:", accuracy_score(test_y, final_preds) * 100)

# Generate confusion matrix and heatmap
cf_matrix = confusion_matrix(test_y, final_preds)
plt.figure(figsize=(12, 8))
sns.heatmap(cf_matrix, annot=True, fmt='d')
plt.title("Confusion Matrix for Combined Model")
plt.show()

# Predict disease from symptoms
def predictDisease(symptoms):
    symptoms = symptoms.split(",")
    input_data = [0] * len(data_dict["symptom_index"])

    for symptom in symptoms:
        symptom = symptom.strip().lower().replace(" ", "_")
        if symptom in data_dict["symptom_index"]:
            index = data_dict["symptom_index"][symptom]
            input_data[index] = 1
        else:
            print(f"Warning: '{symptom}' not recognized.")

    input_data = np.array(input_data).reshape(1, -1)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        rf_prediction = data_dict["predictions_classes"][final_RF_model.predict(input_data)[0]]
        nb_prediction = data_dict["predictions_classes"][final_NB_model.predict(input_data)[0]]
        svm_prediction = data_dict["predictions_classes"][final_SVM_model.predict(input_data)[0]]

    final_prediction = statistics.mode([rf_prediction, nb_prediction, svm_prediction])

    return {
        "rf_model_prediction": rf_prediction,
        "naive_bayes_prediction": nb_prediction,
        "svm_model_prediction": svm_prediction,
        "final_prediction": final_prediction
    }

# Testing the function with user input
user_input = input("Enter symptoms separated by commas: ")
result = predictDisease(user_input)

# Display the prediction results
if result:
    print("Predicted disease based on symptoms:")
    for model, prediction in result.items():
        print(f"{model}: {prediction}")
