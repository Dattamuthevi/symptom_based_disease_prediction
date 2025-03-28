from flask import Flask, request, jsonify, render_template
import pandas as pd

# Define Flask App using default 'templates' and 'static' folders in the current directory
app = Flask(__name__, template_folder="template", static_folder="static")

# Load Data
severity_df = pd.read_csv("Symptom_severity.csv")
description_df = pd.read_csv("symptom_Description.csv", names=["Disease", "Description"], header=0)
precaution_df = pd.read_csv("symptom_precaution.csv")
training_df = pd.read_csv("Training.csv")

# Fix Column Names in severity_df
severity_df.columns = ["Symptom", "Weight"]

# Dictionaries for Lookup
severity_dict = dict(zip(severity_df["Symptom"], severity_df["Weight"]))
description_dict = dict(zip(description_df["Disease"], description_df["Description"]))
precaution_dict = {row[0]: row[1:].tolist() for row in precaution_df.values}

# Dummy Classifier to replace the pre-trained model
class DummyClassifier:
    def predict(self, X):
        # A simple rule: if any symptom is active (i.e. value 1 in the input vector), return 'Flu'
        # Otherwise, return 'Healthy'
        for row in X:
            if sum(row) > 0:
                return ["Flu"]
        return ["Healthy"]

# Instantiate the dummy classifier
clf = DummyClassifier()

# Route for Frontend
@app.route("/")
def home():
    return render_template("index.html")

# API Route for Predictions
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    symptoms = data.get("symptoms", [])

    if not symptoms:
        return jsonify({"error": "No symptoms provided."})

    # Convert symptoms to input vector based on the columns in training_df
    symptom_indices = [training_df.columns.get_loc(symptom) for symptom in symptoms if symptom in training_df.columns]
    input_vector = [0] * len(training_df.columns)
    for idx in symptom_indices:
        input_vector[idx] = 1

    # Predict Disease using the dummy classifier
    disease_predicted = clf.predict([input_vector])[0]

    # Get Description & Precautions
    description = description_dict.get(disease_predicted, "No description available")
    precautions = precaution_dict.get(disease_predicted, ["No precautions available"])

    return jsonify({
        "disease": disease_predicted,
        "description": description,
        "precautions": precautions
    })

if __name__ == "__main__":
    app.run(debug=True)
