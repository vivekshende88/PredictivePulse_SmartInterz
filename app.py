from flask import Flask, render_template, request, url_for
import os, pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

app = Flask(__name__)
MODEL_PATH = "model.pkl"

def train_and_save():
    df = pd.read_csv("data.csv")
    X = df[["heart_rate","activity_level","age"]]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    with open(MODEL_PATH,"wb") as f:
        pickle.dump(model, f)
    return model

# Load or train model
if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH,"rb") as f:
            model = pickle.load(f)
    except Exception as e:
        # if unpickle fails, retrain
        model = train_and_save()
else:
    model = train_and_save()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Patient info
        patient_name = request.form.get("patient_name", "").strip()
        gender = request.form.get("gender", "")
        diet = request.form.get("diet", "")
        history = request.form.get("history", "")
        # vitals
        heart_rate = float(request.form.get("heart_rate", 70))
        activity_level = float(request.form.get("activity_level", 1))
        age = float(request.form.get("age", 30))
        features = [[heart_rate, activity_level, age]]
        probs = model.predict_proba(features)[0]
        # Decide risk using highest probability class or thresholds
        # classes: 0 low,1 moderate,2 high
        class_idx = int(model.predict(features)[0])
        prob = probs[class_idx]
        if class_idx == 2 or (probs[2] > 0.5):
            result = "High Risk"
            alert_class = "danger"
            image = "red_alert.png"
            suggestions = [
                "Seek medical consultation immediately.",
                "Avoid strenuous activity until cleared by a doctor.",
                "Reduce salt intake and monitor blood pressure frequently.",
                "Ensure medication adherence if prescribed."
            ]
        elif class_idx == 1 or (probs[1] > 0.4):
            result = "Moderate Risk"
            alert_class = "warning"
            image = "yellow_alert.png"
            suggestions = [
                "Monitor your blood pressure daily.",
                "Follow a low-sodium diet and increase physical activity moderately.",
                "Limit caffeine and alcohol; manage stress."
            ]
        else:
            result = "Low Risk"
            alert_class = "success"
            image = "green_alert.png"
            suggestions = [
                "Maintain a balanced diet and regular exercise.",
                "Keep routine check-ups and continue healthy lifestyle."
            ]

        # choose avatar based on gender
        avatar = "male_avatar.png" if gender.lower()=="male" else "female_avatar.png" if gender.lower()=="female" else "male_avatar.png"

        return render_template("result.html",
                               result=result,
                               alert_class=alert_class,
                               image=image,
                               patient_name=patient_name,
                               gender=gender,
                               age=age,
                               diet=diet,
                               history=history,
                               avatar=avatar,
                               suggestions=suggestions,
                               probs=[round(float(p),3) for p in probs])
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)
