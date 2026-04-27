from flask import Flask, render_template, request
import pickle
import pandas as pd
from sqlalchemy import create_engine
import os
from urllib.parse import quote_plus

# -------------------------------
# INIT APP
# -------------------------------
app = Flask(__name__)

# -------------------------------
# LOAD MODEL + COLUMNS
# -------------------------------
model = pickle.load(open('churn_model.pkl', 'rb'))
columns = pickle.load(open('columns.pkl', 'rb'))

# -------------------------------
# DATABASE CONNECTION
# -------------------------------

# If deploying later, this will be used
DATABASE_URL = os.getenv("DATABASE_URL")

if DATABASE_URL:
    engine = create_engine(DATABASE_URL)
else:
    # 🔥 LOCAL DATABASE (change password if needed)
    password = quote_plus("Sr@22102003")   # <-- your password
    engine = create_engine(f"postgresql://postgres:{password}@localhost:5432/churn_db")


# -------------------------------
# HOME ROUTE
# -------------------------------
@app.route('/')
def home():
    return render_template('index.html')


# -------------------------------
# PREDICT ROUTE
# -------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form.to_dict()

        print("FORM DATA:", data)

        # -------------------------------
        # PREPARE INPUT FOR MODEL
        # -------------------------------
        input_data = []

        for col in columns:
            value = data.get(col, 0)

    # Handle categorical values (basic encoding)
            if isinstance(value, str):
                if value.lower() in ['yes', 'true']:
                    value = 1
                elif value.lower() in ['no', 'false']:
                    value = 0
                else:
                    value = 0

            try:
                input_data.append(float(value))
            except:
                input_data.append(0)

        # -------------------------------
        # PREDICTION
        # -------------------------------
        prediction = model.predict([input_data])[0]
        probability = model.predict_proba([input_data])[0][1]

        result = "Churn" if prediction == 1 else "Stay"

        print("Prediction:", result)

        # -------------------------------
        # SAVE TO DATABASE
        # -------------------------------
        print("Saving to DB...")

        df = pd.DataFrame([{
            "tenure": float(data.get('tenure', 0)),
            "monthly": float(data.get('MonthlyCharges', 0)),
            "total": float(data.get('TotalCharges', 0)),
            "result": result
        }])

        df.to_sql("predictions", engine, if_exists='append', index=False)

        print("Saved successfully!")

        # -------------------------------
        # RETURN RESULT
        # -------------------------------
        return render_template(
            'index.html',
            prediction_text=f"Result: {result} (Probability: {round(probability, 2)})"
        )

    except Exception as e:
        print("ERROR:", str(e))
        return render_template('index.html', prediction_text="Something went wrong")


# -------------------------------
# RUN APP
# -------------------------------
if __name__ == "__main__":
    app.run()