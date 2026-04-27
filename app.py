from flask import Flask, render_template, request
import joblib
import pandas as pd
import os
from sqlalchemy import create_engine

app = Flask(__name__)

# =========================
# LOAD MODEL + COLUMNS
# =========================
model = joblib.load('churn_model.pkl')
columns = joblib.load('columns.pkl')

# =========================
# DATABASE CONNECTION (RENDER)
# =========================
DATABASE_URL = os.getenv("DATABASE_URL")

# Fallback (your Render DB URL)
if not DATABASE_URL:
    DATABASE_URL = "postgresql://postgre:WyW4B4CseTdMfRTPkTV6roah03tzIn4J@dpg-d7np5giqqhas738271e0-a.ohio-postgres.render.com/churn_db_np39"

engine = create_engine(DATABASE_URL)


# =========================
# CREATE TABLE IF NOT EXISTS
# =========================
def create_table():
    df = pd.DataFrame(columns=["tenure", "monthly", "total", "result"])
    df.to_sql("predictions", engine, if_exists="replace", index=False)

create_table()


# =========================
# HOME
# =========================
@app.route('/')
def home():
    return render_template('index.html')


# =========================
# PREDICT
# =========================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        tenure = float(request.form['tenure'])
        monthly = float(request.form['MonthlyCharges'])
        total = float(request.form['TotalCharges'])

        # Build full input for model
        input_dict = {col: 0 for col in columns}
        input_dict['tenure'] = tenure
        input_dict['MonthlyCharges'] = monthly
        input_dict['TotalCharges'] = total

        input_df = pd.DataFrame([input_dict])

        prediction = model.predict(input_df)[0]
        result = "Churn" if prediction == 1 else "Stay"

        print("Prediction:", result)

        # Save to DB
        df = pd.DataFrame([{
            "tenure": tenure,
            "monthly": monthly,
            "total": total,
            "result": result
        }])

        df.to_sql("predictions", engine, if_exists="append", index=False)

        print("Saved to DB")

        return result

    except Exception as e:
        print("ERROR:", e)
        return "Error"


# =========================
# STATS (FOR CHART)
# =========================
@app.route('/stats')
def stats():
    try:
        df = pd.read_sql("SELECT * FROM predictions", engine)

        churn = len(df[df['result'] == 'Churn'])
        stay = len(df[df['result'] == 'Stay'])

        print("Stats:", churn, stay)

        return {"churn": churn, "stay": stay}

    except Exception as e:
        print("ERROR:", e)
        return {"churn": 0, "stay": 0}


# =========================
# RUN
# =========================
if __name__ == "__main__":
    app.run(debug=True)