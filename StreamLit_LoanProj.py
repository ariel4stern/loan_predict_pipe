import pandas as pd
import streamlit as st
import os
import joblib


######################## TO RUN:
# 1 INSTALL requirements.txt

st.set_page_config(
    page_title="Loan Approval Checker",
    page_icon="🏦",
    layout="centered"
)

# Title
st.title("🏦 Loan Approval Checker")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "loan_svm_model.joblib")

# If model missing -> show waiting message and stop
if not os.path.exists(MODEL_PATH):
     st.warning("The system is initializing, please wait")
     st.code(MODEL_PATH)
     st.stop()


@st.cache_resource
def load_model(path: str):
    return joblib.load(path)

pipe_model = load_model(MODEL_PATH)

model = pipe_model["model"]

cm_table = pipe_model["confusion_matrix_table"]
cm_df = pd.DataFrame(
    cm_table,
    index=["Yes (True)", "No (True)"],
    columns=["Yes (Prediction)", "No (Prediction)"]
)

if model is None or cm_df is None:
    st.warning("Model or Score not found")
    st.stop()


# ---- Input form (screenshot-friendly UI) ----
with st.form("loan_form"):
    st.subheader("Loan Application Details")

    ApplicantIncome = st.number_input(
        "Applicant Income (monthly)",
        min_value=0.0,
        value=5000.0,
        step=100.0
    )

    CoapplicantIncome = st.number_input(
        "Coapplicant Income (monthly)",
        min_value=0.0,
        value=0.0,
        step=100.0
    )

    LoanAmount = st.number_input(
        "Requested Loan Amount",
        min_value=0.0,
        value=150.0,
        step=10.0
    )

    Loan_Amount_Term = st.number_input(
        "Loan Term (months)",
        min_value=1.0,
        value=360.0,
        step=12.0
    )

    dict_opt = {"Yes": "Married", "No": "Single"}
    Married = st.selectbox(
        "Marital Status",
        options=["Yes", "No"],
        index=0,
        format_func=lambda x: dict_opt[x]
    )

    Credit_History = st.selectbox(
        "Credit History",
        options=[1, 0],
        index=0,
        format_func=lambda x: "Exists (1)" if x == 1 else "Does not exist (0)"
    )

    check_model_score = st.form_submit_button("Check mean score model score")

    submitted = st.form_submit_button("Check Loan Eligibility")


# Runs only after clicking the button
if submitted:
    X_to_pred = pd.DataFrame({
        "ApplicantIncome": float(ApplicantIncome),
        "CoapplicantIncome": float(CoapplicantIncome),
        "LoanAmount": float(LoanAmount),
        "Loan_Amount_Term": float(Loan_Amount_Term),
        "Married": Married, #I leave it as "Yes"/"No" because the pipe_line_model(loan_svm_model.joblib) convert by his own OneHotEncoder_pipe,
                            # also the model got the input from the column as "Yes"/"No" unlike Credit_History(its got 1.0/0.0)
        "Credit_History": Credit_History
    },
    index=["Client"])

    x_df = pd.DataFrame(
        X_to_pred
    )

    st.dataframe(x_df)

    prediction = model.predict(X_to_pred)[0]

    if prediction is None:
        st.warning("Prediction Error")
        st.stop()

    if prediction == 1:
        st.success("You approved for taking a loan Good luck")
    else:
        st.error("You are not allowed to take a loan we are sorry")

if check_model_score:
    st.dataframe(cm_df)


