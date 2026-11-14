"""
Churn Risk Checker: enter a customer's details, get a leave/stay prediction,
risk level, the reasons, and the recommended retention actions.

Run 05_model_improvement.ipynb first (it creates churn_model_v2.joblib), then:
    .venv\\Scripts\\python app.py
and open http://127.0.0.1:7860
"""
import os

import gradio as gr
import joblib
import pandas as pd

from churn_utils import ADDONS, prepare

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "churn_model_v2.joblib")
if not os.path.exists(MODEL_PATH):
    raise SystemExit("churn_model_v2.joblib not found. Run 05_model_improvement.ipynb first "
                     "(and 01_data_types.ipynb before it), then start the app again.")
saved = joblib.load(MODEL_PATH)
model = saved["model"]
THRESHOLD = saved["threshold"]

# price list found in notebook 04 (bills are almost exactly these sums)
PRICES = {"PhoneService": 20, "MultipleLines": 5, "DSL": 25, "Fiber optic": 50,
          "OnlineSecurity": 5, "OnlineBackup": 5, "DeviceProtection": 5,
          "TechSupport": 5, "StreamingTV": 10, "StreamingMovies": 10}


def monthly_charge(phone, multiple_lines, internet, addons):
    total = 0
    if phone == "Yes":
        total += PRICES["PhoneService"]
        if multiple_lines == "Yes":
            total += PRICES["MultipleLines"]
    if internet != "No":
        total += PRICES[internet]
        for name, value in addons.items():
            if value == "Yes":
                total += PRICES[name]
    return float(total)


def build_customer(gender, senior, partner, dependents, tenure,
                   phone, multiple_lines, internet,
                   security, backup, protection, support, tv, movies,
                   contract, paperless, payment):
    """Turn the form answers into one row that looks exactly like the dataset."""
    addons = dict(zip(ADDONS, [security, backup, protection, support, tv, movies]))

    # keep answers consistent with how the dataset records them
    if phone == "No":
        multiple_lines = "No phone service"
    if internet == "No":
        addons = {name: "No internet service" for name in addons}

    charge = monthly_charge(phone, multiple_lines, internet, addons)

    row = {"gender": gender, "SeniorCitizen": senior, "Partner": partner,
           "Dependents": dependents, "tenure": int(tenure), "PhoneService": phone,
           "MultipleLines": multiple_lines, "InternetService": internet,
           **addons,
           "Contract": contract, "PaperlessBilling": paperless,
           "PaymentMethod": payment, "MonthlyCharges": charge}
    return pd.DataFrame([row])


def reasons(c):
    """Plain-English reasons, based on the drivers found in the EDA and notebook 03."""
    up, down = [], []
    has_internet = c["InternetService"] != "No"

    if c["Contract"] == "Month-to-month":
        up.append("Month-to-month contract (43% churn vs 3% on two-year)")
    else:
        down.append(f"{c['Contract']} contract (customers on contracts rarely leave)")

    if c["tenure"] <= 6:
        up.append("Brand-new customer, first 6 months (53% churn)")
    elif c["tenure"] <= 12:
        up.append("Still in the first year (36% churn)")
    elif c["tenure"] > 48:
        down.append("Loyal for more than 4 years (under 10% churn)")

    if c["InternetService"] == "Fiber optic":
        up.append("Fiber internet, our most expensive plan (42% churn)")
    if has_internet and c["TechSupport"] == "No":
        up.append("No TechSupport (42% churn vs 15% with it)")
    elif c["TechSupport"] == "Yes":
        down.append("Has TechSupport")
    if has_internet and c["OnlineSecurity"] == "No":
        up.append("No OnlineSecurity (42% churn vs 15% with it)")
    elif c["OnlineSecurity"] == "Yes":
        down.append("Has OnlineSecurity")

    if c["PaymentMethod"] == "Electronic check":
        up.append("Pays by electronic check (45% churn)")
    elif "automatic" in c["PaymentMethod"]:
        down.append("Pays automatically (about 16% churn)")

    if c["SeniorCitizen"] == "Yes":
        up.append("Senior citizen (42% churn)")
    if c["Partner"] == "No" and c["Dependents"] == "No":
        up.append("No partner or dependents (single customers leave more)")

    return up, down


def recommended_actions(c):
    """The retention playbook from notebook 04."""
    actions = []
    if c["Contract"] == "Month-to-month":
        actions.append("**Offer a 1-year contract at 10% off.** This is the biggest lever in the simulation.")
    if c["InternetService"] != "No" and (c["TechSupport"] == "No" or c["OnlineSecurity"] == "No"):
        actions.append("**Give free TechSupport + OnlineSecurity for 12 months**, the add-ons that keep customers.")
    if c["PaymentMethod"] == "Electronic check":
        actions.append("**Offer a $25 credit to switch to autopay.**")
    if c["tenure"] <= 6:
        actions.append("**Add to the onboarding programme:** welcome call and a check-in at month 3.")
    if not actions:
        actions.append("No special offer needed. Keep up normal service.")
    return actions


def predict(*answers):
    customer = build_customer(*answers)
    c = customer.iloc[0]

    score = float(model.predict_proba(prepare(customer, saved))[:, 1][0])

    # bands start at the model's threshold, so "likely to leave" is always at least MEDIUM
    if score >= 0.7:
        risk = "🔴 HIGH risk"
    elif score >= THRESHOLD:
        risk = "🟠 MEDIUM risk"
    else:
        risk = "🟢 LOW risk"
    verdict = "likely to **LEAVE**" if score >= THRESHOLD else "likely to **STAY**"

    summary = (f"## {risk}\n"
               f"This customer is {verdict}.\n\n"
               f"- Risk score: **{score * 100:.0f}%**\n"
               f"- Monthly bill (from the price list): **${c['MonthlyCharges']:.2f}**\n"
               f"- Priority score (risk x bill): **{score * c['MonthlyCharges']:.1f}**\n\n"
               f"_The risk score is for ranking customers (higher = contact sooner), "
               f"not an exact chance of leaving._")

    up, down = reasons(c)
    why = "### Why\n"
    why += "**Raises risk:**\n" + ("\n".join(f"- {r}" for r in up) if up else "- nothing major") + "\n\n"
    why += "**Lowers risk:**\n" + ("\n".join(f"- {r}" for r in down) if down else "- nothing major")

    todo = "### What to do\n" + "\n".join(f"{i}. {a}" for i, a in enumerate(recommended_actions(c), 1))

    return {"Leave": score, "Stay": 1 - score}, summary, why, todo


YES_NO = ["Yes", "No"]

with gr.Blocks(title="Churn Risk Checker") as demo:
    gr.Markdown("# 📞 Churn Risk Checker\n"
                "Enter a customer's details to see if they are likely to leave, why, and what to offer them.")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Customer")
            gender = gr.Radio(["Female", "Male"], value="Female", label="Gender")
            senior = gr.Radio(YES_NO, value="No", label="Senior citizen")
            partner = gr.Radio(YES_NO, value="No", label="Has a partner")
            dependents = gr.Radio(YES_NO, value="No", label="Has dependents")
            tenure = gr.Slider(0, 72, value=12, step=1, label="Tenure (months with us)")

            gr.Markdown("### Account")
            contract = gr.Dropdown(["Month-to-month", "One year", "Two year"],
                                   value="Month-to-month", label="Contract")
            paperless = gr.Radio(YES_NO, value="Yes", label="Paperless billing")
            payment = gr.Dropdown(["Electronic check", "Mailed check",
                                   "Bank transfer (automatic)", "Credit card (automatic)"],
                                  value="Electronic check", label="Payment method")

        with gr.Column():
            gr.Markdown("### Services")
            phone = gr.Radio(YES_NO, value="Yes", label="Phone service")
            multiple_lines = gr.Radio(YES_NO, value="No", label="Multiple lines")
            internet = gr.Dropdown(["Fiber optic", "DSL", "No"], value="Fiber optic",
                                   label="Internet service")
            security = gr.Radio(YES_NO, value="No", label="Online security")
            backup = gr.Radio(YES_NO, value="No", label="Online backup")
            protection = gr.Radio(YES_NO, value="No", label="Device protection")
            support = gr.Radio(YES_NO, value="No", label="Tech support")
            tv = gr.Radio(YES_NO, value="No", label="Streaming TV")
            movies = gr.Radio(YES_NO, value="No", label="Streaming movies")
            gr.Markdown("_Add-ons are ignored if the customer has no internet. "
                        "The monthly bill is calculated from the price list._")

        with gr.Column():
            gr.Markdown("### Prediction")
            button = gr.Button("Check churn risk", variant="primary")
            label = gr.Label(label="Leave vs stay")
            summary = gr.Markdown()
            why = gr.Markdown()
            todo = gr.Markdown()

    inputs = [gender, senior, partner, dependents, tenure,
              phone, multiple_lines, internet,
              security, backup, protection, support, tv, movies,
              contract, paperless, payment]
    outputs = [label, summary, why, todo]

    gr.Examples(
        label="Try an example customer",
        examples=[
            # high risk: new, month-to-month, fiber, no support, e-check
            ["Female", "Yes", "No", "No", 3, "Yes", "No", "Fiber optic",
             "No", "No", "No", "No", "Yes", "No", "Month-to-month", "Yes", "Electronic check"],
            # in-between: just over a year, DSL, month-to-month but has security
            ["Male", "No", "Yes", "No", 14, "Yes", "No", "DSL",
             "Yes", "No", "No", "No", "No", "No", "Month-to-month", "No", "Mailed check"],
            # loyal: long tenure, two-year, autopay, full support
            ["Male", "No", "Yes", "Yes", 60, "Yes", "Yes", "DSL",
             "Yes", "Yes", "Yes", "Yes", "No", "No", "Two year", "No", "Credit card (automatic)"],
        ],
        inputs=inputs,
    )

    button.click(predict, inputs=inputs, outputs=outputs)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
