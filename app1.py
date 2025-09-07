from flask import Flask, request, jsonify, render_template, send_file, session, redirect, url_for
import pandas as pd
import io
import joblib
import logging
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from FINALMODEL import explain_and_generate_llm_report
from database import AuthManager, db_manager  
from PyPDF2 import PdfReader
import os
from werkzeug.utils import secure_filename


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = "your-secret-key-change-this"
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_PERMANENT"] = False

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

auth_manager = AuthManager(db_manager)


try:
    patient_df = pd.read_csv("final_data.csv")
    intervention_model = joblib.load("intervention_model.joblib")
    meta_model = joblib.load("meta_risk_model.joblib")   
    logger.info("✅ Models and patient data loaded successfully")
except Exception as e:
    logger.error(f"❌ Failed to load models or data: {e}")
    patient_df, intervention_model, meta_model = None, None, None


def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session or "user_type" not in session:
            return redirect(url_for("landing_page"))
        return f(*args, **kwargs)
    return decorated

def get_current_user():
    if "user_id" in session and "user_type" in session:
        return {
            "id": session["user_id"],
            "type": session["user_type"],
            "name": session.get("user_name", ""),
            "email": session.get("user_email", "")
        }
    return None


@app.before_request
def restrict_ngo_dashboard():
    path = request.path or ""
    if path.startswith("/ngo/dashboard"):
        if session.get("user_type") != "ngo":
            return redirect(url_for("ngologin"))
        return None


@app.route("/")
@app.route("/landingPage.html")
def landing_page():
    return render_template("landingPage.html")

@app.route("/ngo/login", methods=["GET", "POST"])
def ngologin():
    if request.method == "POST":
        data = request.get_json()
        success, user, message = auth_manager.authenticate_ngo(
            data.get("email"), data.get("password")
        )
        if success:
            session.update({
                "user_id": user["id"],
                "user_type": "ngo",
                "user_name": user["full_name"],
                "user_email": user["email"]
            })
            return jsonify({"success": True, "message": message, "redirect": "/ngo/dashboard/"})
        return jsonify({"success": False, "message": message}), 401
    return render_template("ngologin.html")

@app.route("/provider/signup", methods=["GET", "POST"])
def providersignup():
    if request.method == "POST":
        data = request.get_json()

        success, user, message = auth_manager.register_provider(
            data.get("fullName"),
            data.get("email"),
            data.get("medicalId"),
            data.get("password"),
            data.get("specialization"),
            data.get("hospitalName"),
            data.get("licenseNumber")
        )
        if success:
            session["provider"] = user
            session["user_type"] = "provider"  
            return jsonify({"success": True, "redirect": "/provider"})
        return jsonify({"success": False, "message": message})

    return render_template("providersignup.html")


@app.route("/provider")
def provider():
    provider = session.get("provider")
    if not provider:
        return redirect(url_for("providersignup"))

    patient_id = request.args.get("patientId")
    fips = request.args.get("fips")

    report = None
    risk_score = None
    if patient_id and patient_df is not None and "Patient_ID" in patient_df.columns:
        try:
            row = patient_df.loc[patient_df["Patient_ID"].astype(str) == str(patient_id)]
        except Exception:
            row = patient_df.loc[patient_df["Patient_ID"] == patient_id]
        if not row.empty:
            patient_data = row.iloc[0].to_dict()
            if meta_model is not None:
                try:
                    X = pd.DataFrame([patient_data])
                    risk_score = round(float(meta_model.predict_proba(X)[0][1]) * 100, 2)
                except Exception as e:
                    logger.error(f"Meta model error: {e}")
            report = explain_and_generate_llm_report(patient_data, patient_data)

            session["last_report"] = {
                "patient_id": patient_id,
                "fips": fips,
                "risk_score": risk_score,
                "report": report
            }

            return redirect(url_for("provider_medicalreportuplload"))

    return render_template(
        "provider.html",
        provider=provider,
        patient_id=patient_id,
        fips=fips,
        risk_score=risk_score,
        report=report
    )
@app.route("/generate_report", methods=["POST"])
def generate_report():
    try:
        data = request.get_json()
        patient_id = data.get("patient_id")
        fips = data.get("fips")

        if not patient_id or not fips:
            return jsonify({"error": "Missing patient_id or fips"}), 400

        report, risk_score = None, None
        if patient_df is not None and "Patient_ID" in patient_df.columns:
            row = patient_df.loc[patient_df["Patient_ID"].astype(str) == str(patient_id)]
            if not row.empty:
                patient_data = row.iloc[0].to_dict()

                if meta_model is not None:
                    try:
                        X = pd.DataFrame([patient_data])
                        risk_score = round(float(meta_model.predict_proba(X)[0][1]) * 100, 2)
                    except Exception as e:
                        logger.error(f"Meta model error: {e}")

                report = explain_and_generate_llm_report(patient_data, patient_data)

                return jsonify({
                    "success": True,
                    "patient_id": patient_id,
                    "fips_code": fips,
                    "risk_score": risk_score,
                    "generated_report": report
                })

        return jsonify({"error": "Patient not found"}), 404

    except Exception as e:
        logger.error(f"Error generating report: {e}")
        return jsonify({"error": "Internal server error"}), 500
@app.route("/download_pdf_report")
def download_pdf_report():
    pid = request.args.get("patient_id")
    row = patient_df.loc[patient_df["Patient_ID"] == pid]
    if row.empty:
        return "Patient not found", 404
    report_text = explain_and_generate_llm_report(row.iloc[0].to_dict(), row.iloc[0].to_dict())
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    story = [Paragraph("Patient Risk Report", styles["Title"]), Spacer(1, 12)]
    for line in report_text.split("\n"):
        story.append(Paragraph(line, styles["BodyText"]))
    doc.build(story)
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name=f"Report_{pid}.pdf")
@app.route("/ngo/signup", methods=["GET", "POST"])
def ngosignup():
    if request.method == "POST":
        data = request.get_json()

        full_name = data.get("full_name")
        email = data.get("email")
        password = data.get("password")
        confirm_password = data.get("confirm_password")
        designation = data.get("designation")
        organization = data.get("organization")
        terms = data.get("terms")

        if not all([full_name, email, password, confirm_password, designation, organization, terms]):
            return jsonify({"success": False, "message": "Please fill out all fields and agree to terms."}), 400

        if password != confirm_password:
            return jsonify({"success": False, "message": "Passwords do not match."}), 400

        success, user, message = auth_manager.register_ngo(
            full_name=full_name,
            email=email,
            password=password,
            designation=designation,
            organization=organization
        )

        if success:
            session.update({
                "user_id": user["id"],
                "user_type": "ngo",
                "user_name": user["full_name"],
                "user_email": user["email"]
            })

            try:
                import ngo_dash
                if hasattr(ngo_dash, "register_dash"):
                    ngo_dash.register_dash(app)
            except Exception as e:
                logger.error(f"Failed to register ngo_dash: {e}")

            return jsonify({"success": True, "message": message, "redirect": "/ngo/dashboard/"})

        return jsonify({"success": False, "message": message}), 400

    return render_template("ngosignup.html")

@app.route("/provider/login", methods=["GET", "POST"])
def provider_login():
    if request.method == "POST":
        data = request.get_json()
        success, user, message = auth_manager.authenticate_provider(
            data.get("email"), data.get("password")
        )
        if success:
            session.update({
                "user_id": user["id"],
                "user_type": "provider",
                "user_name": user["full_name"],
                "user_email": user["email"],
                "provider": user  
            })

            return jsonify({"success": True, "message": message, "redirect": "/provider"})
        return jsonify({"success": False, "message": message}), 401
    return render_template("provider_login.html")


@app.route("/provider/dashboard")
@login_required
def provider_dashboard():
    if session.get("user_type") != "provider":
        return redirect(url_for("landing_page"))
    return render_template("provider.html", provider=get_current_user())

@app.route("/chatbot")
def chatbot():
    return render_template("chatbot.html")

@app.route('/medicalreportuplload.html')
def medical_report_upload_page():
    return render_template('medicalreportuplload.html')


@app.route("/analyze_report", methods=["POST"])
@login_required
def analyze_report():
    if session.get("user_type") != "provider":
        return redirect(url_for("landing_page"))

    file = request.files.get("reportFile")
    if not file:
        return render_template(
            "medicalreportuplload.html",
            error="No file uploaded",
            provider=get_current_user()
        )

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    try:
        reader = PdfReader(filepath)
        text = "".join([page.extract_text() or "" for page in reader.pages])
    except Exception as e:
        return render_template(
            "medicalreportuplload.html",
            error=f"PDF read error: {e}",
            provider=get_current_user()
        )

    intervention_features = {
        "blood_pressure": 1 if "blood pressure" in text.lower() else 0,
        "cholesterol": 1 if "cholesterol" in text.lower() else 0,
        "food_assistance": 1 if "food assistance" in text.lower() else 0,
        "urban": 1 if "urban" in text.lower() else 0
    }
    X = pd.DataFrame([intervention_features])

    if intervention_model is not None:
        try:
            X = X.reindex(columns=intervention_features, fill_value=0)

            initial_risk = round(float(intervention_model.predict(X)[0]), 2)

            predicted_risk_after = max(initial_risk - 20.0, 0)
            risk_reduction = round(initial_risk - predicted_risk_after, 2)

            interventions_found = [k for k, v in intervention_features.items() if v == 1]
        except Exception as e:
            return render_template(
                "medicalreportuplload.html",
                error=f"Model prediction error: {e}",
                provider=get_current_user()
            )
    else:
        initial_risk, predicted_risk_after = 72.5, 45.0
        risk_reduction = 27.5
        interventions_found = ["Nutritional Support", "Follow-up Visits"]

    return render_template(
        "medicalreportuplload.html",
        provider=get_current_user(),
        initial_risk=initial_risk,
        predicted_risk_after=predicted_risk_after,
        risk_reduction=risk_reduction,
        interventions_found=interventions_found
    )


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("landing_page"))


try:
    import ngo_dash  
    if hasattr(ngo_dash, "register_dash"):
        ngo_dash.register_dash(app)
    else:
        logger.warning("ngo_dashboard.py found but register_dash(app) not present.")
except Exception as e:
    logger.warning(f"Could not import/register ngo_dashboard: {e}")


if __name__ == "__main__":
    app.run(debug=True)
