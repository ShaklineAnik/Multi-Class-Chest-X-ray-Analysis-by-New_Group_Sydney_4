import os
import uuid
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename

from config import UPLOAD_FOLDER, RESULT_FOLDER, ALLOWED_EXTENSIONS, CLASS_NAMES
from model_utils import load_model, predict_image
from gradcam_utils import create_gradcam_image
from explanation_utils import generate_research_explanation

app = Flask(__name__)
app.secret_key = "cxr-explainable-ai-secret"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs("models", exist_ok=True)

model, device, model_loaded = load_model()


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template(
        "index.html",
        model_loaded=model_loaded,
        class_names=CLASS_NAMES
    )


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        flash("No image file selected.")
        return redirect(url_for("index"))

    file = request.files["image"]

    if file.filename == "":
        flash("Please choose a chest X-ray image.")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash("Unsupported file type. Use PNG, JPG, JPEG, BMP, or WEBP.")
        return redirect(url_for("index"))

    filename = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{filename}"
    upload_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
    file.save(upload_path)

    if not model_loaded:
        flash("Model file not found. Place your trained best_model.pt inside the models/ folder.")
        return render_template(
            "result.html",
            model_loaded=False,
            uploaded_image=url_for("static", filename=f"uploads/{unique_name}"),
            prediction=None,
            gradcam_image=None,
            explanation=None
        )

    prediction = predict_image(model, device, upload_path)

    gradcam_name = f"gradcam_{unique_name}.png"
    gradcam_path = create_gradcam_image(model, device, upload_path, gradcam_name)

    explanation = generate_research_explanation(prediction)

    return render_template(
        "result.html",
        model_loaded=True,
        uploaded_image=url_for("static", filename=f"uploads/{unique_name}"),
        gradcam_image=url_for("static", filename=f"results/{gradcam_name}"),
        prediction=prediction,
        explanation=explanation
    )


@app.route("/predict_json", methods=["POST"])
def predict_json():
    if "image" not in request.files:
        return jsonify({"ok": False, "error": "No image file selected."}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"ok": False, "error": "Please choose a chest X-ray image."}), 400

    if not allowed_file(file.filename):
        return jsonify({"ok": False, "error": "Unsupported file type. Use PNG, JPG, JPEG, BMP, or WEBP."}), 400

    filename = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{filename}"
    upload_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
    file.save(upload_path)

    if not model_loaded:
        return jsonify({
            "ok": False,
            "model_loaded": False,
            "error": "Model file not found. Place your trained best_model.pt inside the models/ folder.",
            "uploaded_image": url_for("static", filename=f"uploads/{unique_name}")
        }), 200

    prediction = predict_image(model, device, upload_path)

    gradcam_name = f"gradcam_{unique_name}.png"
    create_gradcam_image(model, device, upload_path, gradcam_name)

    explanation = generate_research_explanation(prediction)

    return jsonify({
        "ok": True,
        "model_loaded": True,
        "uploaded_image": url_for("static", filename=f"uploads/{unique_name}"),
        "gradcam_image": url_for("static", filename=f"results/{gradcam_name}"),
        "prediction": prediction,
        "explanation": explanation
    })


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
