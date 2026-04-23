from flask import Flask, render_template, request, jsonify
from modules.object_detection import detect_objects
from modules.model_comparison import compare_models
from modules.currency_detection import detect_currency_image
from modules.currency_detection import detect_currency_live

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/object")
def object_page():
    return render_template("object.html")

@app.route("/detect", methods=["POST"])
def detect():
    file = request.files["image"]
    image_bytes = file.read()

    detections = detect_objects(image_bytes)

    return jsonify({"detections": detections})
@app.route("/compare")
def compare_page():
    return render_template("compare.html")


@app.route("/compare_models", methods=["POST"])
def compare():
    file = request.files["image"]
    image_bytes = file.read()

    results = compare_models(image_bytes)

    return jsonify(results)

@app.route("/currency")
def currency_page():
    return render_template("currency.html")


@app.route("/detect_currency", methods=["POST"])
def detect_currency_api():
    file = request.files["image"]
    image_bytes = file.read()

    detections = detect_currency_live(image_bytes)

    return jsonify({"detections": detections})

@app.route("/detect_currency_image", methods=["POST"])
def detect_currency_image_api():
    file = request.files["image"]
    image_bytes = file.read()

    img, labels = detect_currency_image(image_bytes)

    return jsonify({
        "image": img,
        "labels": labels
    })


@app.route("/currency_live")
def currency_live():
    return render_template("currency_live.html")

@app.route("/navigation")
def navigation_page():
    return render_template("navigation.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)