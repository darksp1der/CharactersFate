from flask import Flask, request, jsonify, send_from_directory
from concurrent.futures import ThreadPoolExecutor, TimeoutError as TE
from predict import predict_status, query_ollama

app = Flask(__name__)
pool = ThreadPoolExecutor(max_workers=2)

@app.route("/")
def index():
    return send_from_directory("frontend", "index.html")

@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory("frontend", filename)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    text = data.get("text", "")
    status, confidence, entropy, margin, logits = predict_status(text)
    return jsonify({
        "status": status,
        "confidence": confidence,
        "entropy": entropy,
        "margin": margin,
        "logits": logits,
        "text": text
    })

@app.route("/explain", methods=["POST"])
def explain():
    data = request.get_json(force=True)
    text = data.get("text", "")
    status = data.get("status", "")

    prompt = (
        f"Here is a character history:\n\"{text}\"\n"
        f"Based on this, the model predicted the character is '{status}'. "
        "Why would this be the case? Explain in a short paragraph."
    )

    try:
        explanation = pool.submit(lambda: query_ollama(prompt)).result(timeout=25)
    except TE:
        explanation = "⚠️ Ollama timed out."

    return jsonify({"explanation": explanation.strip()})


if __name__ == "__main__":
    app.run(debug=True)
