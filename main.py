from flask import Flask, request, jsonify, render_template
from transformers import pipeline

app = Flask(__name__)

# Inicijalizacija lokalnog modela (ovo može biti Mistral ili drugi dostupni model)
qa_pipeline = pipeline("text-generation", model="tiiuae/falcon-7b-instruct", device=-1)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    user_input = request.json.get("question", "")
    if not user_input:
        return jsonify({"answer": "Molim vas postavite pitanje."})
    
    try:
        result = qa_pipeline(user_input, max_length=200, do_sample=True, temperature=0.7)[0]["generated_text"]
        return jsonify({"answer": result})
    except Exception as e:
        return jsonify({"answer": f"Došlo je do greške: {str(e)}"})

if __name__ == "__main__":
    app.run(debug=True)