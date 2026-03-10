from flask import Flask, request, render_template, jsonify
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

app = Flask(__name__)



model_path = "Deba1207/summariseAI" 
tokenizer = AutoTokenizer.from_pretrained(model_path,use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/summarize", methods=["POST"])
def summarize():
    text = request.form.get("text", "")
    
    if not text.strip():
        return jsonify({"error": "Input text is empty!"}), 400

    try:
        
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=1024
        )

        
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=512,
            min_length=60,  
            num_beams=4,
            early_stopping=False,  
            length_penalty=2.0,   
            no_repeat_ngram_size=3  
        )

        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        return jsonify({"summary": summary})

    except Exception as e:
        print(f"Error during summarization: {e}")
        return jsonify({"error": "Failed to generate summary."}), 500

if __name__ == "__main__":
    app.run(debug=True)
