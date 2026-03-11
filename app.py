import gradio as gr
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# Load model from HuggingFace
model_path = "Deba1207/summariseAI"
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

def summarize(text):
    # Validate input
    if not text.strip():
        return "⚠️ Please enter some text to summarize!"

    try:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=1024
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
        return summary

    except Exception as e:
        return f"❌ Error: {str(e)}"


iface = gr.Interface(
    fn=summarize,
    inputs=gr.Textbox(
        lines=10,
        placeholder="Paste your text here...",
        label="📝 Input Text"
    ),
    outputs=gr.Textbox(
        lines=5,
        label="📄 Summary"
    ),
    title="🤖 AI Text Summarizer",
    description="Paste any long text and get an AI-generated summary instantly!",
    examples=[
        ["Artificial intelligence is transforming the world. It is being used in healthcare, finance, education and many other fields to automate tasks and improve efficiency."]
    ],
    theme=gr.themes.Soft()
)

iface.launch()
