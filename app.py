from flask import Flask, render_template, request
from transformers import TFT5ForConditionalGeneration, RobertaTokenizer
import random
import tensorflow as tf
from pathlib import Path

app = Flask(__name__)

class Args:
    # Define training arguments

    # MODEL
    model_type = 't5'
    tokenizer_name = 'Salesforce/codet5-base'
    model_name_or_path = 'Salesforce/codet5-base'

    # DATA
    train_batch_size = 8
    validation_batch_size = 8
    max_input_length = 48
    max_target_length = 128
    prefix = "Generate Python: "

    # DIRECTORIES
    save_dir = "saved_model/"

args = Args()

def run_predict(args, text):
    # Load saved finetuned model
    model = TFT5ForConditionalGeneration.from_pretrained(args.save_dir)
    # Load saved tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(args.save_dir)

    # Encode texts by prepending the task for input sequence and appending the test sequence
    query = args.prefix + text
    encoded_text = tokenizer(query, return_tensors='tf', padding='max_length', truncation=True, max_length=args.max_input_length)

    # Inference
    generated_code = model.generate(
        encoded_text["input_ids"], attention_mask=encoded_text["attention_mask"],
        max_length=args.max_target_length, top_p=0.95, top_k=50, repetition_penalty=2.0, num_return_sequences=1
    )

    # Decode generated tokens
    decoded_code = tokenizer.decode(generated_code.numpy()[0], skip_special_tokens=True)
    print(decoded_code)
    return decoded_code

def predict_from_text(args, text):
    # Run predict on text
    decoded_code = run_predict(args, text)
    return decoded_code

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    prediction = predict_from_text(args, text)
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
