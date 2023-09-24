from flask import Flask, request, render_template
from transformers import pipeline
from sacremoses import MosesTokenizer, MosesDetokenizer

app = Flask(__name__)

translation_pipeline = pipeline("translation", model="Helsinki-NLP/opus-mt-en-fi")

tokenizer = MosesTokenizer('fi')  # Specify the language if needed
detokenizer = MosesDetokenizer('fi')  # Specify the language if needed


def preprocess_text(input_text):
    input_text = ' '.join(input_text.split())
    input_text_tokens = tokenizer.tokenize(input_text)
    input_text = detokenizer.detokenize(input_text_tokens)
    return input_text


@app.route('/', methods=['GET', 'POST'])
def translate_text():  # put application's code here
    translated_text = ""

    if request.method == 'POST':
        input_text = request.form['text_field']
        translations = translation_pipeline(input_text)

        if translations and 'translation_text' in translations[0]:
            translated_text = translations[0]['translation_text']

    cleaned_text = preprocess_text(translated_text)
    return render_template('index.html', translated_text=cleaned_text)


if __name__ == '__main__':
    app.run()
