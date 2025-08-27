from flask import Flask, request, render_template, send_from_directory
import os
import shutil
import fitz
from transformers import pipeline
from werkzeug.utils import secure_filename

# ----------------- Agent 1: Classification Agent ----------------- #
class ClassificationAgent:
    def __init__(self, categories):
        self.categories = categories
        self.classifier = pipeline("zero-shot-classification", model="MoritzLaurer/DeBERTa-v3-base-mnli")
    
    def extract_text(self, pdf_path):
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text

    def classify(self, text):
        result = self.classifier(text, candidate_labels=self.categories)
        return result['labels'][0]

# ----------------- Agent 2: Organization Agent ----------------- #
class OrganizationAgent:
    def __init__(self, base_folder, categories):
        self.base_folder = base_folder
        self.categories = categories
        self._prepare_folders()

    def _prepare_folders(self):
        os.makedirs(self.base_folder, exist_ok=True)
        for category in self.categories:
            os.makedirs(os.path.join(self.base_folder, category), exist_ok=True)

    def move_file(self, pdf_path, category):
        filename = os.path.basename(pdf_path)
        destination = os.path.join(self.base_folder, category, filename)
        shutil.move(pdf_path, destination)
        return destination

# ----------------- Agent 3: Content Summarizer Agent ----------------- #
class SummarizerAgent:
    def __init__(self):
        self.summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    
    def summarize(self, text, max_length=150, min_length=40):
        if len(text) > 1024:
            text = text[:1024]
        summary = self.summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]['summary_text']

# ----------------- Flask App ----------------- #
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'incoming_pdfs'
app.config['CLASSIFIED_FOLDER'] = 'classified_pdfs'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

categories = ['Finance', 'Legal', 'HR', 'Technical', 'Marketing', 'Operations']
classifier_agent = ClassificationAgent(categories)
organizer_agent = OrganizationAgent(app.config['CLASSIFIED_FOLDER'], categories)
summarizer_agent = SummarizerAgent()

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/assets/<path:filename>')
def serve_asset(filename):
    asset_folder = os.path.abspath("assets")
    return send_from_directory(asset_folder, filename)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    category = None
    summary = None
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(pdf_path)
            
            text = classifier_agent.extract_text(pdf_path)
            category = classifier_agent.classify(text)
            summary = summarizer_agent.summarize(text)
            organizer_agent.move_file(pdf_path, category)
    
    return render_template('template.html', category=category, summary=summary)

if __name__ == '__main__':
    app.run(debug=True)
