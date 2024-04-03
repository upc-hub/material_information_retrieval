from flask import Flask, render_template, request, send_from_directory
import os
import PyPDF2
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import fitz  # PyMuPDF

import spacy
from spacy.training import Example  # Import Example class
import random
from spacy.util import minibatch

from bs4 import BeautifulSoup
import requests
app = Flask(__name__)

PDF_FOLDER = os.path.join(app.static_folder, 'pdf')
model_output_path = '/Users/soethandara/Desktop/test/material_science_ner_finetuned'

def scrape_web_page(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raises an HTTPError if the response status code is 4XX/5XX
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Example: Extract all paragraph texts
        paragraphs = [p.text for p in soup.find_all('p')]
        print (paragraphs)
        return paragraphs
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return []
    
def test_model(model_path, test_data):
    # Load the fine-tuned model
    nlp = spacy.load(model_path)
    # Test the model on the test data
    #for text in test_data:
    doc = nlp(test_data)
    material, propety, application = [],[],[] 
    print("Text:", test_data)
    print("Entities and Labels:")
    for ent in doc.ents:
        if ent.label_ == "MATERIAL":
            material.append(ent.text)
        elif ent.label_ == "PROPERTY":
            propety.append(ent.text)
        else:
            application.append(ent.text)
        print(f" - {ent.text} ({ent.label_})")
    print(material)
    print(propety)
    print(application)
    return material, propety, application
        
def extract_paragraphs_from_pdf(file_path):
    paragraphs = []
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text = page.extract_text()
            if text:  # Check if text extraction was successful
                paragraphs.extend(text.split('\n\n'))
    return paragraphs

def split_paragraph(paragraph):
    # Split the paragraph into lines
    lines = paragraph.split('\n')
    
    # Split lines into chunks of maximum 10 lines each
    chunks = [lines[i:i+10] for i in range(0, len(lines), 10)]
    
    # Join each chunk back into a paragraph
    paragraphs = ['\n'.join(chunk) for chunk in chunks]
    
    return paragraphs


def load_data(data_folder):
    paragraphs = []
    file_names = []
    topics = []  # List to store the topic of each paragraph
    for root, dirs, files in os.walk(data_folder):
        for file in files:
            if file.endswith(".pdf"):
                file_path = os.path.join(root, file)
                pdf_paragraphs = extract_paragraphs_from_pdf(file_path)
                topic = os.path.basename(root)  # Assuming the folder name is the topic
                paragraphs.extend(pdf_paragraphs)
                file_names.extend([file] * len(pdf_paragraphs))
                topics.extend([topic] * len(pdf_paragraphs))  # Associate each paragraph with its topic
    url_list = [
        "https://en.wikipedia.org/wiki/Materials_science",
        "https://a-lab.material.nagoya-u.ac.jp/en/"   
    ]
    
    #url = "https://en.wikipedia.org/wiki/Materials_science"
    for url in url_list:
        web_paragraphs = scrape_web_page(url)
        paragraphs.extend(web_paragraphs)
        file_names.extend(url)
        topics.extend(url)
    print (topics)
    return paragraphs, file_names, topics

def find_most_relevant_paragraph(query, paragraphs, file_names, topics, vectorizer):
    query_vec = vectorizer.transform([query])
    sim_scores = cosine_similarity(query_vec, X).flatten()
    most_similar_idx = np.argmax(sim_scores)
    return paragraphs[most_similar_idx], file_names[most_similar_idx], topics[most_similar_idx]

def generate_thumbnail(pdf_path, output_path):
    # Open the PDF file
    doc = fitz.open(pdf_path)
    # Get the first page
    page = doc.load_page(0)
    # Generate thumbnail (scale down to 100x100)
    pix = page.get_pixmap(matrix=fitz.Matrix(0.1, 0.1))
    # Save thumbnail image
    pix.save(output_path)
    # Close the document
    doc.close()

# Path to the folder containing subfolders for each topic
data_folder = "/Users/soethandara/Desktop/search_app/static/pdf"
#Titanium dioxide is used in sunscreen for UV protection.
# Load data from PDF files
paragraphs, file_names, topics = load_data(data_folder)

# Preprocessing: Tokenization and vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(paragraphs)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query = request.form['query']
        most_similar_paragraph, file_name, topic = find_most_relevant_paragraph(query, paragraphs, file_names, topics, vectorizer)
        #topic1 = [item for item in topic if len(item) > 1]
        #topic1 = ["web" if len(item) == 1 else item for item in topic]
        mat, pro, appli = test_model(model_output_path, query)
        return render_template('search_results.html', paragraph=most_similar_paragraph, file_name=file_name, topic=topic, query=query, material=mat, properties=pro, application=appli)
    #print (set(topics))  
    pdf_file = []  
    for root, dirs, files in os.walk(data_folder):
        for file in files:
            if file.endswith(".pdf"):
                file_path = os.path.join(root, file)
                parts = file_path.split('/')
                last_two_parts = parts[-2:]
                pdf_file.append(last_two_parts[0]+"/"+last_two_parts[1])
    print (pdf_file)
    #topic_files = []
    #for to in set(topics):
    #    print (to)
    #    topic_files = list(set([file_name for file_name, file_topic in zip(file_names, to) if file_topic == to]))
    #    print (topic_files)
        #for file_name, file_topic in zip(file_names, to):
        #    print (file_name)
        #    if to+"/"+file_name not in topic_files:
        #        topic_files.append(to+"/"+file_name)
        
    #print(topic_files)
    print (set())
    #topics1 = [item for item in topics if len(item) > 1]
    topics1 = ["web" if len(item) == 1 else item for item in topics]
    url_list = [
        "https://en.wikipedia.org/wiki/Materials_science",
        "https://a-lab.material.nagoya-u.ac.jp/en/",
        "https://connect.acspubs.org/materials-science-authors?utm_source=googl&utm_medium=sem&utm_campaign=IC001_ST0002D_T000457_Materials_Science_Ad_Program&src=IC001_ST0002D_T000457_Materials_Science_Ad_Program&utm_content=&gad_source=1&gclid=Cj0KCQiArrCvBhCNARIsAOkAGcVS8-sZucGSy2zamSIwraHUc5WLNpQ_farA1Y2b6cU0xDZGmnkXgVcaAl-HEALw_wcB",
        "https://www.brookesbell.com/news-and-knowledge/article/what-are-materials-science-and-materials-testing-and-why-are-they-important-157751/?gad_source=1&gclid=Cj0KCQiArrCvBhCNARIsAOkAGcX3HQgsvuvERfY_aKFmj0EgnMsG94Ujf7BXRR7FBkK5UdtneY25O2gaApYDEALw_wcB",
        "https://www.britannica.com/technology/materials-science",
        "https://link.springer.com/journal/11003"   
    ]
    return render_template('index.html', topics=set(topics1), files=pdf_file, web_urls=url_list)

@app.route('/pdf/<filename>')
def pdf_viewer(filename):
    return send_from_directory(PDF_FOLDER, filename)

@app.route('/pdf/<topic>/<filename>')
def pdf_viewer_1(topic, filename):
    # Construct the path to the requested PDF file
    filepath = os.path.join(PDF_FOLDER, topic, filename)
    
    # Check if the file exists
    if os.path.exists(filepath):
        # Serve the file from the specified folder
        return send_from_directory(PDF_FOLDER, f'{topic}/{filename}')
    else:
        # Return a 404 error if the file does not exist
        return 'File not found', 404

@app.route('/topic/<topic>')
def topic_files(topic):
    # Filter unique file names for the given topic
    topic_files = list(set([file_name for file_name, file_topic in zip(file_names, topics) if file_topic == topic]))
    
    # Generate thumbnails for PDF files
    thumbnail_folder = os.path.join(app.static_folder, 'thumbnails')
    os.makedirs(thumbnail_folder, exist_ok=True)
    for file_name in topic_files:
        pdf_path = os.path.join(PDF_FOLDER, topic, file_name)
        thumbnail_path = os.path.join(thumbnail_folder, f'{file_name}.png')
        generate_thumbnail(pdf_path, thumbnail_path)
    print (topic_files)
    return render_template('topic_files.html', topic=topic, files=topic_files)


if __name__ == '__main__':
    app.run(debug=True, port="5003")
