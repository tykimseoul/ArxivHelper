from flask import Flask, json, request
from bs4 import BeautifulSoup
import requests
import re
import time
import fitz
from pathlib import Path
from PIL import Image
import base64
from io import BytesIO
import os

app = Flask(__name__)
base_url = 'https://arxiv.org/abs/{}'
thumbnails_dir = Path("/tmp/thumbnails")
thumbnails_dir.mkdir(parents=True, exist_ok=True)


def get_html(url, t):
    print(url)
    try:
        return requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
    except requests.exceptions.ConnectionError:
        print('pausing for {}'.format(url))
        time.sleep(t)
        return get_html(url, t + 5)


def parse_page(code):
    html = get_html(base_url.format(code), 5)
    document = BeautifulSoup(html.text, "html.parser")
    title = document.select_one('h1.title').text
    authors = list(map(lambda a: a.text, document.select('div.authors > a')))
    abstract = document.select_one('blockquote.abstract').findAll(text=True, recursive=False)
    abstract = list(filter(lambda a: a != '\n', abstract))
    abstract = ' '.join(abstract)
    abstract = re.sub(r'[\n\r]', ' ', abstract)
    abstract = abstract.strip()
    abstract = re.sub(r'\s+', ' ', abstract)
    thumbnail = extract_thumbnail(code)
    buffer = BytesIO()
    thumbnail.save(buffer, format="JPEG")
    thumbnail = base64.b64encode(buffer.getvalue()).decode('utf-8')
    print(title)
    print(authors)
    print(abstract)
    print(thumbnail)
    return {'title': title, 'authors': authors, 'abstract': abstract, 'thumbnail': thumbnail}


@app.route('/', methods=['GET'])
def index():
    return 'hello world'


@app.route('/arxiv', methods=['GET'])
def get_paper_data():
    code = request.args.get('code')
    data = parse_page(code)
    return json.dumps(data)


def extract_thumbnail(code):
    if not check_thumbnail(code):
        response = requests.get("https://arxiv.org/pdf/{}.pdf".format(code))
        with open('/tmp/downloaded.pdf', 'wb') as f:
            f.write(response.content)
            doc = fitz.open('/tmp/downloaded.pdf')
            for i in range(len(doc)):
                images = doc.getPageImageList(i)
                if len(images) > 0:
                    img = images[0]
                    print(img)
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    store_thumbnail(pix, code)
                    break
    return read_thumbnail(code)


def store_thumbnail(pix, key):
    pix.writeImage('{}/{}.jpg'.format(str(thumbnails_dir), key))


def read_thumbnail(key):
    return Image.open('{}/{}.jpg'.format(str(thumbnails_dir), key))


def check_thumbnail(key):
    return '{}.jpg'.format(key) in os.listdir(str(thumbnails_dir))


if __name__ == '__main__':
    app.run(ssl_context='adhoc')
