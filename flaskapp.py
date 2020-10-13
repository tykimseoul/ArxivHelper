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


def parse_cvf_page(url):
    html = get_html(url, 5)
    document = BeautifulSoup(html.text, "html.parser")
    title = document.select_one('div#papertitle').text
    authors = document.select_one('div#authors > b > i').text
    authors = authors.split(', ')
    abstract = document.select_one('div#abstract').text
    thumbnail = extract_cvf_thumbnail(url)
    buffer = BytesIO()
    thumbnail.save(buffer, format="JPEG")
    thumbnail = base64.b64encode(buffer.getvalue()).decode('utf-8')
    print(title)
    print(authors)
    print(abstract)
    print(thumbnail)
    return {'title': title, 'authors': authors, 'abstract': abstract, 'thumbnail': thumbnail}


def extract_cvf_thumbnail(url):
    code = re.match(r'https://openaccess.thecvf.com/(\w+)/html/([\w-]+).html', url).groups()
    if not check_thumbnail(code[1]):
        response = requests.get('https://openaccess.thecvf.com/{}/papers/{}.pdf'.format(code[0], code[1]))
        store_thumbnail(response, code[1])
    return read_thumbnail(code[1])


@app.route('/', methods=['GET'])
def index():
    return 'hello world'


@app.route('/paper', methods=['GET'])
def get_paper_data():
    link = request.args.get('link')
    if re.match(r'https://openaccess.thecvf.com/(\w+)/html/([\w-]+).html', link):
        data = parse_cvf_page(link)
    else:
        data = {}
    return json.dumps(data)


def store_thumbnail(response, key):
    with open('downloaded.pdf', 'wb') as f:
        f.write(response.content)
        doc = fitz.open('downloaded.pdf')
        for i in range(len(doc)):
            images = doc.getPageImageList(i)
            if len(images) > 0:
                img = images[0]
                print(img)
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)
                pix = fitz.Pixmap(pix, 0)
                pix.writeImage('{}/{}.jpg'.format(str(thumbnails_dir), key))
                break


def read_thumbnail(key):
    return Image.open('{}/{}.jpg'.format(str(thumbnails_dir), key))


def check_thumbnail(key):
    return '{}.jpg'.format(key) in os.listdir(str(thumbnails_dir))


if __name__ == '__main__':
    app.run(ssl_context='adhoc')
