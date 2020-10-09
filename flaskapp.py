from flask import Flask, json, request
from bs4 import BeautifulSoup
import requests
import re
import time
import fitz
# import numpy as np
from pathlib import Path
from PIL import Image
import base64
from io import BytesIO
import os

app = Flask(__name__)
base_url = 'https://arxiv.org/abs/{}'
thumbnails_dir = Path("/tmp/thumbnails")
thumbnails_dir.mkdir(parents=True, exist_ok=True)

#
#
# class Thumbnail:
#     def __init__(self, image):
#         self.channels = 3
#         self.size = (image.width, image.height)
#         self.image = image.samples
#
#     def get_image(self):
#         image = np.frombuffer(self.image, dtype=np.uint8)
#         return image.reshape(*self.size, self.channels)


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
    pdf_link = document.select_one('div.extra-services > div.full-text > ul > li > a.download-pdf').get('href')
    pdf_link = re.sub(r'^/pdf/', '', pdf_link)
    abstract = document.select_one('blockquote.abstract').findAll(text=True, recursive=False)
    abstract = list(filter(lambda a: a != '\n', abstract))
    abstract = ' '.join(abstract)
    abstract = re.sub(r'[\n\r]', ' ', abstract)
    abstract = abstract.strip()
    abstract = re.sub(r'\s+', ' ', abstract)
    thumbnail = extract_thumbnail(code)
    # thumbnail = Image.fromarray(thumbnail)
    buffer = BytesIO()
    thumbnail.save(buffer, format="JPEG")
    thumbnail = base64.b64encode(buffer.getvalue()).decode('utf-8')
    print(title)
    print(authors)
    print(pdf_link)
    print(abstract)
    print(thumbnail)
    return {'title': title, 'authors': authors, 'pdf': pdf_link, 'abstract': abstract, 'thumbnail': thumbnail}


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
    # image.writeImage(key, 'jpg')
    # env = lmdb.open(str(lmdb_dir), map_size=int(1e9))
    #
    # # Start a new write transaction
    # with env.begin(write=True) as txn:
    #     # All key-value pairs need to be strings
    #     value = Thumbnail(image)
    #     txn.put(key.encode("ascii"), pickle.dumps(value))
    # env.close()


def read_thumbnail(key):
    return Image.open('{}/{}.jpg'.format(str(thumbnails_dir), key))

    # env = lmdb.open(str(lmdb_dir), readonly=True)
    #
    # # Start a new read transaction
    # with env.begin() as txn:
    #     # Encode the key the same way as we stored it
    #     data = txn.get(key.encode("ascii"))
    #
    #     # Remember it's a CIFAR_Image object that is loaded
    #     image = pickle.loads(data)
    #     # Retrieve the relevant bits
    #     thumbnail = image.get_image()
    # env.close()
    #
    # return thumbnail


def check_thumbnail(key):
    return '{}.jpg'.format(key) in os.listdir(str(thumbnails_dir))
    # env = lmdb.open(str(lmdb_dir), readonly=True)
    #
    # # Start a new read transaction
    # with env.begin() as txn:
    #     # Encode the key the same way as we stored it
    #     data = txn.get(key.encode("ascii"))
    #     return data is not None


if __name__ == '__main__':
    app.run(ssl_context='adhoc')
