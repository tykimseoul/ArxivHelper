from flask import Flask, json, request
from bs4 import BeautifulSoup
import requests
import re

api = Flask(__name__)
base_url = 'https://arxiv.org/abs/{}'


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
    abstract = document.select_one('blockquote.abstract').findAll(text=True, recursive=False)
    abstract = list(filter(lambda a: a != '\n', abstract))
    abstract = ' '.join(abstract)
    abstract = re.sub(r'[\n\r]', ' ', abstract)
    abstract = abstract.strip()
    abstract = re.sub(r'\s+', ' ', abstract)
    print(title)
    print(authors)
    print(pdf_link)
    print(abstract)
    return {'title': title, 'authors': authors, 'pdf_link': pdf_link, 'abstract': abstract}


@api.route('/arxiv', methods=['GET'])
def get_paper_data():
    code = request.args.get('code')
    data = parse_page(code)
    return json.dumps(data)


if __name__ == '__main__':
    api.run()
