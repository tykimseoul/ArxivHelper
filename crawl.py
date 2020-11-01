import requests
from bs4 import BeautifulSoup
import re
import fitz
from pathlib import Path
from PIL import Image

from flaskapp import get_html

arxiv_base_link = 'https://arxiv.org/{}'
temp_dir = Path("./tmp")
temp_dir.mkdir(parents=True, exist_ok=True)
covers_dir = Path("./tmp/covers")
covers_dir.mkdir(parents=True, exist_ok=True)
a4_dir = Path("./tmp/covers/a4")
a4_dir.mkdir(parents=True, exist_ok=True)
letter_dir = Path("./tmp/covers/letter")
letter_dir.mkdir(parents=True, exist_ok=True)


def crawl(start_link, count=0):
    print(count)
    if count == target_count:
        return
    html = get_html(start_link, 5)
    document = BeautifulSoup(html.text, "html.parser")
    paper_id = document.select_one('div.extra-services > div.full-text > ul > li > a.download-pdf').get('href')
    paper_id = re.sub(r'^/pdf/', '', paper_id)
    response = requests.get('https://arxiv.org/pdf/{}'.format(paper_id))
    save_pdf(response, paper_id)
    count += 1
    next_link = document.select_one('span.arrow > a.next-url').get('href')
    next_link = get_html('https://arxiv.org/{}'.format(next_link), 5).url
    crawl(next_link, count)


def save_pdf(response, key):
    with open('{}/paper.pdf'.format(str(temp_dir)), 'wb') as f:
        f.write(response.content)
        doc = fitz.open('{}/paper.pdf'.format(str(temp_dir)))
        pix = doc[0].getPixmap(alpha=False)
        if (pix.width, pix.height) == (612, 792):
            paper_size = 'letter'
        else:
            paper_size = 'a4'
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples).convert('L')
        file_name = re.sub(r'\.', '_', key)
        img.save('{}/{}/{}.png'.format(str(covers_dir), paper_size, file_name), 'PNG')


if __name__ == '__main__':
    target_count = 200
    crawl('https://arxiv.org/abs/2010.12260')
