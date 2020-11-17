import requests
from bs4 import BeautifulSoup
import re
import fitz
from pathlib import Path
from PIL import Image
from pdfminer.layout import LAParams, LTTextBox, LTTextLine
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
import jellyfish
import time
from random import randint
import pandas as pd
import json
import schedule

import numpy as np


def get_html(url, t):
    print(url)
    return requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)


arxiv_base_link = 'https://arxiv.org/{}'
temp_dir = Path("./tmp")
temp_dir.mkdir(parents=True, exist_ok=True)
covers_dir = Path("./tmp/covers_nn")
covers_dir.mkdir(parents=True, exist_ok=True)
masks_dir = Path("./tmp/masks")
masks_dir.mkdir(parents=True, exist_ok=True)
redaction_dir = Path("./tmp/redaction_nn")
redaction_dir.mkdir(parents=True, exist_ok=True)
redaction_label_dir = Path("./tmp/redaction_label_nn")
redaction_label_dir.mkdir(parents=True, exist_ok=True)


class TextArea:
    def __init__(self, text=None, bbox=None):
        self.text = text
        self.bbox = bbox

    def __str__(self):
        return 'TextArea({}, {})'.format(self.text, self.bbox)

    def __repr__(self):
        return self.__str__()


def crawl(start_link, count=0, t=int(time.time() * 1000)):
    print(count)
    if count == target_count:
        return
    html = get_html(start_link, 5)
    document = BeautifulSoup(html.text, "html.parser")
    try:
        title = document.select_one('h1.title').find(text=True, recursive=False).strip()
        authors = document.select('div.authors > a')
        authors = list(map(lambda a: a.text, authors))
        abstract = document.select_one('blockquote.abstract').findAll(text=True, recursive=False)
        abstract = ''.join(abstract).strip()
        paper_id = document.select_one('div.extra-services > div.full-text > ul > li > a.download-pdf').get('href')
        paper_id = re.sub(r'^/pdf/', '', paper_id)
        if Path('{}/{}.png'.format(str(redaction_dir), paper_id)).exists() or Path('./train_data/redaction_nn/{}.png'.format(paper_id)).exists():
            print('duplicate file')
            raise Exception
        response = requests.get('https://arxiv.org/pdf/{}'.format(paper_id))
        save_pdf(response, paper_id, title, authors, abstract, t)
        count += 1
    except AttributeError:
        pass
    try:
        next_link = document.select_one('span.arrow > a.next-url').get('href')
        next_link = get_html('https://arxiv.org/{}'.format(next_link), 5).url
        time.sleep(randint(10, 15))
        crawl(next_link, count, t)
    except AttributeError:
        print(document)
        return


def save_row(file_name, pix, t, masks):
    try:
        df = pd.read_csv('{}/metadata_{}.csv'.format(redaction_label_dir, t), index_col=0)
    except FileNotFoundError:
        df = pd.DataFrame(columns=['file', 'width', 'height', 'title', 'authors', 'abstract', 'redaction'])
    data = ['{}.png'.format(file_name), pix.width, pix.height, json.dumps(masks[0]), json.dumps(masks[1]), json.dumps(masks[2]), json.dumps(masks[3])]
    df = pd.concat([df, pd.DataFrame([data], columns=df.columns)])
    df.reset_index(drop=True, inplace=True)
    df.to_csv('{}/metadata_{}.csv'.format(redaction_label_dir, t))


def save_pdf(response, key, title, authors, abstract, t):
    with open('{}/paper.pdf'.format(str(temp_dir)), 'wb') as f:
        f.write(response.content)
        doc = fitz.open('{}/paper.pdf'.format(str(temp_dir)))
        pix = doc[0].getPixmap(alpha=False)
        masks = mask_pdf(title, authors, abstract, pix)
        if masks is None:
            return
        file_name = re.sub(r'\.', '_', key)
        # save_thumbnail(file_name, pix)
        save_redaction(file_name, pix, masks)
        save_row(file_name, pix, t, masks)


def save_thumbnail(key, pix):
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples).convert('L')
    img.save('{}/{}.png'.format(str(covers_dir), key), 'PNG')


def save_mask(key, pix, paper_size, masks):
    mask = np.zeros((pix.height, pix.width, 3), np.uint8)
    mask[pix.height - int(masks[0][3]):pix.height - int(masks[0][1]), int(masks[0][0]):int(masks[0][2])] = [255, 0, 0]
    for author_mask in masks[1]:
        mask[pix.height - int(author_mask[3]):pix.height - int(author_mask[1]), int(author_mask[0]):int(author_mask[2])] = [0, 255, 0]
    mask[pix.height - int(masks[2][3]):pix.height - int(masks[2][1]), int(masks[2][0]):int(masks[2][2])] = [0, 0, 255]
    mask = Image.fromarray(mask)
    mask.save('{}/{}/{}.png'.format(str(masks_dir), paper_size, key), 'PNG')


def save_redaction(key, pix, masks):
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples).convert('L')
    img = np.array(img)
    for bbox in masks[3]:
        img[pix.height - int(bbox[3]):pix.height - int(bbox[1]), int(bbox[0]):int(bbox[2])] = 0
    mask = Image.fromarray(img)
    mask.save('{}/{}.png'.format(str(redaction_dir), key), 'PNG')


def mask_pdf(title, authors, abstract, pix):
    raw_boxes, text_boxes, text_lines = parse_layout()
    if len(raw_boxes) * len(text_boxes) * len(text_lines) == 0:
        return None
    title_boxes = list(filter(lambda b: b.bbox[1] > pix.height / 2, raw_boxes))
    if len(title_boxes) == 0:
        return None
    title_box = sorted(title_boxes, key=lambda b: measure_distance(b.text[:min(len(b.text), len(title))], title))[0]
    author_boxes = list(filter(lambda b: b is not title_box, text_lines))
    author_boxes = list(filter(lambda b: b.bbox[1] > pix.height / 2, author_boxes))
    authors_boxes = list(map(lambda a: sorted(author_boxes, key=lambda b: measure_distance(b.text, a))[0], authors))
    abstract_box = sorted(text_boxes, key=lambda b: measure_distance_for_abstracts(b.text, abstract))[0]
    print(normalize_text(title_box.text), list(map(lambda a: normalize_text(a.text), authors_boxes)), normalize_text(abstract_box.text))
    return title_box.bbox, list(map(lambda b: b.bbox, authors_boxes)), abstract_box.bbox, list(map(lambda b: b.bbox, raw_boxes))


def measure_distance(text1, text2):
    return jellyfish.levenshtein_distance(normalize_text(text1), normalize_text(text2))


def measure_distance_for_abstracts(text1, text2):
    text1 = re.sub("Abstract[:-]?", "", text1, count=1)
    text1 = re.sub("ABSTRACT[:-]?", "", text1, count=1)
    text1 = text1[:min(300, len(text1))]
    text2 = text2[:min(300, len(text2))]
    return measure_distance(text1, text2)


def normalize_text(text):
    text = text.strip()
    text = text.lower()
    text = re.sub(r'\s\d+\s', '', text)
    text = re.sub(r'\s?-\s?', '', text)
    text = re.sub(u'\xa0', ' ', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def parse_layout():
    fp = open('{}/paper.pdf'.format(str(temp_dir)), 'rb')
    manager = PDFResourceManager()
    params = LAParams()
    device = PDFPageAggregator(manager, laparams=params)
    interpreter = PDFPageInterpreter(manager, device)
    pages = PDFPage.get_pages(fp)

    for idx, page in enumerate(pages):
        if idx > 0:
            break
        interpreter.process_page(page)
        layout = device.get_result()
        layout = list(filter(lambda l: isinstance(l, LTTextBox), layout))
        raw_boxes = list(map(lambda l: TextArea(l.get_text(), l.bbox), layout))
        text_boxes = []
        text_lines = []
        temp_bbox = None
        temp_text = None
        for l in layout:
            lines = list(filter(lambda b: isinstance(b, LTTextLine), l))
            for line in lines:
                tokens = re.split(r',|\d|‡|†|\*', line.get_text())
                tokens = list(map(lambda t: TextArea(re.sub(r'^\s*and', '', t).strip(), line.bbox), tokens))
                text_lines.extend(tokens)
            if temp_bbox is None and temp_text is None:
                temp_bbox = list(l.bbox)
                temp_text = l.get_text()
            elif l.bbox[0] != temp_bbox[0]:
                text_box = TextArea()
                text_box.bbox = temp_bbox
                text_box.text = temp_text
                text_boxes.append(text_box)
                temp_bbox = list(l.bbox)
                temp_text = l.get_text()
            else:
                temp_bbox[0] = l.bbox[0]
                temp_bbox[1] = min(temp_bbox[1], l.bbox[1])
                temp_bbox[2] = max(temp_bbox[2], l.bbox[2])
                temp_bbox[3] = max(temp_bbox[3], l.bbox[3])
                temp_text = temp_text + l.get_text()

        return raw_boxes, text_boxes, text_lines


if __name__ == '__main__':
    target_count = 20000


    def start_crawling():
        print('starting crawling..')
        try:
            crawl('https://arxiv.org/abs/{}{}.{}?context=cs'.format(str(randint(15, 19)), str(randint(1, 12)).zfill(2), str(randint(0, 10) * 1000).zfill(5)))
        except Exception:
            print('exception caught')
            return


    schedule.every(5).minutes.do(start_crawling)

    while True:
        schedule.run_pending()
