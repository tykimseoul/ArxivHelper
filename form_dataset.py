import shutil
from pathlib import Path
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import re
from pdfminer.layout import LAParams, LTTextBox, LTTextLine
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
import jellyfish
import fitz
from multiprocessing import cpu_count
from multiprocessing.dummy import Pool as ThreadPool

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('max_colwidth', 100)


class TextArea:
    def __init__(self, text=None, bbox=None):
        self.text = text
        self.bbox = bbox

    def __str__(self):
        return 'TextArea({}, {})'.format(self.text, self.bbox)

    def __repr__(self):
        return self.__str__()


def dejson_dataframe(df):
    print('dejsoning')

    def unwrap_columns(df, key):
        parsed = df[key].map(str).apply(json.loads).apply(pd.Series)
        df[list(map(lambda c: '{}_{}'.format(key, c), [0, 1, 2, 3]))] = parsed
        df['{}_area'.format(key)] = (df['{}_3'.format(key)] - df['{}_1'.format(key)]) * (df['{}_2'.format(key)] - df['{}_0'.format(key)])
        df['{}_0'.format(key)] = df['{}_0'.format(key)] / df['width']
        df['{}_1'.format(key)] = df['{}_1'.format(key)] / df['height']
        df['{}_2'.format(key)] = df['{}_2'.format(key)] / df['width']
        df['{}_3'.format(key)] = df['{}_3'.format(key)] / df['height']
        return df

    df.drop_duplicates(subset=['file'], keep='last', inplace=True)
    redaction = df['redaction_bbox'].map(str).apply(json.loads).tolist()
    redaction = list(map(lambda r: list(map(lambda a: (a[3] - a[1]) * (a[2] - a[0]), r)), redaction))
    redaction = list(map(sum, redaction))
    df['redaction_area'] = redaction
    df = unwrap_columns(df, 'title_bbox')
    df = unwrap_columns(df, 'abstract_bbox')
    # df['redaction_area'].hist(bins=1000)
    # plt.show()
    print(df.head(10))
    return df


def filter_dataframe(df):
    print('filtering')
    df.drop(['redaction_bbox', 'title_bbox', 'authors_bbox', 'abstract_bbox'], axis=1, inplace=True)
    df.drop(df[df['width'] != 612].index, inplace=True)
    df.drop(df[df['height'] != 792].index, inplace=True)
    df.drop(df[df['abstract_bbox_area'] > 250000].index, inplace=True)
    df.drop(df[df['abstract_bbox_area'] < 10000].index, inplace=True)
    df.drop(df[df['title_bbox_area'] > 50000].index, inplace=True)
    df.drop(df[df['title_bbox_area'] < 1000].index, inplace=True)
    df.drop(df[df['redaction_area'] < 40000].index, inplace=True)
    df.drop(df[df['redaction_area'] > 350000].index, inplace=True)
    df.drop(df[df['abstract_bbox_3'] > df['title_bbox_1']].index, inplace=True)
    # plt.figure()
    # df['title_bbox_area'].hist(bins=100)
    # plt.show()
    # plt.figure()
    # df['abstract_bbox_area'].hist(bins=100)
    # plt.show()
    df.reset_index(drop=True, inplace=True)
    return df


train_dir = Path("./train_data/train_unet")
shutil.rmtree(train_dir)
train_dir.mkdir(parents=True, exist_ok=True)
area_dir = Path('./train_data/area')
area_dir.mkdir(parents=True, exist_ok=True)
mask_dir = Path('./train_data/masks_unet')
shutil.rmtree(mask_dir)
mask_dir.mkdir(parents=True, exist_ok=True)
covers_dir = Path('./tmp/thumbnail')
shutil.rmtree(covers_dir)
covers_dir.mkdir(parents=True, exist_ok=True)
pdf_dir = Path("./tmp/pdf")
pdf_dir.mkdir(parents=True, exist_ok=True)


def copy_thumbnail(file):
    try:
        shutil.copy('{}/{}.png'.format(str(covers_dir), file), train_dir)
    except FileNotFoundError:
        pass


def make_mask(key, width, height, title_0, title_1, title_2, title_3, abstract_0, abstract_1, abstract_2, abstract_3):
    img = np.zeros((height, width, 3), np.uint8)
    img[height - int(title_3 * height): height - int(title_1 * height), int(title_0 * width):int(title_2 * width), :] = [255, 0, 0]
    img[height - int(abstract_3 * height): height - int(abstract_1 * height), int(abstract_0 * width):int(abstract_2 * width), :] = [0, 0, 255]
    img = Image.fromarray(img)
    img.save('{}/{}.png'.format(mask_dir, key))


def parse(samples_df, pdfs):
    rows = []
    for pdf in pdfs:
        print(pdf)
        try:
            doc = fitz.open('{}/{}'.format(str(pdf_dir), pdf), filetype="pdf")
            pix = doc[0].getPixmap(alpha=False)
            id = re.sub(r'\.pdf$', '', pdf)
            save_thumbnail(id, pix)
            fp = open('{}/{}'.format(str(pdf_dir), pdf), 'rb')
            row = samples_df[samples_df['id'] == id].iloc[0]
            masks = mask_pdf(fp, row['title'], re.split(r',\s+', row['authors']), row['abstract'], pix)
            if masks is None:
                continue
            parsed_row = parse_row(row['id'], pix, masks)
            rows.append(parsed_row)
        except RuntimeError:
            continue
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return pd.DataFrame(columns=['file', 'width', 'height', 'title_bbox', 'authors_bbox', 'abstract_bbox', 'redaction_bbox'])
    df.columns = ['file', 'width', 'height', 'title_bbox', 'authors_bbox', 'abstract_bbox', 'redaction_bbox']
    return df


def save_thumbnail(key, pix):
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples).convert('L')
    img.save('{}/{}.png'.format(covers_dir, key), 'PNG')


def parse_row(file_name, pix, masks):
    return [file_name, pix.width, pix.height, json.dumps(masks[0]), json.dumps(masks[1]), json.dumps(masks[2]), json.dumps(masks[3])]


def mask_pdf(fp, title, authors, abstract, pix):
    raw_boxes, text_boxes, text_lines = parse_layout(fp)
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
    # print(normalize_text(title_box.text), list(map(lambda a: normalize_text(a.text), authors_boxes)), normalize_text(abstract_box.text))
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


def parse_layout(fp):
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


def form_dataset(pdfs):
    samples_df = pd.read_csv('./sampled_metadata.csv')
    df = parse(samples_df, pdfs)

    df = dejson_dataframe(df)
    df = filter_dataframe(df)
    df.apply(lambda r: copy_thumbnail(r['file']), axis=1)
    df.apply(lambda r: make_mask(r['file'], r['width'], r['height'], r['title_bbox_0'], r['title_bbox_1'], r['title_bbox_2'], r['title_bbox_3'], r['abstract_bbox_0'], r['abstract_bbox_1'], r['abstract_bbox_2'], r['abstract_bbox_3']), axis=1)
    return df


processes = cpu_count()
pdfs = sorted(os.listdir(pdf_dir))[:20]
pdf_chunk_size = int(len(pdfs) / processes)
pdf_chunks = [pdfs[i:i + pdf_chunk_size] for i in range(0, len(pdfs), pdf_chunk_size)]
print(list(map(len, pdf_chunks)))
assert sum(list(map(len, pdf_chunks))) == len(pdfs)


def apply_to_df(pdf_chunk):
    pdf_chunk = form_dataset(pdf_chunk)
    return pdf_chunk


# Process dataframes
with ThreadPool(processes) as p:
    result = p.map(apply_to_df, pdf_chunks)

# Concat all chunks
df_reconstructed = pd.concat(result)
df_reconstructed.reset_index(drop=True, inplace=True)
print('done', len(df_reconstructed))
df_reconstructed.to_csv('./train_data/filtered_metadata.csv')
