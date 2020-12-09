import shutil
from pathlib import Path
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from pdfminer.psparser import PSSyntaxError
from PIL import Image
import re
from pdfminer.layout import LAParams, LTTextBox, LTTextLine
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
import jellyfish
import fitz
from multiprocessing import Pool, current_process

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('max_colwidth', 100)

train_dir = Path("./train_data/train_unet")
shutil.rmtree(train_dir)
train_dir.mkdir(parents=True, exist_ok=True)
mask_dir = Path('./train_data/masks_unet')
shutil.rmtree(mask_dir)
mask_dir.mkdir(parents=True, exist_ok=True)
covers_dir = Path('./tmp/thumbnail')
covers_dir.mkdir(parents=True, exist_ok=True)
temp_mask_dir = Path('./tmp/masks')
temp_mask_dir.mkdir(parents=True, exist_ok=True)
pdf_dir = Path("./tmp/pdf")
pdf_dir.mkdir(parents=True, exist_ok=True)
temp_redaction_dir = Path("./tmp/redaction")
temp_redaction_dir.mkdir(parents=True, exist_ok=True)
redaction_dir = Path("./train_data/redaction")
redaction_dir.mkdir(parents=True, exist_ok=True)


class TextArea:
    def __init__(self, text=None, bbox=None):
        self.text = text
        self.bbox = bbox

    def __str__(self):
        return 'TextArea({}, {})'.format(self.text, self.bbox)

    def __repr__(self):
        return self.__str__()


def dejson_dataframe(df):
    print(current_process().name, len(df), 'dejsoning')

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
    df.drop(['redaction_bbox', 'title_bbox', 'authors_bbox', 'abstract_bbox'], axis=1, inplace=True)
    return df


def filter_dataframe(df):
    print(current_process().name, len(df), 'filtering')
    filtered = df
    filtered.reset_index(drop=True, inplace=True)
    filtered = filtered.drop(filtered[filtered['width'] != 612].index)
    filtered = filtered.drop(filtered[filtered['height'] != 792].index)
    filtered = filtered.drop(filtered[filtered['abstract_bbox_area'] > 250000].index)
    filtered = filtered.drop(filtered[filtered['abstract_bbox_area'] < 20000].index)
    filtered = filtered.drop(filtered[filtered['title_bbox_area'] > 50000].index)
    filtered = filtered.drop(filtered[filtered['title_bbox_area'] < 1000].index)
    filtered = filtered.drop(filtered[filtered['redaction_area'] < 40000].index)
    filtered = filtered.drop(filtered[filtered['redaction_area'] > 350000].index)
    filtered = filtered.drop(filtered[filtered['abstract_bbox_3'] > filtered['title_bbox_1']].index)
    return filtered


def copy_image(file, dir, dest):
    try:
        shutil.copy('{}/{}.png'.format(dir, file), dest)
    except FileNotFoundError:
        pass


def make_mask(key, width, height, title_0, title_1, title_2, title_3, abstract_0, abstract_1, abstract_2, abstract_3):
    if not Path('{}/{}.png'.format(temp_mask_dir, key)).exists():
        print('making mask', key)
        img = np.zeros((height, width, 3), np.uint8)
        img[height - int(title_3 * height): height - int(title_1 * height), int(title_0 * width):int(title_2 * width), :] = [255, 0, 0]
        img[height - int(abstract_3 * height): height - int(abstract_1 * height), int(abstract_0 * width):int(abstract_2 * width), :] = [0, 0, 255]
        img = Image.fromarray(img)
        img.save('{}/{}.png'.format(temp_mask_dir, key))


def save_redaction(key, pix, masks):
    if not Path('{}/{}.png'.format(temp_redaction_dir, key)).exists():
        print('redacting', key)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples).convert('L')
        img = np.array(img)
        for bbox in masks[3]:
            img[pix.height - int(bbox[3]):pix.height - int(bbox[1]), int(bbox[0]):int(bbox[2])] = 0
        mask = Image.fromarray(img)
        mask.save('{}/{}.png'.format(str(temp_redaction_dir), key), 'PNG')


def parse(samples_df, pdfs):
    count = 0
    for pdf in pdfs:
        print(current_process().name, count, pdf)
        try:
            id = re.sub(r'\.pdf$', '', pdf)
            if Path('{}/{}.png'.format(covers_dir, id)).exists() and Path('{}/{}.png'.format(temp_redaction_dir, id)).exists() and id in pd.read_csv('./tmp/metadata_{}.csv'.format(current_process().name), index_col=0)['file'].tolist():
                print('skipping', id)
                count += 1
                continue
            doc = fitz.open('{}/{}'.format(str(pdf_dir), pdf), filetype="pdf")
            if doc.isDirty:
                print(current_process().name, pdf, 'dirty')
                count += 1
                continue
            pix = doc[0].getPixmap(alpha=False)
            save_thumbnail(id, pix)
            fp = open('{}/{}'.format(str(pdf_dir), pdf), 'rb')
            row = samples_df[samples_df['id'] == id].iloc[0]
            masks = mask_pdf(fp, row['title'], re.split(r',\s+', row['authors']), row['abstract'], pix)
            if masks is None:
                print('mask none', pdf)
                continue
            save_redaction(id, pix, masks)
            save_row(row['id'], pix, masks, current_process().name)
            count += 1
        except (RuntimeError, ValueError):
            print(current_process().name, pdf, 'runtime')
            continue
        except Exception as e:
            print(current_process().name, 'new error', e, type(e))
            continue
    df = pd.read_csv('./tmp/metadata_{}.csv'.format(current_process().name), index_col=0)
    if len(df) == 0:
        return pd.DataFrame(columns=['file', 'width', 'height', 'title_bbox', 'authors_bbox', 'abstract_bbox', 'redaction_bbox'])
    return df


def save_thumbnail(key, pix):
    if not Path('{}/{}.png'.format(covers_dir, key)).exists():
        print('saving thumbnail', key)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples).convert('L')
        img.save('{}/{}.png'.format(covers_dir, key), 'PNG')


def save_row(file_name, pix, masks, thread):
    try:
        df = pd.read_csv('./tmp/metadata_{}.csv'.format(thread), index_col=0)
    except FileNotFoundError:
        df = pd.DataFrame(columns=['file', 'width', 'height', 'title_bbox', 'authors_bbox', 'abstract_bbox', 'redaction_bbox'])
    if file_name not in df['file'].tolist():
        print('saving row', file_name)
        data = [file_name, pix.width, pix.height, json.dumps(masks[0]), json.dumps(masks[1]), json.dumps(masks[2]), json.dumps(masks[3])]
        df = pd.concat([df, pd.DataFrame([data], columns=df.columns)])
        df.to_csv('./tmp/metadata_{}.csv'.format(thread))


def parse_row(file_name, pix, masks):
    return [file_name, pix.width, pix.height, json.dumps(masks[0]), json.dumps(masks[1]), json.dumps(masks[2]), json.dumps(masks[3])]


def mask_pdf(fp, title, authors, abstract, pix):
    try:
        raw_boxes, text_boxes, text_lines = parse_layout(fp)
    except (TypeError, PSSyntaxError):
        return None
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
    df = parse(metadata, pdfs)
    df = dejson_dataframe(df)
    df.apply(lambda r: make_mask(r['file'], r['width'], r['height'], r['title_bbox_0'], r['title_bbox_1'], r['title_bbox_2'], r['title_bbox_3'], r['abstract_bbox_0'], r['abstract_bbox_1'], r['abstract_bbox_2'], r['abstract_bbox_3']), axis=1)
    return df


processes = 4
pdfs = sorted(os.listdir(pdf_dir))
pdf_chunk_size = int(len(pdfs) / processes)
pdf_chunks = [pdfs[i:i + pdf_chunk_size] for i in range(0, len(pdfs), pdf_chunk_size)]
print(list(map(len, pdf_chunks)))
pdfs = set(map(lambda p: re.sub(r'\.pdf$', '', p), pdfs))


def apply_to_df(pdf_chunk):
    df = form_dataset(pdf_chunk)
    return df


metadata = pd.read_csv('./metadata.csv')
metadata = metadata[metadata['id'].isin(pdfs)]
print(len(metadata))
# Process dataframes
with Pool(processes) as p:
    result = p.map(apply_to_df, pdf_chunks)

# Concat all chunks
df_reconstructed = pd.concat(result)
df_reconstructed.drop_duplicates(subset=['file'], inplace=True)
df_reconstructed.reset_index(drop=True, inplace=True)
print('done', len(df_reconstructed))
df_reconstructed.to_csv('./train_data/dejsoned_metadata.csv')

# faster without multiprocessing
filtered_df = filter_dataframe(df_reconstructed)
filtered_df.to_csv('./train_data/filtered_metadata.csv')
filtered_df.apply(lambda r: copy_image(r['file'], covers_dir, train_dir), axis=1)
filtered_df.apply(lambda r: copy_image(r['file'], temp_mask_dir, mask_dir), axis=1)
filtered_df.apply(lambda r: copy_image(r['file'], temp_redaction_dir, redaction_dir), axis=1)
