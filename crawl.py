import requests
import re
import fitz
from bs4 import BeautifulSoup
from pathlib import Path
import time
import pandas as pd
import json
from multiprocessing.dummy import Pool as ThreadPool
import os

temp_dir = Path("./tmp")
temp_dir.mkdir(parents=True, exist_ok=True)
covers_dir = Path("./tmp/temp")
covers_dir.mkdir(parents=True, exist_ok=True)
pdf_dir = Path("./tmp/pdf")
pdf_dir.mkdir(parents=True, exist_ok=True)
skip_dir = Path("./tmp/skip")
skip_dir.mkdir(parents=True, exist_ok=True)
masks_dir = Path("./tmp/masks")
masks_dir.mkdir(parents=True, exist_ok=True)
redaction_dir = Path("./tmp/redaction_nn")
redaction_dir.mkdir(parents=True, exist_ok=True)
redaction_label_dir = Path("./tmp/redaction_label_nn")
redaction_label_dir.mkdir(parents=True, exist_ok=True)


def get_metadata():
    with open('./arxiv-metadata.json', 'r') as f:
        for line in f:
            yield line


def form_dataframe():
    metadata = get_metadata()
    papers = []
    for paper in metadata:
        paper_dict = json.loads(paper)
        category = paper_dict.get('categories').split()
        try:
            year = int(paper_dict.get('id')[:4])
        except ValueError:
            continue
        if all(list(map(lambda c: re.match(r'cs\..+', c), category))) \
                and 1701 <= year:
            print(year, category)
            paper_dict['id'] = re.sub(r'\.', '_', str(paper_dict['id']))
            papers.append(paper_dict)
    print(len(papers))
    df = pd.DataFrame(papers)
    df.drop(['versions', 'submitter', 'comments', 'license', 'doi', 'update_date', 'authors_parsed', 'report-no', 'journal-ref'], axis=1, inplace=True)
    print(df.columns)
    print(df.head())
    df.to_csv('./metadata.csv')


def crawl(id):
    if Path('{}/{}.pdf'.format(str(pdf_dir), id)).exists() or Path('{}/{}.txt'.format(str(skip_dir), id)).exists():
        return
    time.sleep(1)
    id = re.sub('_', '.', id)

    def skip(msg, id):
        f = open('{}/{}.txt'.format(skip_dir, id), "x")
        f.close()
        print(msg, id)

    file_name = re.sub(r'\.', '_', str(id))
    try:
        response = requests.get('https://export.arxiv.org/pdf/{}'.format(id), timeout=(2, 3))
    except requests.exceptions.Timeout:
        skip('timeout error', file_name)
        return

    # document = BeautifulSoup(response.text, "html.parser")
    # try:
    #     if document.select_one('div#content > h1').text.startswith('PDF unavailable for'):
    #         f = open('{}/{}.txt'.format(skip_dir, re.sub('\.', '_', id)), "x")
    #         f.close()
    #         print('skipping', id)
    #         return
    # except AttributeError:
    #     pass
    with open('{}/paper_{}.pdf'.format(str(temp_dir), file_name), 'wb') as f:
        f.write(response.content)
        try:
            doc = fitz.open('{}/paper_{}.pdf'.format(str(temp_dir), file_name))
            doc.select([0])
            doc.save('{}/{}.pdf'.format(str(pdf_dir), file_name))
            os.remove('{}/paper_{}.pdf'.format(str(temp_dir), file_name))
            print('saving', file_name)
        except:
            skip('runtime error', file_name)
            if Path('{}/paper_{}.pdf'.format(str(temp_dir), file_name)).exists():
                os.remove('{}/paper_{}.pdf'.format(str(temp_dir), file_name))
            return


if __name__ == '__main__':
    # form_dataframe()
    # df = pd.read_csv('./metadata.csv')
    # df = df[df['id'].str.match(r'\d{4}_\d{4,5}')]
    # print(len(df))
    # df = df.sample(n=40000)
    # df.to_csv('./sampled_metadata_{}.csv'.format(int(time.time())))
    df = pd.read_csv('./sampled_metadata_1606382293.csv')

    processes = 8
    chunk_size = int(len(df) / processes)
    chunks = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]


    def apply_to_df(df_chunks):
        df_chunks = df_chunks.apply(lambda r: crawl(r['id']), axis=1)
        return df_chunks


    # Process dataframes
    with ThreadPool(processes) as p:
        result = p.map(apply_to_df, chunks)

    # Concat all chunks
    df_reconstructed = pd.concat(result)
