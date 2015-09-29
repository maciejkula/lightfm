import codecs
import os
import subprocess

from lxml import etree

import numpy as np

import requests

import scipy.sparse as sp

from sklearn.feature_extraction import DictVectorizer


def _get_data_path(fname):

    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        fname)


def _get_download_path():

        return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'crossvalidated.7z')


def _download():
    """
    Download the dataset.
    """

    url = 'https://archive.org/download/stackexchange/stats.stackexchange.com.7z'
    req = requests.get(url, stream=True)

    download_path = _get_download_path()
    data_path = _get_data_path('')

    if not os.path.isfile(download_path):
        print('Downloading data...')
        with open(download_path, 'wb') as fd:
            for chunk in req.iter_content():
                fd.write(chunk)

    with open(os.devnull, 'w') as fnull:
        print('Extracting data...')
        try:
            subprocess.check_call(['7za', 'x', download_path],
                                  cwd=data_path, stdout=fnull)
        except subprocess.CalledProcessError:
            print('You must install p7zip to extract the data.')


def _get_raw_data(fname):
    """
    Return the raw lines of the train and test files.
    """

    path = _get_data_path(fname)

    if not os.path.isfile(path):
        _download()

    return codecs.open(path, 'r', encoding='utf-8')


def _process_post_tags(tags_string):

    return [x for x in tags_string.replace('<', ' ').replace('>', ' ').split(' ') if x]


def _read_raw_post_data():

    with _get_raw_data('Posts.xml') as datafile:

        for i, line in enumerate(datafile):
            try:
                datum = dict(etree.fromstring(line).items())

                post_id = datum['Id']
                parent_post_id = datum.get('ParentId', None)
                user_id = datum.get('OwnerUserId', None)

                tags = _process_post_tags(datum.get('Tags', ''))

                if None in (post_id, user_id):
                    continue

            except etree.XMLSyntaxError:
                continue

            yield post_id, user_id, parent_post_id, tags


def read_post_data():
    """
    Construct a user-thread matrix, where a user interacts
    with a thread if they post an answer in it.
    """

    user_mapping = {}
    post_mapping = {}

    question_tags = {}
    uids = []
    pids = []
    data = []

    for (post_id, user_id,
         parent_post_id, tags) in _read_raw_post_data():

        if parent_post_id is None:
            # This is a question

            pid = post_mapping.setdefault(post_id,
                                          len(post_mapping))
            tag_dict = question_tags.setdefault(pid, {})

            for tag in tags:
                tag_dict[tag] = 1
        else:
            # This is an answer
            uid = user_mapping.setdefault(user_id,
                                          len(user_mapping))
            pid = post_mapping.setdefault(parent_post_id,
                                          len(post_mapping))

            uids.append(uid)
            pids.append(pid)
            data.append(1)

    interaction_matrix = sp.coo_matrix((data, (uids, pids)),
                                       shape=(len(user_mapping),
                                              len(post_mapping)),
                                       dtype=np.int32)

    tag_list = [question_tags.get(x, {}) for x in range(len(post_mapping))]

    vectorizer = DictVectorizer(dtype=np.int32)
    tag_matrix = vectorizer.fit_transform(tag_list)
    assert tag_matrix.shape[0] == interaction_matrix.shape[1]

    return interaction_matrix, tag_matrix, vectorizer





import itertools

data = read_post_data()
