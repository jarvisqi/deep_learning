#!/usr/bin/env python3
import pandas as pd
import numpy as np
import requests
import json
import urllib
import os.path

rating_f = 'data\\ml-latest-small\\ratings.csv'
link_f = 'data\\ml-latest-small\\links.csv'
poster_pt = 'data\\posters/'
download_posters = 1000

headers = {'Accept': 'application/json'}
payload = {'api_key': '20047cd838219fb54d1f8fc32c45cda4'}
response = requests.get(
    'http://api.themoviedb.org/3/configuration',
    params=payload,
    headers=headers)
response = json.loads(response.text)

base_url = response['images']['base_url'] + 'w185'


def get_poster(imdb, base_url):
    #query themovie.org API for movie poster path.
    file_path = ''
    imdb_id = 'tt0{0}'.format(imdb)
    movie_url = 'http://api.themoviedb.org/3/movie/{:}/images'.format(imdb_id)
    response = requests.get(movie_url, params=payload, headers=headers)

    try:
        file_path = json.loads(response.text)['posters'][0]['file_path']
    except:
        print('Failed to get url for imdb: {0}'.format(imdb))

    return base_url + file_path


df_id = pd.read_csv(link_f, sep=',')
idx_to_mv = {}
for row in df_id.itertuples():
    idx_to_mv[row[1] - 1] = row[2]

mvs = [0] * len(idx_to_mv.keys())
for i in range(len(mvs)):
    if i in idx_to_mv.keys() and len(str(idx_to_mv[i])) == 6:
        mvs[i] = idx_to_mv[i]
mvs = list(filter(lambda imdb: imdb != 0, mvs))
mvs = mvs[:download_posters]
total_mvs = len(mvs)

URL = [0] * total_mvs
URL_IMDB = {'url': [], 'imdb': []}

i = 0
for m in mvs:
    if (os.path.exists(poster_pt + str(i) + '.jpg')):
        # print('Skip downloading exists jpg: {0}.jpg'.format(poster_pt+str(i)))
        i += 1
        continue
    URL[i] = get_poster(m, base_url)
    if (URL[i] == base_url):
        print('Bad imdb id: {0}'.format(m))
        mvs.remove(m)
        continue
    print('No.{0}: Downloading jpg(imdb {1}) {2}'.format(i, m, URL[i]))
    urllib.request.urlretrieve(URL[i], poster_pt + str(i) + '.jpg')
    URL_IMDB['url'].append(URL[i])
    URL_IMDB['imdb'].append(m)
    i += 1

#df = pd.DataFrame(data=URL_IMDB,index=range(i))
#df.to_csv('imdb_url.csv')

from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image as kimage
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

image = [0] * total_mvs
x = [0] * total_mvs
prediction_result = 'data\\prediction_result.csv'


def save_predict_res(matrix_res):
    pe = dict([(x, []) for x in range(matrix_res.shape[0])])
    for i in range(matrix_res.shape[0]):
        pe[i].extend(matrix_res[i])
    df = pd.DataFrame(data=pe)
    df.to_csv(prediction_result)


def load_result_matrix(file):
    if not os.path.exists(file):
        return None
    m_r = pd.read_csv(file, sep=',')
    f_r = np.zeros((m_r.shape[1] - 1, m_r.shape[0]))
    for i in range(m_r.values.shape[1] - 1):
        f_r[i] = m_r[str(i)].values.tolist()
    return f_r


matrix_res = load_result_matrix(prediction_result)
if matrix_res is None:
    for i in range(total_mvs):
        image[i] = kimage.load_img(
            poster_pt + str(i) + '.jpg', target_size=(224, 224))
        x[i] = kimage.img_to_array(image[i])
        x[i] = np.expand_dims(x[i], axis=0)
        x[i] = preprocess_input(x[i])

    # weights：None代表随机初始化，即不加载预训练权重。'imagenet'代表加载预训练权重
    # include_top：是否保留顶层的3个全连接网络
    model = VGG16(include_top=False, weights='imagenet')
    prediction = [0] * total_mvs
    matrix_res = np.zeros([total_mvs, 25088])
    for i in range(total_mvs):
        # VGG16提取特征
        prediction[i] = model.predict(x[i]).ravel()
        matrix_res[i, :] = prediction[i]
    # 保存特征矩阵
    save_predict_res(matrix_res)

#计算相似度
similarity_deep = matrix_res.dot(matrix_res.T)
norms = np.array([np.sqrt(np.diagonal(similarity_deep))])
similarity_deep = (similarity_deep / (norms * norms.T))

#显示推荐结果
n_display = 5
base_mv_idx = 0
mv = [x for x in np.argsort(similarity_deep[base_mv_idx])[:-n_display - 1:-1]]
for i in range(len(mv)):
	print(str(mv[i])+".jpg")
