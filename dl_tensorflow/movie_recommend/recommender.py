# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import pickle
import numpy as np
from collections import Counter
from network import load_params


# 读取数据
title_count, title_set, genres2int, features, targets_values, ratings, users, movies, data, movies_orig, users_orig = pickle.load(
    open('./data/ml-1m/preprocess.p', mode='rb'))

#电影名长度
sentences_size = title_count # = 15
#电影ID转下标的字典，数据集中电影ID跟下标不一致，比如第5行的数据电影ID不一定是5
movieid2idx = {val[0]:i for i, val in enumerate(movies.values)}

load_dir = load_params()

def get_param_tensors(loaded_graph):
    targets = loaded_graph.get_tensor_by_name("targets:0")
    dropout_keep_prob = loaded_graph.get_tensor_by_name("dropout_keep_prob:0")
    lr = loaded_graph.get_tensor_by_name("LearningRate:0")
    #两种不同计算预测评分的方案使用不同的name获取tensor inference
    inference = loaded_graph.get_tensor_by_name("inference/ExpandDims:0")#
    user_combine_layer_flat = loaded_graph.get_tensor_by_name("user_fc/Reshape:0")
    return targets, lr, dropout_keep_prob, inference



def get_user_tensors(loaded_graph):
    # tf.get_default_graph().as_graph_def().node    All tensors 
    uid = loaded_graph.get_tensor_by_name("uid:0")
    user_gender = loaded_graph.get_tensor_by_name("user_gender:0")
    user_age = loaded_graph.get_tensor_by_name("user_age:0")
    user_job = loaded_graph.get_tensor_by_name("user_job:0")

    user_combine_layer_flat = loaded_graph.get_tensor_by_name("user_fc/Reshape:0")
    return uid, user_gender, user_age, user_job, user_combine_layer_flat


def get_movie_tensors(loaded_graph):
    # tf.get_default_graph().as_graph_def().node    All tensors 
    movie_id = loaded_graph.get_tensor_by_name("movie_id:0")
    movie_categories = loaded_graph.get_tensor_by_name("movie_categories:0")
    movie_titles = loaded_graph.get_tensor_by_name("movie_titles:0")
    movie_combine_layer_flat = loaded_graph.get_tensor_by_name("movie_fc/Reshape:0")
    return movie_id, movie_categories, movie_titles, movie_combine_layer_flat


def rating_movie(user_id_val, movie_id_val):
    """
    指定用户和电影进行评分
    return [array([[3.6275628]], dtype=float32)]
    """
    loaded_graph = tf.Graph()  #
    with tf.Session(graph=loaded_graph) as sess:  #
        # Load saved model
        loader = tf.train.import_meta_graph(load_dir + '.meta')
        loader.restore(sess, load_dir)
        targets, lr, dropout_keep_prob, inference = get_param_tensors(loaded_graph) 
        uid, user_gender, user_age, user_job, _ = get_user_tensors(loaded_graph)  
        movie_id, movie_categories, movie_titles, _ = get_movie_tensors(loaded_graph) 

        categories = np.zeros([1, 18])
        categories[0] = movies.values[movieid2idx[movie_id_val]][2]

        titles = np.zeros([1, sentences_size])
        titles[0] = movies.values[movieid2idx[movie_id_val]][1]

        feed = {
            uid: np.reshape(users.values[user_id_val-1][0], [1, 1]),
            user_gender: np.reshape(users.values[user_id_val-1][1], [1, 1]),
            user_age: np.reshape(users.values[user_id_val-1][2], [1, 1]),
            user_job: np.reshape(users.values[user_id_val-1][3], [1, 1]),
            movie_id: np.reshape(movies.values[movieid2idx[movie_id_val]][0], [1, 1]),
            movie_categories: categories,  # x.take(6,1)
            movie_titles: titles,  # x.take(5,1)
            dropout_keep_prob: 1}
        # Get Prediction
        inference_val = sess.run([inference], feed)

        return (inference_val)


def movie_feature_matrics():
    """
    生成Movie特征矩阵
    """
    loaded_graph = tf.Graph()  #
    movie_matrics = []
    with tf.Session(graph=loaded_graph) as sess:
        # Load saved model
        loader = tf.train.import_meta_graph(load_dir + '.meta')
        loader.restore(sess, load_dir)
        # Get Tensors from loaded model
        targets, lr, dropout_keep_prob, inference = get_param_tensors(loaded_graph) 
        movie_id, movie_categories, movie_titles, movie_combine_layer_flat = get_movie_tensors(loaded_graph) 
        for item in movies.values:
            categories = np.zeros([1, 18])
            categories[0] = item.take(2)

            titles = np.zeros([1, sentences_size])
            titles[0] = item.take(1)

            feed = {
                movie_id: np.reshape(item.take(0), [1, 1]),
                movie_categories: categories,  # x.take(6,1)
                movie_titles: titles,  # x.take(5,1)
                dropout_keep_prob: 1}
            movie_combine_layer_flat_val = sess.run([movie_combine_layer_flat], feed)  
            movie_matrics.append(movie_combine_layer_flat_val)    
    # 保存
    pickle.dump((np.array(movie_matrics).reshape(-1, 200)), open('./data/ml-1m/movie_matrics.p', 'wb'))


def user_feature_matrics():
    """
    生成User特征矩阵
    """
    loaded_graph = tf.Graph()  #
    users_matrics = []
    with tf.Session(graph=loaded_graph) as sess:  #
        # Load saved model
        loader = tf.train.import_meta_graph(load_dir + '.meta')
        loader.restore(sess, load_dir)

        targets, lr, dropout_keep_prob, inference = get_param_tensors(loaded_graph)
        uid, user_gender, user_age, user_job,user_combine_layer_flat = get_user_tensors(loaded_graph)

        for item in users.values:
            feed = {
                uid: np.reshape(item.take(0), [1, 1]),
                user_gender: np.reshape(item.take(1), [1, 1]),
                user_age: np.reshape(item.take(2), [1, 1]),
                user_job: np.reshape(item.take(3), [1, 1]),
                dropout_keep_prob: 1}

            user_combine_layer_flat_val = sess.run([user_combine_layer_flat], feed)
            users_matrics.append(user_combine_layer_flat_val)
    # 保存
    pickle.dump((np.array(users_matrics).reshape(-1, 200)), open('./data/ml-1m/users_matrics.p', 'wb'))

    # users_matrics = pickle.load(open('users_matrics.p', mode='rb'))


def recommend_sametype_movie(movie_id, top_k = 10,isprint=True):
    """
    推荐同类型的电影
    """
    # 读取保存的电影特征
    movie_matrics = pickle.load(open('./data/ml-1m/movie_matrics.p', mode='rb'))
    loaded_graph = tf.Graph()
    with tf.Session(graph=loaded_graph) as sess:
         # Load saved model
        loader = tf.train.import_meta_graph(load_dir + '.meta')
        loader.restore(sess, load_dir)
        # 特征数据归一化
        norm_movie_matrics = tf.sqrt(tf.reduce_sum(tf.square(movie_matrics),1,keepdims=True))      #求模
        normalized_movie_matrics = movie_matrics / norm_movie_matrics

        #推荐同类型的电影  计算余弦相似度
        input_embeddings = (movie_matrics[movieid2idx[movie_id]]).reshape([1, 200])                  # 输入的movie特征值
        probs_similarity = tf.matmul(input_embeddings, tf.transpose(normalized_movie_matrics))       # 内积
        # sim = (probs_similarity.eval())
        sim = sess.run(probs_similarity)

        p = sim[0]            
        p[np.argsort(p)[:-top_k]] = 0           # 除了top_k个保留值，其他的都置为0
        score = p / np.sum(p)
        # print([x for x in score if x > 0])
        results = set()
        while len(results) != top_k:
            choice = np.random.choice(len(score), 1, p=score)[0]
            results.add(choice)

        if isprint:
        print("您看的电影是：{}".format(movies_orig[movieid2idx[movie_id]]))
        print("以下是给您的推荐：")
            for val in (results):
                print(val, "\t", movies_orig[val])

        return results




if __name__ == '__main__':
    
    recommend_sametype_movie(1401,top_k =5)
