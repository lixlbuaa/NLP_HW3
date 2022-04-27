import os
import jieba
import random
import logging
import numpy as np
import pandas as pd
from skimage import measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import precision_score, recall_score, f1_score
logging.basicConfig(level=logging.INFO)

# 断句分词
class Paragraph:
    def __init__(self, txtname='', content='', sentences=[], words=''):
        self.fromtxt = txtname
        self.content = content
        self.sentences = sentences
        self.words = words
        global punctuation
        self.punctuation = punctuation
        global stopwords
        self.stopwords = stopwords

    def sepSentences(self):
        line = ''
        sentences = []
        for w in self.content:
            if w in self.punctuation and line != '\n':
                if line.strip() != '':
                    sentences.append(line.strip())
                    line = ''
            elif w not in self.punctuation:
                line += w
        self.sentences = sentences

    def sepWords(self):
        words = []
        dete_stopwords = 1
        if dete_stopwords:
            for i in range(len(self.sentences)):
                words.extend([x for x in jieba.cut(
                    self.sentences[i]) if x not in self.stopwords])
        else:
            for i in range(len(self.sentences)):
                words.extend([x for x in jieba.cut(self.sentences[i])])
        reswords = ' '.join(words)
        self.words = reswords

    def processData(self):
        self.sepSentences()
        self.sepWords()


def txt_convert_2_excel(file_path, data_path, K=3):
    """
    :param file_path: 小说集存储的路径
    :param data_path: excel路径
    :param K: 随机选取的小说篇数
    :return: 将txt变成excel数据，返回excel路径
    """
    logging.info('Converting txt to excel...')
    files = []
    for x in os.listdir(file_path):
        files.append(x)
    selected_files = random.sample(files, k=3)

    txt = []
    txtname = []
    n = 150  # 每篇选取150段

    for file in selected_files:
        filename = os.path.join(file_path, file)
        with open(filename, 'r', encoding='ANSI') as f:
            full_txt = f.readlines()
            lenth_lines = len(full_txt)
            i = 200
            for j in range(n):
                txt_j = ''
                while(len(txt_j) < 500):
                    txt_j += full_txt[i]
                    i += 1
                txt.append(txt_j)
                txtname.append(file.split('.')[0])
                i += int(lenth_lines / (3 * n))

    dic = {'Content': txt, 'Txtname': txtname}
    df = pd.DataFrame(dic)
    out_path = data_path+'\\data.xlsx'
    df.to_excel(out_path, index=False)
    logging.info('Convert done!')
    return out_path


def read_all_data(path):
    data_list = []
    data_all = pd.read_excel(path)
    for i in range(len(data_all['Content'])):
        d = Paragraph()
        d.content = data_all['Content'][i]
        d.fromtxt = data_all['Txtname'][i]
        data_list.append(d)
    return data_list


def read_punctuation_list(path):
    punctuation = [line.strip()
                   for line in open(path, encoding='UTF-8').readlines()]
    punctuation.extend(['\n', '\u3000', '\u0020', '\u00A0'])
    return punctuation


def read_stopwords_list(path):
    stopwords = [line.strip()
                 for line in open(path, encoding='UTF-8').readlines()]
    return stopwords


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))

def main():
    # path
    file_dir_path = '.\\DatabaseChinese'
    data_dir_path = '.\\DataExcel'
    stopwords_path = '.\\StopWord\\cn_stopwords.txt'
    punctuation_path = '.\\StopWord\\cn_punctuation.txt'

    # read files
    global stopwords
    stopwords = read_stopwords_list(stopwords_path)
    global punctuation
    punctuation = read_punctuation_list(punctuation_path)
    data_list = read_all_data(
        txt_convert_2_excel(file_dir_path, data_dir_path))

    # data process
    corpus = []
    for i in range(len(data_list)):
        data_list[i].processData()
        corpus.append(data_list[i].words)

    # LDA
    logging.info('Training LDA model...')
    cntVector = CountVectorizer(max_features=40)  # 特征词数设置为40个
    cntTf = cntVector.fit_transform(corpus)
    lda = LatentDirichletAllocation(
        n_components=3, learning_offset=50., max_iter=1000, random_state=0)  # 主题3个，迭代1000次
    docres = lda.fit_transform(cntTf)

    # SVM
    logging.info('SVM classify...')
    X = docres
    y = [data_list[i].fromtxt for i in range(len(data_list))]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)   # 训练集150*0.2=30个段落
    svm_model = LinearSVC()  # model = SVC()
    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)

    # analysis
    p = precision_score(y_test, y_pred, average='macro')
    r = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    logging.info('Precision:{:.3f},Recall:{:.3f},F1:{:.3f}'.format(p, r, f1))

    # show test result
    print('Topic real:', '\t', 'Topic predict:', '\n')
    count = 0
    for i in range(len(y_test)):
        if y_pred[i] == y_test[i]:
            count += 1
        print(y_test[i], '\t', y_pred[i], '\n')
    print('分类正确率：',count/len(y_test))
    # show LDA result
    feature_names = cntVector.get_feature_names()
    print_top_words(lda, feature_names, 10)



if __name__ == "__main__":
    main()
