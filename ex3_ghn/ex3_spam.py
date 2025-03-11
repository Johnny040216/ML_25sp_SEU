import scipy, scipy.io, scipy.optimize
from sklearn import model_selection
import numpy as np
import matplotlib.pyplot as plt
import nltk
import csv
import re
from sklearn import svm

def vocaburary_mapping():
    vocab_list = {}
    with open('vocab.txt', 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            vocab_list[row[1]] = int(row[0])
    return vocab_list

def email_preprocess(email):
    # 读取指定邮件文本
    with open(email, 'r') as f:
        email_contents = f.read()
    vocab_list = vocaburary_mapping()
    word_indices = []
    # 邮件文本预处理
    email_contents = email_contents.lower()
    email_contents = re.sub('<[^<>]+>', ' ', email_contents)
    email_contents = re.sub('[0-9]+', 'number', email_contents)
    email_contents = re.sub('(http|https)://[^\s]*', 'httpaddr', email_contents)
    email_contents = re.sub('[^\s]+@[^\s]+', 'emailaddr', email_contents)
    email_contents = re.sub('[$]+', 'dollar', email_contents)
    stemmer = nltk.stem.porter.PorterStemmer()
    tokens = re.split('[ ' + re.escape("@$/#.-:&*+=[]?!(){},'\">_<;%\n") + ']', email_contents)

    for token in tokens:
        token = re.sub('[^a-zA-Z0-9]', '', token)
        token = stemmer.stem(token.strip())
        if len(token) == 0:
            continue
        if token in vocab_list:
            word_indices.append(vocab_list[token])
    # 返回邮件文本词汇与词典中词汇的对应关系，以及得到的预处理文本
    return word_indices, ' '.join(tokens)

word_indices, processed_contents = email_preprocess('emailSample1.txt')
print(word_indices)
print(processed_contents)

def feature_extraction(word_indices):
    features = np.zeros((1899, 1))
    for index in word_indices:
        features[index] = 1
    return features

word_indices, processed_contents = email_preprocess('emailSample1.txt')
features = feature_extraction(word_indices)
print(features)

#加载训练集
mat = scipy.io.loadmat("spamTrain.mat")
X, y = mat['X'], mat['y']
#训练SVM
linear_svm = svm.SVC(C=0.1, kernel='linear')
linear_svm.fit(X, y.ravel())
# 预测并计算训练集正确率
predictions = linear_svm.predict(X)
predictions = predictions.reshape(np.shape(predictions)[0], 1)
print('{}%'.format((predictions == y).mean() * 100.0))
# 加载测试集
mat = scipy.io.loadmat("spamTest.mat")
X_test, y_test = mat['Xtest'], mat['ytest']
# 预测并计算测试集正确率
predictions = linear_svm.predict(X_test)
predictions = predictions.reshape(np.shape(predictions)[0], 1)
print('{}%'.format((predictions == y_test).mean() * 100.0))

vocab_list = vocaburary_mapping()
reversed_vocab_list = dict((v, k) for (k, v) in vocab_list.items())
sorted_indices = np.argsort(linear_svm.coef_, axis=None)
for i in sorted_indices[0:15]:
    print(reversed_vocab_list[i])

def part_4(email_file):
    word_indices, processed_contents = email_preprocess(email_file)
    # 提取特征
    features = feature_extraction(word_indices)
    # 使用训练好的模型进行预测
    prediction = linear_svm.predict(features.T)
    # 输出预测结果
    is_spam = 1 if prediction[0] == 1 else 0
    print(f"邮件 '{email_file}' 的预测结果: {'是垃圾邮件' if is_spam == 1 else '不是垃圾邮件'}")
    return is_spam, features
# 对spamSample2.txt进行预测
is_spam_2, features_2 = part_4('spamSample2.txt')
# 对spamSample1.txt进行预测
is_spam_1, features_1 = part_4('spamSample1.txt')
# 对正常邮件样例进行预测
is_normal_1, features_normal_1 = part_4('emailSample1.txt')
is_normal_2, features_normal_2 = part_4('emailSample2.txt')
# 打印训练集中最有影响力的特征词（垃圾邮件的指标词）
print("\n垃圾邮件中最常见的15个单词：")
vocab_list = vocaburary_mapping()
reversed_vocab_list = dict((v, k) for (k, v) in vocab_list.items())
sorted_indices = np.argsort(linear_svm.coef_, axis=None)[::-1]  # 倒序排列，找出最重要的正向特征
for i in sorted_indices[0:15]:
    print(reversed_vocab_list[i])


