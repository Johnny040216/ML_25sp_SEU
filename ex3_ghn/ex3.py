import scipy, scipy.io, scipy.optimize
from sklearn import svm
from sklearn import model_selection
import numpy as np
import matplotlib.pyplot as plt
# 使matplotlib绘图支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
def plot(data):
    positives = data[data[:, 2] == 1]
    negatives = data[data[:, 2] == 0]
    # 正样本用+号绘制
    plt.plot(positives[:, 0], positives[:, 1], 'b+')
    # 负样本用o号绘制
    plt.plot(negatives[:, 0], negatives[:, 1], 'yo')

def visualize_boundary(X, trained_svm):
    kernel = trained_svm.get_params()['kernel']
    # 线性核函数
    if kernel == 'linear':
        w = trained_svm.coef_[0]
        i = trained_svm.intercept_
        xp = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
        a = -w[0] / w[1]
        b = i[0] / w[1]
        yp = a * xp - b
        plt.plot(xp, yp, 'b-')
    # 高斯核函数
    elif kernel == 'rbf':
        x1plot = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
        x2plot = np.linspace(min(X[:, 1]), max(X[:, 1]), 100)
        X1, X2 = np.meshgrid(x1plot, x2plot)
        vals = np.zeros(np.shape(X1))
        for i in range(0, np.shape(X1)[1]):
            this_X = np.c_[X1[:, i], X2[:, i]]
            vals[:, i] = trained_svm.predict(this_X)
        plt.contour(X1, X2, vals, colors='blue')
# 加载数据集1
mat = scipy.io.loadmat("dataset_1.mat")
X, y = mat['X'], mat['y']
# 绘制数据集1
plt.title('数据集1分布')
plot(np.c_[X, y])
plt.show(block=True)

linear_svm = svm.SVC(C=1, kernel='linear')
linear_svm.fit(X, y.ravel())
# 绘制C=1的SVM决策边界
plt.title('C=1的SVM决策边界')
plot(np.c_[X, y])
visualize_boundary(X, linear_svm)
plt.show(block=True)

linear_svm = svm.SVC(C=100, kernel='linear')
linear_svm.fit(X, y.ravel())
# 绘制C=100的SVM决策边界
plt.title('C=100的SVM决策边界')
plot(np.c_[X, y])
visualize_boundary(X, linear_svm)
plt.show(block=True)

def gaussian_kernel(x1, x2, sigma):
    return np.exp(-np.sum((x1 - x2)**2 / float(2 * sigma ** 2)))
# 计算高斯核函数
x1 = np.array([1, 2, 1])
x2 = np.array([0, 4, -1])
sigma = 2
print("样本x1和x2之间的相似度: %f" % gaussian_kernel(x1, x2, sigma))


# 加载数据集1
mat = scipy.io.loadmat("dataset_2.mat")
X, y = mat['X'], mat['y']
# 绘制数据集2
plt.title('数据集2分布')
plot(np.c_[X, y])
plt.show(block=True)

sigma = 0.01
rbf_svm = svm.SVC(C=1, kernel='rbf', gamma=1.0 / sigma)  # gamma实际上是sigma的倒数
rbf_svm.fit(X, y.ravel())

plt.title("高斯核函数SVM决策边界")
plot(np.c_[X, y])
visualize_boundary(X, rbf_svm)
plt.show(block=True)

# 加载数据集3获得训练集和验证集
mat = scipy.io.loadmat("dataset_3.mat")
X, y = mat['X'], mat['y'] # 训练集
X_val, y_val = mat['Xval'], mat['yval'] # 验证集

# 绘制数据集3
plt.title('数据集3分布')
plot(np.c_[X, y])
plt.show(block=True)

# 绘制验证集
plt.title('验证集分布')
plot(np.c_[X_val, y_val])
plt.show(block=True)


def params_search(X, y, X_val, y_val):
    np.c_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    sigma_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    raveled_y = y.ravel()
    m_val = np.shape(X_val)[0]

    rbf_svm = svm.SVC(kernel='rbf')
    best = {'error': 999, 'C': 0.0, 'sigma': 0.0}

    for C in np.c_values:
        for sigma in sigma_values:
            # 根据不同参数训练SVM
            rbf_svm.set_params(C=C)
            rbf_svm.set_params(gamma=1.0 / sigma)
            rbf_svm.fit(X, raveled_y)

            # 预测并计算误差
            predictions = []
            for i in range(0, m_val):
                prediction_result = rbf_svm.predict(X_val[i].reshape(-1, 2))
                predictions.append(prediction_result[0])

            predictions = np.array(predictions).reshape(m_val, 1)
            error = (predictions != y_val.reshape(m_val, 1)).mean()

            # 记录误差最小的一组参数
            if error < best['error']:
                best['error'] = error
                best['C'] = C
                best['sigma'] = sigma
    best['gamma'] = 1.0 / best['sigma']
    return best
# 训练高斯核函数SVM并搜索使用最优模型参数
rbf_svm = svm.SVC(kernel='rbf')
# your code here (通过rbf_svm.set_params可设定模型的C和gamma值)

C=(params_search(X, y, X_val, y_val))['C']
gamma=(params_search(X, y, X_val, y_val))['gamma']
rbf_svm.set_params(C=C, gamma=gamma)
rbf_svm.fit(X, y.ravel())

# 绘制决策边界
plt.title('参数搜索后的决策边界')
plot(np.c_[X, y])
visualize_boundary(X, rbf_svm)
plt.show(block=True)