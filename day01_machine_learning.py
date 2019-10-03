from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
import jieba
import numpy as np
import pandas as pd


def datasets_demo():
    """
    sklearn数据集使用
    :return:
    """
    # 获取数据集
    iris = load_iris()
    print("鸢尾花数据集：\n", iris)
    print("查看数据集描述：\n", iris["DESCR"])
    print("查看特征值的名字：\n", iris.feature_names)
    print("查看特征值：\n", iris.data, iris.data.shape)

    # 数据集划分  返回值：训练集特征值， 测试集特征值， 训练集目标值， 测试集目标值
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=22)
    print("训练集的特征值：\n", x_train, x_train.shape)
    return None


def dict_demo():
    """
    字典特征抽取: DictVectorizer
    :return:
    """
    data = [{'city': '北京', 'temperature': 100}, {'city': '上海', 'temperature': 60}, {'city': '深圳', 'temperature': 30}]

    # 1. 实例化一个转换器类
    transfer = DictVectorizer(sparse=False)

    # 2. 调用fit_transform()
    data_new = transfer.fit_transform(data)

    print("data_new:\n", data_new)
    print("特征名字:\n", transfer.get_feature_names())

    return None


def count_demo():
    """
    文本特征抽取：CountVecotrizer
    :return:
    """
    data = ["life is short, i like like python", "life is too long, i dislike python"]

    # 1. 实例化一个转换器类
    transfer = CountVectorizer()

    # 2. 调用fit_transform
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new.toarray())
    print("特征名字：\n", transfer.get_feature_names())

    return None


def count_chinese_demo():
    """
    中文文本特征抽取：CountVecotrizer
    :return:
    """
    data = ["我 爱 北京 天安门", "天安门 上 太阳 升"]

    # 1. 实例化一个转换器类
    transfer = CountVectorizer()

    # 2. 调用fit_transform
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new.toarray())
    print("特征名字：\n", transfer.get_feature_names())

    return None


def cut_word(text):
    """
    进行中文分词
    :param text:
    :return:
    """
    return " ".join(jieba.cut(text))


def count_chinese_demo2():
    """
    中文文本特征抽取：CountVecotrizer
    :return:
    """
    data = ["北京市，简称“京”，是中华人民共和国省级行政区、首都、直辖市、国家中心城市、超大城市，全国政治中心、文化中心、国际交往中心、科技创新中心，是世界著名古都和现代化国际城市",
            "是中国共产党中央委员会、中华人民共和国中央人民政府和全国人民代表大会常务委员会的办公所在地。"]

    data_new = []
    for sentence in data:
        data_new.append(cut_word(sentence))

    # 1. 实例化一个转换器类
    transfer = CountVectorizer()

    # 2. 调用fit_transform
    data_final = transfer.fit_transform(data_new)
    print("data_new:\n", data_final.toarray())
    print("特征名字：\n", transfer.get_feature_names())
    return None


def tfidf_demo():
    """
    用TF-IDF的方法进行文本特征抽取
    :return:
    """
    data = ["北京市，简称“京”，是中华人民共和国省级行政区、首都、直辖市、国家中心城市、超大城市，全国政治中心、文化中心、国际交往中心、科技创新中心，是世界著名古都和现代化国际城市",
            "是中国共产党中央委员会、中华人民共和国中央人民政府和全国人民代表大会常务委员会的办公所在地。"]

    data_new = []
    for sentence in data:
        data_new.append(cut_word(sentence))
    data_new = np.array(data_new).reshape(1, -1)

    # 1. 实例化一个转换器类
    transfer = TfidfTransformer()

    # 2. 调用fit_transform
    print(data_new)
    # data_new.reshape(1, -1)
    data_final = transfer.fit_transform(data_new)
    print("data_new:\n", data_final.toarray())
    print("特征名字：\n", transfer.get_feature_names())

    return None


def minmax_demo():
    """
    归一化
    :return:
    """
    # 1. 获取数据
    data = pd.read_csv("dating.txt")
    data = data.iloc[:, :3]

    # 2. 实例化一个转换器
    transfer = MinMaxScaler()

    # 3. 调用fit_transform
    data_new = transfer.fit_transform(data)

    print(data_new)
    return None


def stand_demo():
    """
    标准化
    :return:
    """
    # 1. 获取数据
    data = pd.read_csv("dating.txt")
    data = data.iloc[:, :3]

    # 2. 实例化一个转换器
    transfer = StandardScaler()

    # 3. 调用fit_transform
    data_new = transfer.fit_transform(data)

    print(data_new)
    return None


def variance_demo():
    """
    过滤低方差特征
    :return:
    """
    # 1. 获取数据
    data = pd.read_csv("factor_returns.csv")
    data = data.iloc[:, 1:-2]
    print("data:\n", data)

    # 2. 实例化一个转换器类
    transfer = VarianceThreshold()

    # 3. 调用fit_transform
    data_new = transfer.fit_transform(data)

    # 计算某两个变量之间的相关系数
    r = pearsonr(data["pe_ratio"], data["pb_ratio"])

    print("相关系数：\n", r)

    print(data_new)
    return None


def pac_demo():
    """
    PCA降维
    :return:
    """
    data = [[2, 8, 4, 5], [6, 3, 0, 8], [5, 4, 9, 1]]

    # 1. 实例化一个转换器类
    transfer = PCA(n_components=0.95)
    # 2. 调用fit_transform
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new)
    return None


if __name__ == "__main__":
    # 代码1：sklearn数据集使用
    # datasets_demo()
    # 代码2: 字典特征抽取
    # dict_demo()
    # 代码3：文本特征抽取：CountVecotrizer
    # count_demo()
    # 代码4：中文文本特征抽取：CountVecotrizer
    # count_chinese_demo()
    # 代码5： jieba分词
    # count_chinese_demo2()
    # 代码6: 用TF-IDF的方法进行文本特征抽取
    # tfidf_demo()
    # 代码7: 归一化
    # minmax_demo()
    # 代码8: 标准化
    # stand_demo()
    # 代码9：低方差特征过滤
    # variance_demo()
    # 代码10：PCA降维
    pac_demo()
