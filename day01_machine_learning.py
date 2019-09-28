from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import jieba


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
    count_chinese_demo2()