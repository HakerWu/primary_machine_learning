from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error


def linear1():
    """
    正规方程对波士顿房价预测
    :return:
    """
    # 1) 获取数据集
    boston = load_boston()

    # 2) 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=22)

    # 3) 特征工程：无量纲化 - 标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 4) 预估器流程
    estimator = LinearRegression()
    estimator.fit(x_train, y_train)

    # 5) 得出模型
    print("正规方程-权重系数为：\n", estimator.coef_)
    print("正规方程-偏置为: \n", estimator.intercept_)

    # 6) 模型评估
    y_predict = estimator.predict(x_test)
    print("预测房价：\n", y_predict)
    error = mean_squared_error(y_test, y_predict)
    print("正规方程-均方误差为：\n", error)
    return None


def linear2():
    """
    梯度下降对波士顿房价预测
    :return:
    """
    # 1) 获取数据集
    boston = load_boston()

    # 2) 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=22)

    # 3) 特征工程：无量纲化 - 标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 4) 预估器流程
    estimator = SGDRegressor()
    estimator.fit(x_train, y_train)

    # 5) 得出模型
    print("梯度下降-权重系数为：\n", estimator.coef_)
    print("梯度下降-偏置为: \n", estimator.intercept_)

    # 6) 模型评估
    y_predict = estimator.predict(x_test)
    print("预测房价：\n", y_predict)
    error = mean_squared_error(y_test, y_predict)
    print("梯度下降-均方误差为：\n", error)

    return None


if __name__ == "__main__":
    linear1()
    linear2()
