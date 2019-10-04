# primary_machine_learning
# 数据预处理  
## 归一化
* 公式  
   X' = (x - min)/(max - min)  
   X" = X' * (mx - mi) + mi  
* API  
    ```python
    sklearn.preprocessing.MinMaxScaler(feature_range=(mi, mx)...)
    MinMaxScaler.fit_transform(X)
    X: numpy array格式的数据[n_samples, n_features]
    返回值: 转换后的形状相同的array
    ```
* 总结  
    最大值和最小值容易受异常点影响，所以这种方法鲁棒性较差，只适合传统精确小数据场景。  

## 标准化  
* 公式  
    X' = (x - mean) / std
* API
    ```python
    sklearn.preprocessing.StandardScaler()
    处理之后， 对每列来说， 所有数据都聚集在均值为0附近， 标准差为1
    StandardScaler.fit_transform(X)
      x: numpy array格式的数据[n_samples, n_features]
    返回值: 转换后的形状相同的array
    ```
* 总结
    适用于数据嘈杂的大数据场景  
   
   
# 数据降维
## 概念
    降维是指在某些限定条件下，降低随机变量（特征）的个数，得到一组“不相关”主变量的过程。  
## 方法
* 特征选择  
    1. Filter 过滤式：  
        (1) 方差选择法：低方差特征过滤  
        (2) 相关系数：特征与特征之间的相关程度强的过滤  
    2. Embedded 嵌入式  
        (1) 决策树  
        (2) 正则化  
        (3) 深度学习  
* 主成分分析

## API  
sklearn.feature_selection  
1. 低方差特征过滤  
    * sklearn.feature_selection.VarianceThreshold(threshold = 0.0)  
    * 删除所有低方差特征
    * Variance.fit_transform(X)
        * X: numpy array 格式的数据[n_samples, n_features]
        * 返回值: 训练集差异低于threshold的特征将被删除。默认值是保留所有非零方差特征，即删除所有样本中具有相同值的特征。  
2. 相关系数
    * 皮尔逊相关系数(取值范围[-1,1])
    ![微信图片_20191003162915.png](https://i.loli.net/2019/10/03/5QdTCqyEB3ZgDlx.png)
    * from scipy.stats import pearsonr
        * x:(N,) array_like
        * y:(N,) array_like  
        * return:(coefficient, p-value)
    * 如果相关性很高怎么办
        * 选取其中一个
        * 加权求和
        * 主成分分析
        
# 主成分分析(PCA)
## API
* sklearn.decomposition.PCA(n_components=None)
    * 将数据分解为较低维空间
    * n_components:
        * 小数：表示保留百分之多少的信息
        * 整数：减少到多少特征
    * PCA.fit_transform(X)
        * X:numpy array格式的数据[n_samples, n_features]
    * 返回值：转换后指定维度的array
## 案例：探究用户对物品类别的喜好细分
    https://jupyter.dgutdev.ml/notebooks/work/personal_folder/wjer/instacart_from_kaggle.ipynb
    

# 分类算法
## sklearn转换器和预估器
### 转换器(transformer)
* 实例化
* 调用fit_transformer
### 预估器(estimator)
* 实例化一个estimator
* estimator.fit(x_train, y_train)
* 模型评估
    * y_predict = estimator.predict(x_test)  
    then check if y_test == y_predict
    * 计算准确率  
    estimator.score(x_test, y_test)
## KNN算法(K-近邻算法)
### 什么是K-近邻算法
* 存在一个样本数据集合，也称为训练样本集，并且样本集中每个数据都存在标签，即我们知道样本集中每一数据与所属分类对应的关系。输入没有标签的数据后，将新数据中的每个特征与样本集中数据对应的特征进行比较，提取出样本集中特征最相似数据（最近邻）的分类标签。  
* 确定最近距离
    * 欧式距离
    ![微信图片_20191004132001.png](https://i.loli.net/2019/10/04/eaUsf9yS3XPdnEV.png)
    * 曼哈顿距离
    ![微信图片_20191004132001.png](https://i.loli.net/2019/10/04/BMhJlQqUDyzp3nZ.png)
    * 明可夫斯基距离  
    ![微信图片_20191004133038.png](https://i.loli.net/2019/10/04/kSEfPVbxHZtQoUq.png)
* K的取值  
    若K过小，容易受异常点影响；若K过大，受样本不均衡的影响。
* 使用KNN前的数据处理
    无量纲化的处理：标准化
### K-近邻算法API
* sklearn.neighbors.KNeighborsClassifier(n_neighbors=5, algorithm='auto')
    * n_neighbors: int,可选(默认=5)
    * algorithm: {'auto', 'ball_tree', 'kd_tree', 'brute'}
### 案例：鸢尾花种类预处
1) 获取数据
2) 数据集划分
3) 特征工程
    标准化
4) KNN
5) 模型评估
### 总结
* 缺点：K值难确定；内存开销大
## 模型选择和调优
### 交叉验证
* 基本思想就是将原始数据（dataset）进行分组，一部分做为训练集来训练模型，另一部分做为测试集来评价模型。
### 超参数搜索-网格搜索
* 一种调参手段；穷举搜索：在所有候选的参数选择中，通过循环遍历，尝试每一种可能性，表现最好的参数就是最终的结果。其原理就像是在数组里找最大值。
* sklearn.model_selection.GridSearchCV(estimator,param_grid,cv)  
    * estimator：估计器对象
    * param_grid：估计器参数，参数名称（字符串）作为key，要测试的参数列表作为value的字典，或这样的字典构成的列表
    * cv：整形，指定K折交叉验证
    * fit()：输入训练数据
    * score()：准确率
    * best_score_：交叉验证中测试的最好的结果
    * best_estimator_：交叉验证中测试的最好的参数模型
    * best_params_：交叉验证中测试的最好的参数
    * cv_results_：每次交叉验证的结果
### 鸢尾花案例增加K值调优
### 案例：预测facebook签到位置
* 由于找不到数据，没实践
### 总结
## 朴素贝叶斯算法

## 决策树

## 随机森林