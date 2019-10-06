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
### 什么是朴素贝叶斯分类方法
* 对于给出的待分类项，求解在此项出现的条件下各个类别出现的概率，哪个最大，就认为此待分类项属于哪个类别。    
### 概率基础
### 联合概率、条件概率与相互独立
### 贝叶斯公式
* ![微信图片_20191005095219.png](https://i.loli.net/2019/10/05/mvBisJCfru3ZkAw.png)
### API
* ![微信图片_20191005101108.png](https://i.loli.net/2019/10/05/q6F8vzJUtTrfxDB.png)
### 案例：20类新闻分类
1) 获取数据
2) 划分数据集
3) 特征工程 文本特征抽取
4) 朴素贝叶斯预估器流程
5) 模型评估
### 朴素贝叶斯算法总结
* 对缺失数据不敏感， 准确度高，常用于文本分类
* 对特征有关联的样本效果不是很好
### 总结
* 文本分类
## 决策树
### 认识决策树
* 决策树(Decision Tree）是在已知各种情况发生概率的基础上，通过构成决策树来求取净现值的期望值大于等于零的概率，评价项目风险，判断其可行性的决策分析方法，是直观运用概率分析的一种图解法。
### 决策树分类原理详解
* 信息论基础  
    * 信息是用来消除随机不定性的东西
    * 信息熵  
    ![微信图片_20191005165005.png](https://i.loli.net/2019/10/05/QceLlpROkmufnCr.png)
    * 条件熵  
    ![微信图片_20191005165155.png](https://i.loli.net/2019/10/05/QCuZbX5q8sLgOUP.png)
    * 信息增益 = 信息熵 - 条件熵  
### 决策树API
* ![微信图片_20191005170010.png](https://i.loli.net/2019/10/05/EJbxV78Gm2lTXRI.png)
### 案例：泰坦尼克号乘客生存预测
* http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt  
* 获取数据
* 数据处理 缺失值处理 特征值->字典类型
* 准备好特征值和目标值
* 划分数据集
* 特征工程：字典特征抽取
* 决策树预估器流程
* 模型评估
### 决策树可视化
* export_graphviz(estimator, out_file="iris_tree.dot", feature_names=iris.feature_names)
* http://webgraphviz.com/
### 决策树总结
* 优点：可视化 
* 缺点：容易过拟合
### 总结
## 随机森林
### 什么是集成学习方法
* 使用一些（不同的）方法改变原始训练样本的分布，从而构建多个不同的分类器，并将这些分类器线性组合得到一个更强大的分类器，来做最后的决策。  
### 什么是随机森林  
* 随机森林就是通过集成学习的思想将多棵树集成的一种算法，它的基本单元是决策树，而它的本质属于机器学习的一大分支——集成学习（Ensemble Learning）方法。
### 随机森林原理过程
* 随机
    * 特征随机
    * 训练集随机 bootstrap抽样
### API  
![微信图片_20191005185522.png](https://i.loli.net/2019/10/05/SC1lMOV4G3Zkybh.png)
### 随机森林预测案例  
### 总结
* 极好的准确率
* 有效的运行在大数据集上， 处理具有高维特征的样本，不需要降维
* 能够评估各个特征在分类问题上的重要性

# 回归与聚类算法
## 线性回归
### 线性回归的原理
* 在统计学中，线性回归(Linear Regression)是利用称为线性回归方程的最小平方函数对一个或多个自变量和因变量之间关系进行建模的一种回归分析。  
### 线性回归的损失和优化原理
* 目标：求模型参数，一开始不知道，只能随意假设一组参数，不断地靠近真正的一组参数  
* 如何衡量真实值与实验值的差异：损失函数/cost/成本函数/目标函数  
    ![微信图片_20191006164005.png](https://i.loli.net/2019/10/06/GsyDvzSdMCjILTH.png)  
* 优化损失  
    * 优化方法  
        * 正规方程  
        ![微信图片_20191006164537.png](https://i.loli.net/2019/10/06/LzKtsc1UPJBuFZv.png)
        * 梯度下降  
### 线性回归API  
* API  
 ![微信图片_20191006170446.png](https://i.loli.net/2019/10/06/iWbwlKPuoGYc5CU.png)
### 波士顿房价预测
1) 获取数据集
2) 划分数据集
3) 特征工程：无量纲化-标准化
4) 预估器流程
5) 模型评估
    ![微信图片_20191006174350.png](https://i.loli.net/2019/10/06/3aBYWeUXyvocsxh.png)
## 欠拟合与过拟合

## 线性回归的改进——岭回归

## 分类算法——逻辑回归与二分类

## 模型保存与加载

## 无监督学习——K-means算法

## 总结
