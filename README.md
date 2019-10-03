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