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