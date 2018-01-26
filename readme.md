# 360机器写作与人类写作的巅峰对决
baseline队(3/589)&nbsp;&nbsp;|&nbsp;&nbsp;[赛题链接](http://www.datafountain.cn/#/competitions/276/intro)

#### 任务：
根据上下文语法语义的一致性与连续性，判断文章是否为人类写作。

#### 数据：

| 阶段      |     训练集 |   测试集   |
| :-------- | --------:| :------: |
| 初赛Part1    |   20w(12w+, 8w-) |  15w  |
| 初赛Part2    |   30w(6w+, 24w-) |  25w  |
| 复赛    |   60w(24w+, 36w-) |  40w  |

##### 模型分数

| 模型      |     类型 |   分数   |
| :-------- | --------:| :------: |
| CNN    |   word |  0.9014  |
| CNN    |   postag+char |  0.8982  |
| HAN    |   word |  0.9047  |
| HCN    |   word |  0.9056  |

#### code
##### 目录说明
* data 原始数据目录
* cache 缓存文件路径
* feature 传统方法样本特征代码
* libs 引用的开源组件
* models 模型代码
* train 模型训练脚本
* utils 数据处理及其他脚本

##### 运行说明
```bash
## 获取数据集
sh ./data/get_final_data.sh
## 获取ltp库
sh ./libs/get_ltp.sh
## 预训练词向量
python3 ./utils/w2v.py
## 构建线下训练验证集，生成序列文件
python3 ./utils/data_preprocess.py
## 后续即可运行train目录下脚本训练模型
```
