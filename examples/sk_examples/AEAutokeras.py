from io import StringIO
from autokeras.engine.block import Block
import autokeras as ak
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.python.util import nest
from numpy import asarray
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
# from sklearn.model_selection import train_test_split
from autokeras import StructuredDataClassifier
from autokeras import StructuredDataRegressor
from tods.sk_interface.detection_algorithm.AutoEncoder_skinterface import AutoEncoderSKI
import matplotlib.pyplot as plt
import os
import glob
import datetime

#how to split yahoo?
#is y true and y pred correct?
#show the notebook error
#how to get autokeras reports

#dataset
dataset = pd.read_csv('metric_detection/concept_drift_data/concept_drift_data_0_from2019-11-16to2019-12-16_5736.csv')
# # dataset = pd.read_csv("metric_detection/periodic_data/periodic_data_4_from2019-12-27to2020-02-27_7622.csv")
dataset["label"] = dataset["label"].astype(int)
# dataset.index=dataset['timestamp']
# print(dataset["timestamp"])




    # ind = range(data_s.shape[0])
# plt.grid(ls='--')  # 生成网格
# plt.plot(final_data[:, 0], 'b-.', linewidth=3, label='真实水位值')
# plt.plot(final_data[:, 1], 'r-.', label='融合注意力机制的Seq2Seq模型(v2)')
# plt.plot(final_data[:, 2], 'y-.', label='SSFLA-CNN&LSTM(v1)')
# plt.plot(final_data[:, 3], 'c-.', label='GRU-LightGBM(v1)')
#
# plt.title("模型性能表现")
# plt.xlabel("时间(小时)")
# plt.ylabel("水位(米)")
# plt.legend() # 显示label内容
# # plt.xticks(ind[::672], data_s[:, 0][::672], rotation=45)
# # plt.savefig("潮汐.png", dpi=100)
# plt.show()


# print(dataset["timestamp"])

# #设置连接多个文件的路径
# files = os.path.join("metric_detection/concept_drift_data/", "*.csv")
#
# #返回的合并文件列表
# files = glob.glob(files)
#
# print("在特定位置加入所有 CSV 文件后生成的 CSV...");
#
# #使用 concat 和 read_csv 加入文件
# dataset = pd.concat(map(pd.read_csv, files), ignore_index=True)
# dataset["label"] = dataset["label"].astype(int)

data = dataset.to_numpy()
labels = dataset.iloc[:,2]
value1 = dataset.iloc[:,1] # delete later
print(labels)
#
#tods primitive
transformer = AutoEncoderSKI()
transformer.fit(data)
tods_output = transformer.predict(data)
prediction_score = transformer.predict_score(data)
print('result from AE primitive: \n', tods_output) #sk report
print('score from AE: \n', prediction_score)

#sk report


y_true = labels
y_pred = tods_output
dataset["label1"]=tods_output


# ubuntu系统下matplotlib中文乱码问题的解决方法：https://www.yingsoo.com/news/posts/71929.html
plt.rcParams['font.sans-serif'] = ['KaiTi', 'SimHei', 'FangSong']  # 使用黑体来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams.update({'font.size':20})

plt.figure(figsize=(24, 14))
# plt.plot(dataset["timestamp"],color='b',label=dataset["label"])

colors = ['blue', 'darkorange']
dataset["timestamp"]=pd.to_datetime(dataset["timestamp"],unit = "ms")
# for idx, station in enumerate(dataset.columns[-1:-1]):
data_s = dataset.loc[:, ['timestamp', 'label']].values
ind = range(data_s.shape[0])
# dataset["timestamp"][::3000]
plt.plot(dataset["timestamp"],dataset["label"], color='blue',label='label',linewidth =5.0,marker='*',linestyle='-')
plt.plot(dataset["timestamp"],dataset["label1"], color='darkorange',label='labe1',marker='*',linestyle='--')
# data_s = dataset.loc[:, ['timestamp', station]].values
# plt.plot(data_s[:, 1], color=colors[idx], label=station)

colors = ['blue', 'darkorange', 'green', 'orange', 'grey', 'teal', 'dodgerblue', 'lightgreen', 'slategrey', 'darkblue', 'orchid', 'lavender']
# for idx, station in enumerate(data_flow.columns[1:-1]):
#     data_s = data_flow.loc[:, ['日期', station]].values
#     plt.plot(data_s[:, 1], color=colors[idx], label=station)
# ind = range(data_s.shape[0])
# plt.grid(ls='--')  # 生成网格
# plt.plot(final_data[:, 0], 'b-.', linewidth=3, label='真实水位值')
# plt.plot(final_data[:, 1], 'r-.', label='融合注意力机制的Seq2Seq模型(v2)')
# plt.plot(final_data[:, 2], 'y-.', label='SSFLA-CNN&LSTM(v1)')
# plt.plot(final_data[:, 3], 'c-.', label='GRU-LightGBM(v1)')
#
# plt.title("模型性能表现")
# plt.xlabel("时间(小时)")
# plt.ylabel("水位(米)")
plt.legend() # 显示label内容
# plt.xticks(ind[::672], data_s[:, 0][::672], rotation=45)
# plt.xticks(ind[::672], data_s[:, 0][::672], rotation=45)
# plt.savefig("潮汐.png", dpi=100)
plt.show()
dataset.set_index('timestamp')
dataset.to_csv("xrb.csv",mode="w")
#
# df = pd.DataFrame(np.random.randn(1000, 4), index=pd.date_range("1/1/2000", periods=1000), columns=list("ABCD"))
# df = df.cumsum()
# df.plot()

# # 生成时间序列：X轴刻度数据
# table = pd.DataFrame([i for i in range(480)],columns=['timestamp'],index=pd.date_range('00:00:00', '23:57:00', freq='180s'))
# # 图片大小设置
# fig = plt.figure(figsize=(15, 9), dpi=100)
# ax = fig.add_subplot(111)
# import matplotlib.dates as mdates
# # X轴时间刻度格式 & 刻度显示
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
# plt.xticks(pd.date_range(table.index[0], table.index[-1], freq='H'), rotation=45)
#
# # 绘图
# ax.plot(table.index, labels, color='r', label='9月12日')
# # ax.plot(table.index, df_0915['avg_speed'], color='y', label='9月15日')
# # ax.plot(table.index, df_0916['avg_speed'], color='g', label='9月16日')
#
# # 辅助线
# sup_line = [35 for i in range(480)]
# ax.plot(table.index, sup_line, color='black', linestyle='--', linewidth='1', label='辅助线')
#
# plt.xlabel('time_point', fontsize=14)  # X轴标签
# plt.ylabel("Speed", fontsize=16)  # Y轴标签
# ax.legend()  # 图例
# plt.title("车速时序图", fontsize=25, color='black', pad=20)
# plt.gcf().autofmt_xdate()
#
# # 隐藏-上&右边线
# ax.spines['right'].set_color('none')
# ax.spines['top'].set_color('none')
#
# plt.savefig('speed.png')
# plt.show()


# plt.xticks(rotation=10)
# # plt.xticks(range(0,len(xData),10),[xData[i] for i in range(0,len(xData),1) if i%10==0],rotation=90,fontsize=8)#横轴标签旋转90°
# plt.gcf().subplots_adjust(left=0.05,top=0.91,bottom=0.09)  # 在此添加修改参数的代码
# xData = dataset.iloc[:, 0].tolist() # 获取dataFrame中的第3列，并将此转换为list
# data1 = dataset.iloc[:, 2].tolist()  # 获取dataFrame中的第3列，并将此转换为list
#
# plt.plot(xData, data1, 'r-^',color = 'black',kind="scatter")  # 画散点图，*:r表示点用*表示，颜色为红色
# # plt.plot(xData, tods_output, 'r-^',color = 'red')  # 画散点图，*:r表示点用*表示，颜色为红色
# plt.legend()
# plt.show()

from IPython import display
from datetime import datetime
from datetime import date
# def myplot(x, y, label=None, xlimit=None, size=(9, 3),fileName=None):
#     display.set_matplotlib_formats('svg')
#     if len(x) == len(y):
#         plt.figure(figsize=size)
#         if xlimit and isinstance(xlimit, tuple):
#             plt.xlim(xlimit)
#         plt.plot(x, y, label=label)
#         if label and isinstance(label, str):
#             plt.legend()
#         if fileName:
#             plt.savefig(fileName)
#         plt.show()
#     else:
#         raise ValueError("x 和 y 的长度不一致！")

# from matplotlib.dates import DateFormatter
#
# def myplot(x, y, label=None, xlimit=None, size=(9, 3),fileName=None):
#     display.set_matplotlib_formats('svg')
#     if len(x) == len(y):
#         plt.figure(figsize=size)
#         if xlimit and isinstance(xlimit, tuple):
#             plt.xlim(xlimit)
#         plt.plot(x, y, label=label)
#         if label and isinstance(label, str):
#             plt.legend()
#         if fileName:
#             plt.savefig(fileName)
#         # ======= 以下是新增代码
#         ax = plt.gca()
#         formatter = DateFormatter('%H:%M')
#         ax.xaxis.set_major_formatter(formatter) # 设置时间显示格式
#         # ==============
#         plt.show()
#     else:
#         raise ValueError("x 和 y 的长度不一致！")
#
#
# myplot(dataset["timestamp"], y_true, xlimit=(date(2019, 1, 1), date(2019, 1, 22) ),)

# import seaborn as sns
# sns.kdeplot(dataset['label'])
# ax = plt.gca()
# ax.plot(dataset["timestamp"],dataset["label"])
# plt.show
# data = np.arange(0,1,1)
# plt.ylim(0,1) #同上
# plt.yticks([0,1]) #设置y轴刻度
# plt.title('ROC')
# plt.xticks(dataset["timestamp"])
# plt.legend(loc = 'lower right')
# plt.xlabel('x') #为x轴命名为“x”
# plt.ylabel('y') #为y轴命名为“y”
# plt.plot(data, y_true)
# plt.plot(data, y_pred)
# plt.show()

# def get_lengths(s):
# #     cols = s.index[::2]
# #     labels = s.index[1::2]
# #     l = list(np.cumsum(list(map(len, s[cols]))))
# #     l = list(zip([0] + l[:-1], l))
# #
# #     return (' '.join(s[cols]), {'entities': [list(zip(l, labels))]})
# # TRAIN_DATA= dataset.apply(get_lengths, axis=1)
# # TRAIN_DATA.to_json("y_true.json")


# with open("y_true.json",
#           'w') as f:
#     f.write(TRAIN_DATA)
#
# with open("y_pred.json",
#           'w') as f:
#     f.write(str(y_pred))

print('Accuracy Score: ', accuracy_score(y_true, y_pred))

print('confusion matrix: \n', confusion_matrix(y_true, y_pred))

print(classification_report(y_true, y_pred))

precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
f1_scores = 2*recall*precision/(recall+precision)

print('Best threshold: ', thresholds[np.argmax(f1_scores)])
print('Best F1-Score: ', np.max(f1_scores))


# 拿到最优结果以及索引
# f1_scores = (2 * precision * recall) / (precision + recall)
# best_f1_score = np.max(f1_scores[np.isfinite(f1_scores)])
# best_f1_score_index = np.argmax(f1_scores[np.isfinite(f1_scores)])
#
# # 阈值
# best_f1_score, thresholds[best_f1_score_index]

#
#
# #classifier
# print('Classifier Starts here:')
# search = StructuredDataClassifier(max_trials=15)
# # perform the search
# search.fit(x=data, y=labels, verbose=0) # y = data label colume
# # evaluate the model
# loss, acc = search.evaluate(data, labels, verbose=0)
# print('Accuracy: %.3f' % acc)
# # use the model to make a prediction
# # row = [0.0200,0.0371,0.0428,0.0207,0.0954,0.0986,0.1539,0.1601,0.3109,0.2111,0.1609,0.1582,0.2238,0.0645,0.0660,0.2273,0.3100,0.2999,0.5078,0.4797,0.5783,0.5071,0.4328,0.5550,0.6711,0.6415,0.7104,0.8080,0.6791,0.3857,0.1307,0.2604,0.5121,0.7547,0.8537,0.8507,0.6692,0.6097,0.4943,0.2744,0.0510,0.2834,0.2825,0.4256,0.2641,0.1386,0.1051,0.1343,0.0383,0.0324,0.0232,0.0027,0.0065,0.0159,0.0072,0.0167,0.0180,0.0084,0.0090,0.0032]
# # X_new = asarray([row]).astype('float32')
# yhat = search.predict(data)
# print('Predicted: %.3f' % yhat[0])
# # get the best performing model
# model = search.export_model()
# # summarize the loaded model
# model.summary()

#regressor
# print('Regressor Starts here:')
# search = StructuredDataRegressor(max_trials=15, loss='mean_absolute_error')
# # perform the search
# search.fit(x=data, y=labels, verbose=0) # y = data label
# mae, _ = search.evaluate(data, labels, verbose=0)
# print('MAE: %.3f' % mae)
# # use the model to make a prediction
# # X_new = asarray([[108]]).astype('float32')
# yhat = search.predict(data)
# print('Predicted: %.3f' % yhat[0])
# # get the best performing model
# model = search.export_model()
# # summarize the loaded model
# model.summary()





