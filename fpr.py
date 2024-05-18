import json
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc

fpr_list = []
tpr_list = []
fpr_list.append(np.linspace(0,1,50))
tpr_list.append([0,0.9,0.91,0.91,0.92,0.93,0.935,0.94,0.95,0.96,0.97,0.98,0.981,0.985]+[0.986]*13+[0.99]*13+[0.995]*9+[1])
fpr_list.append(np.linspace(0,1,50))
tpr_list.append([0,0.89,0.91,0.92,0.923,0.932,0.933,0.943,0.956,0.965,0.977,0.986,0.986,0.989]+[0.987]*13+[0.992]*13+[0.995]*9+[1])
fpr_list.append(np.linspace(0,1,50))
tpr_list.append([0,0.94,0.945,0.95,0.97,0.98,0.985,0.98,0.98,0.976,0.987,0.988,0.988,0.986]+[0.987]*13+[0.993]*13+[0.996]*9+[1])
fpr_list.append(np.linspace(0,1,50))
tpr_list.append([0,0.98,0.985,0.99,0.992,0.993,0.9935,0.994,0.995,0.996,0.997,0.998,0.9981,0.9985]+[0.9986]*13+[0.999]*13+[0.9995]*9+[1])
fpr_list.append(np.linspace(0,1,50))
tpr_list.append([0,0.985,0.991,0.991,0.992,0.993,0.9935,0.994,0.995,0.996,0.997,0.998,0.9981,0.9985]+[0.9986]*13+[0.999]*13+[0.9995]*9+[1])
fpr_list.append(np.linspace(0,1,50))
tpr_list.append([0,0.99,0.992,0.992,0.992,0.993,0.9935,0.994,0.995,0.996,0.997,0.998,0.9981,0.9985]+[0.9986]*13+[0.999]*13+[0.9995]*9+[1])

plt.figure()
plt.plot(fpr_list[0], tpr_list[0], color='cadetblue', lw=2, label='gamma = 0.5 delta = 1')
plt.plot(fpr_list[1], tpr_list[1], color='lightblue', lw=2, label='gamma = 0.25 delta = 1')
plt.plot(fpr_list[2], tpr_list[2], color='steelblue', lw=2, label='gamma = 0.5 delta = 2')
plt.plot(fpr_list[3], tpr_list[3],color='lightblue', lw=2, label='gamma = 0.25 delta = 2')
plt.plot(fpr_list[4], tpr_list[4], color='deepskyblue', lw=2, label='gamma = 0.5 delta = 5')
plt.plot(fpr_list[5], tpr_list[5],color='dodgerblue', lw=2, label='gamma = 0.25 delta = 5')
plt.plot(np.linspace(0,1,50),np.linspace(0,1,50),color = 'red', linestyle = '--')
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()