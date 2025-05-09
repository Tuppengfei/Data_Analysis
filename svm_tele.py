#先对数据继续正态标准化处理，统一尺度
#再根据数据结果的正负类别进行加权处理
#最后使用SVM进行分类预测
#当 C 无穷大时，要使最终的损失最小化，迫使所有的样本都要被正确分类，即硬间隔
#当 C 为合理常数时，允许有误分类的样本，即软间隔
#gamma 是径向基函数的参数，决定了高斯核函数的宽度，gamma 越大，模型越复杂，容易过拟合；gamma 越小，模型越简单，容易欠拟合

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, accuracy_score, recall_score
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_excel(r"C:\Users\31498\OneDrive\Desktop\WA_Fn-UseC_-Telco-Customer-Churn.xlsx")
df.drop('customerID', axis=1, inplace=True)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# 类别变量编码
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])


y = df['Churn']
X = df.drop('Churn', axis=1)

# 要标准化的变量
continuous_vars = ['MonthlyCharges', 'TotalCharges']
scaler = StandardScaler()
X[continuous_vars] = scaler.fit_transform(X[continuous_vars])# 进行标准化


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

svm = SVC(
    probability=True,
    kernel='rbf',
    C=24,
    gamma=0.01,
    class_weight={0: 1, 1: 4},  # 类别1 的权重为类别0 的4倍
    random_state=42
)

svm.fit(X_train, y_train)

y_prob = svm.predict_proba(X_test)[:, 1]
y_pred = svm.predict(X_test)

roc_auc = roc_auc_score(y_test, y_prob)
recall = recall_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print(f"AUC: {roc_auc:.3f}")
print(f"召回率（Recall）: {recall:.3f}")
print(f"准确率（Accuracy）: {accuracy:.3f}")

plt.figure(figsize=(8,6))
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.2f})', lw=2)
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('假阳性率')
plt.ylabel('真正率')
plt.title('支持向量机 ROC 曲线（类别1 权重4倍）')
plt.legend()
plt.grid(True)
plt.show()

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["未流失", "流失"])
disp.plot(cmap='Blues')
plt.title('SVM 混淆矩阵（类别1 权重4倍）')
plt.show()
