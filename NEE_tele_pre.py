import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, recall_score
from imblearn.combine import SMOTEENN
import matplotlib.pyplot as plt


df = pd.read_excel(r"C:\Users\31498\OneDrive\Desktop\WA_Fn-UseC_-Telco-Customer-Churn.xlsx")

df.drop('customerID', axis=1, inplace=True)

# 处理TotalCharges
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

le = LabelEncoder()
for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = le.fit_transform(df[column])

X = df.drop('Churn', axis=1)
y = df['Churn']

# 使用 SMOTEENN 处理类别不平衡（先做采样再划分训练测试集）
sm = SMOTEENN()
X_resampled, y_resampled = sm.fit_resample(X, y)


X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)


clf = DecisionTreeClassifier(
    max_depth=6, 
    min_samples_leaf=8,
    criterion="gini",
    random_state=42
)

# 训练模型
clf.fit(X_train, y_train)

# 预测概率
y_prob = clf.predict_proba(X_test)[:, 1]

# ROC曲线
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# 9. 混淆矩阵
y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

# 决策树可视化
plt.figure(figsize=(20,10))
plot_tree(clf, 
          feature_names=X.columns, 
          class_names=["No Churn", "Churn"], 
          filled=True, 
          rounded=True, 
          fontsize=10)
plt.title("Decision Tree Diagram (max_depth=6)")
plt.show()

# 输出性能指标
acc = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"准确率（Accuracy）：{acc:.2f}")
print(f"召回率（Recall）：{recall:.2f}")
