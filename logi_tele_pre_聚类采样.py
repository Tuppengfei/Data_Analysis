# 0. 导入所需库
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import statsmodels.formula.api as smf
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, accuracy_score, recall_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 设置matplotlib正常显示中文和负号
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 加载数据
try:
    df = pd.read_excel(r"C:\Users\31498\OneDrive\Desktop\WA_Fn-UseC_-Telco-Customer-Churn.xlsx")
    print("数据加载成功。")
except FileNotFoundError:
    print("错误：无法找到指定路径的文件，请检查文件路径是否正确。")
    exit() # 如果文件不存在，则退出程序

# 2. 初始数据预处理
# 删除ID列
df.drop('customerID', axis=1, inplace=True)

# 转换目标变量
df['Churn'] = df['Churn'].replace({'No': 0, 'Yes': 1})

# 处理数值类型列中的错误和缺失值
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
df['MonthlyCharges'].fillna(df['MonthlyCharges'].median(), inplace=True)

# 定义分类变量列表，确保它们是字符串类型以便后续处理
char_vars = ['gender', 'Partner', 'Dependents', 'PhoneService', 
             'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
             'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
             'Contract', 'PaperlessBilling', 'PaymentMethod']

for var in char_vars:
    df[var] = df[var].astype(str)
    
print("初始数据预处理完成。")

# 3. 划分训练集与测试集
# 这是保证模型评估可靠性的关键一步，使用分层抽样确保训练集和测试集中流失比例相似
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Churn'])
print(f"数据已划分为训练集（{len(train_df)}条）和测试集（{len(test_df)}条）。")


# 4. 进一步数据处理（在划分后进行）
# 定义处理极端值的函数（3σ原则 + 线性插值）
def clean_outliers(data, column):
    mean = data[column].mean()
    std = data[column].std()
    # 识别超出3个标准差的值
    is_outlier = ~((data[column] >= mean - 3*std) & (data[column] <= mean + 3*std))
    # 将极端值设为NaN，然后用线性插值填充
    data.loc[is_outlier, column] = np.nan
    data[column] = data[column].interpolate(method='linear')
    return data

# 分别对训练集和测试集进行异常值处理
for col in ['TotalCharges', 'MonthlyCharges']:
    train_df = clean_outliers(train_df, col)
    test_df = clean_outliers(test_df, col)

print("训练集和测试集的异常值处理完成。")

# 5. 对训练集的多数类进行K-Means欠采样
print("开始对训练集进行K-Means欠采样...")
minority = train_df[train_df['Churn'] == 1].copy()
majority = train_df[train_df['Churn'] == 0].copy()

# 定义用于聚类的数值特征
features_for_clustering = ['MonthlyCharges', 'TotalCharges']

# 设定聚类数量k等于少数类的样本数
k = len(minority)
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10) # n_init=10 避免局部最优

# 拟合K-means并为多数类样本分配簇
majority['cluster'] = kmeans.fit_predict(majority[features_for_clustering])

# 基于聚类中心构造新的代表性样本
centroid_rows = []
for i in range(k):
    group = majority[majority['cluster'] == i]
    # 如果某个簇为空，则跳过
    if len(group) == 0:
        continue
        
    # 创建行，包含数值特征的聚类中心
    row = dict(zip(features_for_clustering, kmeans.cluster_centers_[i]))
    # 设定目标变量为0（多数类）
    row['Churn'] = 0
    # 为分类变量找到该簇中最常见的值（众数）
    for var in char_vars:
        row[var] = group[var].mode()[0]
        
    centroid_rows.append(row)

# 将构造的代表性样本转换为DataFrame
centroids = pd.DataFrame(centroid_rows)

# 合并新的代表性多数类样本与原始的少数类样本，构成最终的训练集
resampled_train_df = pd.concat([centroids, minority], ignore_index=True)
print(f"欠采样完成，新的训练集包含 {len(resampled_train_df)} 条数据。")

# 6. 构建并训练Logistic回归模型
# 构建statsmodels公式，使用C()来指定分类变量
formula = 'Churn ~ ' + ' + '.join([f'C({v})' for v in char_vars] + features_for_clustering)

# 在重采样后的训练集上拟合模型
model = smf.logit(formula, data=resampled_train_df)
results = model.fit()
print("\n--- 模型训练完成 ---")
print(results.summary())

# 7. 在独立的测试集上进行预测与评估
print("\n--- 在测试集上进行评估 ---")
# 使用训练好的模型对`test_df`进行预测
test_df['phat'] = results.predict(test_df)

# 根据阈值进行分类（0.3是一个可调整的参数，旨在提高召回率）
threshold = 0.3
test_df['phat_class'] = (test_df['phat'] > threshold).astype(int)

# 计算各项评估指标
fpr, tpr, _ = roc_curve(test_df['Churn'], test_df['phat'])
auc = roc_auc_score(test_df['Churn'], test_df['phat'])
acc = accuracy_score(test_df['Churn'], test_df['phat_class'])
recall = recall_score(test_df['Churn'], test_df['phat_class'])

# 8. 展示评估结果
print("\n混淆矩阵 (测试集):")
cm = confusion_matrix(test_df['Churn'], test_df['phat_class'])
print(cm)
print(f"\n准确率 (Accuracy): {acc:.3f}")
print(f"召回率 (Recall): {recall:.3f}")
print(f"AUC 值: {auc:.3f}")

# 9. 结果可视化
# 绘制混淆矩阵图
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix on Test Set')
plt.show()

# 绘制ROC曲线
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'AUC on Test Set = {auc:.3f}')
plt.plot([0,1], [0,1], 'k--', lw=2, label='Random Chance')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve on Test Data')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()