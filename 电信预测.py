import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, accuracy_score, recall_score
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False


df = pd.read_excel(r"C:\Users\31498\OneDrive\Desktop\WA_Fn-UseC_-Telco-Customer-Churn.xlsx")

df.drop('customerID',axis=1,inplace=True)

print(df.dtypes)

df['TotalCharges']=pd.to_numeric(df['TotalCharges'],errors='coerce')#将无法转换的数据强制转换为NaN
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

char_vars = ['gender', 'Partner', 'Dependents', 'PhoneService','SeniorCitizen',
             'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
             'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
             'Contract', 'PaperlessBilling', 'PaymentMethod']


label_encoders = {}
le=LabelEncoder()
for var in char_vars:
    if df[var].dtype == 'object':
        df[var] = le.fit_transform(df[var])
        label_encoders[var] = le

df['Churn'] = df['Churn'].replace({'No': 0, 'Yes': 1})#yes为已经流失；赋值为1；


def clear_data(df,column):
    mean = df[column].mean()#取该列的平均值
    std = df[column].std()#取该列的方差

    df[column] = df[column].where((df[column] >= mean - 3 * std) & (df[column] <= mean + 3 * std))#用3σ原则取出极端值

    return df

for columns in ['TotalCharges','MonthlyCharges']:
    df=clear_data(df,columns)
#print(df)
print(df.dtypes)


plt.figure(1)
count_classes=pd.value_counts(df['Churn'],sort=True).sort_index()
count_classes.plot(kind='pie')
plt.title("类别占比图")
plt.xlabel("类别")
plt.ylabel("数量")
plt.show()

X=df.drop('Churn', axis=1)
y=df['Churn']

number_recods=len(df[df['Churn']==1])
number_recods_index=np.array(df[df['Churn']==1].index)

normal_index=df[df['Churn']==0].index

random_normal_indices = np.random.choice(normal_index,number_recods,replace=False)

random_normal_indices = np.array(random_normal_indices)

unsample_index=np.concatenate([number_recods_index,random_normal_indices])

unsample_data=df.iloc[unsample_index,:]

X_unsample = unsample_data.drop('Churn',axis=1)
y_unsample = unsample_data['Churn']

X_train_undersample, X_test_undersample, y_train_undersample, y_test_undersample = train_test_split(X_unsample
                                                                                                   ,y_unsample
                                                                                                   ,test_size = 0.3
                                                                                                   ,random_state = 0)

print("")
print("下采样训练集包含样本数量: ", len(X_train_undersample))
print("下采样测试集包含样本数量: ", len(X_test_undersample))
print("下采样样本总数: ", len(X_train_undersample)+len(X_test_undersample))


train_df=pd.DataFrame(X_train_undersample,columns=X.columns)
train_df['Churn'] = y_train_undersample

test_df=pd.DataFrame(X_test_undersample,columns=X.columns)
test_df['Churn'] = y_test_undersample


formula_parts = []
for col in X.columns:
    if col in char_vars:
        formula_parts.append(f"C({col})")
    else:
        formula_parts.append(col)
formula = 'Churn ~ ' + ' + '.join(formula_parts)
print(f"\n使用的模型公式:\n{formula}\n")


model = smf.logit(formula,data=train_df)
result = model.fit()
print(result.summary())


test_df['phat'] = result.predict(test_df)

fpr, tpr, _ = roc_curve(test_df['Churn'], test_df['phat'])
auc_score = roc_auc_score(test_df['Churn'], test_df['phat'])

test_df['phat_class'] = (test_df['phat'] > 0.5).astype(int)
cm = confusion_matrix(test_df['Churn'], test_df['phat_class'])
acc = accuracy_score(test_df['Churn'], test_df['phat_class'])
recall = recall_score(test_df['Churn'], test_df['phat_class'])

print("\n测试集混淆矩阵:")
print(cm)
print(f"准确率: {acc:.3f}")
print(f"召回率: {recall:.3f}")
print(f"AUC 值: {auc_score:.3f}")
# ROC 曲线绘制
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}", linewidth=2)
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("ROC Curve (Test Data)")
plt.legend(loc='lower right')
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.show()


#  混淆矩阵热力图 
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Churn (0)', 'Churn (1)'],
            yticklabels=['Not Churn (0)', 'Churn (1)'])
plt.title('Confusion Matrix (Test Data)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

# 概率分布图 
plt.figure(figsize=(10, 6))
sns.histplot(test_df['phat'], kde=True, bins=50, color='skyblue', stat="density")
plt.axvline(0.5, color='red', linestyle='--', label='Threshold (0.5)')
plt.title('Predicted Churn Probability Distribution (Test Data)')
plt.xlabel('Predicted Probability of Churn')
plt.ylabel('Density')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.show()

number_of_rows = len(train_df)
print(f"DataFrame train_df 中有 {number_of_rows} 行。")

count_churn_1 = (df['Churn'] == 1).sum()
print(f"'Churn' 列中值为 1 的行数是: {count_churn_1}")






























