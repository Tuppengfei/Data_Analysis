The data set includes information about:

Customers who left within the last month – the column is called Churn
Services that each customer has signed up for – phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies
Customer account information – how long they’ve been a customer, contract, payment method, paperless billing, monthly charges, and total charges
Demographic info about customers – gender, age range, and if they have partners and dependents


根据这些建立logit回归模型，auc=0.83
再根据续约的客户盈利100元，而预测续约实际没有续约的客户会导致公司损失20元，结合预测结果确定阈值以获得最大利润。
