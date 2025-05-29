The data set includes information about:
Customers who left within the last month – the column is called Churn
Services that each customer has signed up for – phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies
Customer account information – how long they’ve been a customer, contract, payment method, paperless billing, monthly charges, and total charges
Demographic info about customers – gender, age range, and if they have partners and dependents

根据客户信息预测客户的流失情况，为了实现利润最大化，我们着重提升模型的召回率
我们在stata上建立逻辑回归模型，得到的模型召回率为0.85；在python上建立的决策树模型，得到的模型召回率为0.95

综合评价：
逻辑回归：权重调整后的模型召回率达到 0.85，解释性强，便于业务理解和实际应用，过拟合风险低。
决策树：召回率 0.95，AUC 显著高于逻辑回归，但解释性较差，模型复杂度较高。
