local char_vars gender Partner Dependents PhoneService MultipleLines InternetService OnlineSecurity OnlineBackup DeviceProtection TechSupport StreamingTV StreamingMovies Contract PaperlessBilling PaymentMethod 

replace Churn = "0" if Churn == "No"
replace Churn = "1" if Churn == "Yes"

foreach var in `char_vars' {
    encode `var', gen(n_`var')  
}


describe
save telco_churn_encoded.dta, replace

destring Churn , replace
destring TotalCharges , replace

gen weight = .
replace weight = 4 if Churn == 1   // 正类：续保
replace weight = 1 if Churn == 0   // 负类：不续保

gen flag_extreme = 0

* 处理 TotalCharges
sum TotalCharges if Churn==0
local mu_TC = r(mean)
local sigma_TC = r(sd)
replace flag_extreme = 1 if Churn==0 & ///
    (TotalCharges < `mu_TC' - 3*`sigma_TC' | TotalCharges > `mu_TC' + 3*`sigma_TC')

* 处理 MonthlyCharges
sum MonthlyCharges if Churn==0
local mu_MC = r(mean)
local sigma_MC = r(sd)
replace flag_extreme = 1 if Churn==0 & ///
    (MonthlyCharges < `mu_MC' - 3*`sigma_MC' | MonthlyCharges > `mu_MC' + 3*`sigma_MC')

* 删除类别0中的极端值
count if flag_extreme==1
display "删除类别0的极端值数量：" r(N)
drop if flag_extreme==1



logit Churn i.n_gender i.n_Partner i.n_Dependents i.n_PhoneService i.n_MultipleLines i.n_InternetService i.n_OnlineSecurity i.n_OnlineBackup i.n_DeviceProtection i.n_TechSupport i.n_StreamingTV i.n_StreamingMovies i.n_Contract i.n_PaperlessBilling i.n_PaymentMethod MonthlyCharges TotalCharges [pweight=weight]
predict phat
roctab Churn phat
//lroc
//estat classification   
//estat gof, group(10) 

gen phat_class = phat > 0.5   // 阈值0.5，可以根据需要调整

tabulate Churn phat_class, matcell(conf)   // 生成混淆矩阵

* 混淆矩阵：
* conf[1,1] = True Negatives (TN)
* conf[1,2] = False Positives (FP)
* conf[2,1] = False Negatives (FN)
* conf[2,2] = True Positives (TP)

local TN = conf[1,1]
local FP = conf[1,2]
local FN = conf[2,1]
local TP = conf[2,2]

local Accuracy = (`TP' + `TN') / (`TP' + `TN' + `FP' + `FN')
local Recall = `TP' / (`TP' + `FN')

display "准确率（Accuracy）: " %6.3f `Accuracy'
display "召回率（Recall）: " %6.3f `Recall'




