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
replace weight = 3.7 if Churn == 1   // 正类：续保
replace weight = 1.4 if Churn == 0   // 负类：不续保

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


<<<<<<< HEAD:电信预测优化版.do

logit Churn i.n_gender i.n_Partner i.n_Dependents i.n_PhoneService i.n_MultipleLines i.n_InternetService i.n_OnlineSecurity i.n_OnlineBackup i.n_DeviceProtection i.n_TechSupport i.n_StreamingTV i.n_StreamingMovies i.n_Contract i.n_PaperlessBilling i.n_PaymentMethod MonthlyCharges TotalCharges [pweight=weight]
=======
logit Churn i.n_gender i.n_Partner i.n_Dependents i.n_PhoneService i.n_MultipleLines i.n_InternetService i.n_OnlineSecurity i.n_OnlineBackup i.n_DeviceProtection i.n_TechSupport i.n_StreamingTV i.n_StreamingMovies i.n_Contract i.n_PaperlessBilling i.n_PaymentMethod MonthlyCharges TotalCharges//[pweight=weight]
>>>>>>> 3bcbd3f05c0db83ed66b161aef63b217d6e30561:电信预测.do
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



*拓展：根据利润最大值计算最优阈值
* 1. 设置随机种子（保持可重复）
set seed 12345

* 2. 如果之前生成过cost，先删除
capture drop cost

* 3. 创建一个空变量 cost（目前其实不用，但保留不影响）
gen cost = .

* 4. 开始循环：cutoff 从 0.10 到 0.90
forvalues c = 10(5)90 {
    local cutoff = `c'/100   // 当前的分类阈值

    * 5. 生成预测类别
    gen pred = (phat > `cutoff')

    * 6. 标记真正例（TP）：实际续保，预测也续保
    gen TP = (Churn == 1 & pred == 1)

    * 7. 标记假阳性（FP）：实际不续保，但预测为续保
    gen FP = (Churn == 0 & pred == 1)

    * 8. 统计 TP 和 FP 数量
    summarize TP
    local tp_count = r(sum)

    summarize FP
    local fp_count = r(sum)

    * 9. 计算总收益 = 100 × TP 数量 - 20 × FP 数量
    local total_profit = 100 * `tp_count' - 20 * `fp_count'

    * 10. 显示结果
    di "cutoff = `cutoff' 总利润 = `total_profit' (真正例:`tp_count' 假阳性:`fp_count')"

    * 11. 清理临时变量
    drop pred TP FP
}


