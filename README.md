# IBM Human Resources Attrition

This project investigates IBM Human Resources Attrition Data Set. 

# Introduction

Companies hire many employees every year. To create a positive working and learning environment, firms invest time and money in trianing the new members and also to get existing employees involved as well. The goal of these programs aim to increase the effectiveness of the employees and in doing so the firm as a whole can have better output in long run.

## Attrition

The single most important feature we are interested in is attrition. Attrition in human resources refer to the gradual loss of employees over time. In general relatively high attrition is problematic for companies. Human Resource professionals often asume a leadership role in designing company compensation programs, work culture and motivation systems that help the organization retain top employees.

## Data

To investigate this topic, I use [IBM HR Analytics Employee Attrition and Performance Dataset](https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset/data). There are total of 1470 samples and 35 features. Among the target, Attrition, there are 237 candidates committed to Yes (i.e. left the company) and the rest 1233 candidates committed to No (i.e. stayed at the company).

## EDA

We can explore using histograms and correlation plots.

Multiple histograms:
<p align="center">
  <img width="800" src="https://github.com/yiqiao-yin/IBM-HR-Attrition/blob/master/figs/fig-2-multi-hist.PNG">
</p>

Correlation plot:
<p align="center">
  <img width="800" src="https://github.com/yiqiao-yin/IBM-HR-Attrition/blob/master/figs/fig-1-corrplot.PNG">
</p>

Based on correlation, Attrition is associated negatively with Age, JobInvolvement, JobLevel, Jobsatisfaction, MonthlyIncome, StockOptionLevel, TotalWorkingYears, YearsAtCompany, YearsInCurrentCompany, YearsInCurrentRole, and YearsWithCurrManager.

For the people who left the firm (committed to Yes to Attrition), the most common JobRole is Laboratory Technician and Sales Representative. From our analysis below, we see that the Laboratory Technician who spent a year at the firm and than left sat on a high of 30.9% among those who committed Yes to Attrition. The second is Sales Representative that stayed at the firm for a year, at 9.1%. The third group of people who stayed at the firm for a year and left are Research Scientist, at a shy of 17.6%. These are the top three demographics that contribute to the Attrition the highest.

# Results

We select the following variables based on partition influence measure.

| Top Module | Measure | Variables Names |
| --- | --- | --- |
| 8, 15, 29 | 13.86 | EducationField, JobRole, YearsAtCompany |
| 15, 29 | 13.3 | JobRole, YearsAtCompany |

We attempted using Bagging, Gradient Boosting Machine, Naive Bayes, Linear Model, Random Forest, iterative Random Forest, and Bayesian Additive Regression Tree (BART).

| Name | Result (Measured by AUC) |
| --- | --- |
| Bagging |	0.904 |			
| GBM |	0.526 |			
| NB |	0.452 |	
| LM |	0.745 |		
| RF |	0.544 |		
| iRF |	0.808 |		
| BART |	0.890 |		
