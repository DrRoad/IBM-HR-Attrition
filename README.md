# IBM Human Resources Attrition

This project investigates IBM Human Resources Attrition Data Set. 

<p align="center">
  <img width="1000" src="https://github.com/yiqiao-yin/IBM-HR-Attrition/blob/master/figs/background.gif">
</p>


# Acknowledgement

I want to thank Natasha for the blog on [People Analytic](https://towardsdatascience.com/people-analytics-with-attrition-predictions-12adcce9573f) for providing background knowledge for me to understand Human Resources departmets. I also want to thank IBM analysts for providing data set on [Kaggle](https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset/data). 

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
| Bagging or Bootstrap Aggregation |	0.904 |			
| Gradient Boosting Machine |	0.526 |			
| Naive Bayes |	0.50 |	
| Linear Model or Least Squares |	0.745 |		
| Random Forest |	0.544 |		
| iterative Random Forest |	0.808 |		
| Bayesian Additive Regression Tree (BART) |	0.890 |	

The implication is that a new candidate walks in the door and by collecting the information of EducationField, JobRole, and YearsAtCompany I can tell manager this employee will commit Yes to Attrition with a certain probability that is on average 90% accurate!
