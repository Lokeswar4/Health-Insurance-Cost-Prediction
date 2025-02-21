# Health-Insurance-Cost-Prediction

# <a id='1'> Table of Contents </a>

1. [Background](#intro)
2. [Objectives](#2)
3. [Attribute Information](#3)
4. [Conclusions & Recommendations](#4)

# <a id='intro' href=#1> Background </a>

In the health insurance industry, insurance companies often face challenges in determining accurate premiums for each policyholder. Errors in assessing health risks can result in significant financial losses. Therefore, accurate health insurance premium determination is crucial to maintain financial stability for insurance companies and provide fair services to policyholders.

In this project, we will use a dataset containing information about health insurance policyholders, including age, gender, Body Mass Index (BMI), number of children, smoking habits, residential region, and individual medical charges billed by insurance. This dataset serves as a valuable data source to develop predictive models that can assist health insurance companies in assessing risks and determining more accurate insurance premiums.

Furthermore, improved risk assessment can aid insurance companies in developing a product portfolio that aligns better with policyholder needs, enhancing competitiveness in the market.

Therefore, this project directly impacts the operational and business strategy of health insurance companies, ultimately providing benefits to both the companies and their policyholders.

# <a id='2' href=#1> Objectives </a>

The primary objectives of this project are to develop predictive models that can help health insurance companies in:

1. **Accurate Premium Determination:** Utilize policyholder data to calculate more accurate premiums based on the health risks faced by each policyholder. Consequently, insurance companies can minimize financial losses due to inaccurate premiums.

2. **Health Risk Assessment:** Identify risk factors that impact individual medical costs, such as age, BMI, number of children, and smoking habits. This can help insurance companies evaluate and manage risks more effectively.

3. **Product Portfolio Development:** Develop insurance products that better align with the individual needs of policyholders based on their characteristics, enhancing customer satisfaction.

# <a id='3' href=#1> Attribute Information </a>

- **age:** age of primary beneficiary
- **sex:** insurance contractor gender, female, male
- **bmi:** Body mass index, providing an understanding of body, weights that are relatively high or low relative to height, objective index of body weight (kg / m<sup>2</sup>) using the ratio of height to weight, ideally 18.5 to 24.9
- **children:** Number of children covered by health insurance / Number of dependents
- **smoker:** Smoking
- **region:** the beneficiary's residential area in the US, northeast, southeast, southwest, northwest
- **charges**: Individual medical costs billed by health insurance.

# <a id='4' href="#1"> Conclusions & Recommendations</a>
Based on the analysis and modeling conducted, here are some key points that can be summarized:
- Through correlation analysis using Spearman and Pearson, as well as exploratory data analysis (EDA), we have identified that the feature with a significant influence on insurance bills is "smoker." Additionally, the features "age," "bmi," and "children" also have a notable impact on insurance bills.
- Based on the permutation feature importance analysis, it turns out that not only those four features have an impact on the model, but "region_northwest" and "region_northeast" also contribute to the model's performance. This phenomenon may occur due by several factors, notably the availability of higher-cost healthcare facilities compared to other regions, elevated risks associated with natural disasters, or stringent insurance regulations in comparison to other areas. However, the influence of "region_northwest" and "region_northeast" is relatively small but in this experiment, we have attained an improved score when compared to utilizing only four features.
- The Random Forest model has proven to be the best model in this experiment, primarily because it is more robust to outliers compared to Linear Regression.
- We have found that the best parameters for the Random Forest model are `max_features=1.0`,`n_estimators=190`, `max_depth=5`, and `criterion='absolute_error'`.We achieved a MAPE of 0.128, which means the relative error between predictions and actual values is approximately 12.8%.
- The model is still good enough to predicting insurance costs; however, it would be advisable to impose both maximum and minimum limits on the predicted outcomes. In the event that the outcomes exceed these thresholds, it should be adjusted to conform to the established limits. This approach ensures control over the predictions, allowing them to align with predefined boundaries while benefiting from the model's overall effectiveness.

Here are some recommendations for the insurance company:
- The insurance company can adjust insurance premiums based on the health condition and age of health insurance policyholders. If the policyholder is above 20 years old, a smoker, and have an abnormal BMI, the company may consider charging a higher premium. This adjustment is typically made to reflect the increased health risks associated with smoking and abnormal BMI, especially in older individuals. It's important for the insurance company to consider these factors in pricing to ensure actuarial fairness and accurate risk assessment.
- The insurance company can also collaborate with healthcare facilities to help analyze the health of insurance policyholders. So, the insurance company can acquire more data for analysis. This can help the company provide insurance billing policies that are more accurate and aligned with the policyholders' health.
- The company should establish rules for health insurance policyholders to undergo health check-ups every three months, allowing the company to monitor their health conditions and adjust insurance billing accordingly. This is done to encourage insurance policyholders to be mindful of their health, rather than becoming indifferent simply because they are covered by insurance.
- The company can also offer more affordable premium rates if insurance policyholders choose to enroll their family members, such as their children, in the insurance plan. This is done to attract policyholders and encourage them to include their family members in the coverage.
