#!/usr/bin/env python
# coding: utf-8

# # Electric Vehical Data Analysis Project
# ## By Devendra Vijaykumar Jaiswal

# #### In this project, we will analyze a dataset related to electric vehicles (EVs). The dataset containsvarious features such as electric range, energy consumption, price, and other relevantattributes. our goal is to conduct a thorough analysis to uncover meaningful insights, tell acompelling story, conduct hypothesis testing and provide actionable recommendations based on the data.

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import warnings
warnings.filterwarnings ('ignore')


# In[7]:


df=pd.read_csv("D:\\data science\\FEV-data - Auta elektryczne.csv")
df.head()


# In[8]:


df.info()


# ## Data Understanding

# #### Car full name: The full name or designation of the vehicle, often combining make, model, and variant.
# 
# #### Make: The brand or manufacturer of the car.
# 
# #### Model: The specific model or version of the car.
# 
# #### Minimal price (gross) [PLN]: The minimum retail price of the car, in Polish złoty (PLN).
# 
# #### Engine power [KM]: The car's engine power, measured in horsepower (KM in Polish).
# 
# #### Maximum torque [Nm]: The peak torque the engine can produce, measured in Newton-meters(Nm).
# 
# #### Type of brakes: The braking system used, such as disc or drum brakes.
# 
# #### Drive type: The drivetrain configuration, like FWD (front-wheel drive), RWD (rear-wheel drive),or AWD (all-wheel drive).
# 
# #### Battery capacity [kWh]: Total energy capacity of the car’s battery, measured in kilowatt-hours(kWh).
# 
# #### Range (WLTP) [km]: Estimated driving range on a full charge under WLTP standards, inkilometers.
# 
# #### Wheelbase [cm]: The distance between the front and rear axles, in centimeters.
# 
# ####  Length [cm]: The overall length of the car, in centimeters.
# 
# #### Width [cm]: The car’s width, in centimeters.
# 
# ####  Height [cm]: The car’s height, in centimeters.
# 
# ####  Minimal empty weight [kg]: The car’s minimum weight when empty, measured in kilograms.
# 
# ####  Permissible gross weight [kg]: Maximum legally allowed weight, including passengers andcargo, in kilograms.
# 
# #### Maximum load capacity [kg]: The maximum weight the car can carry, in kilograms.
# 
# ####  Number of seats: The number of passenger seats in the car.
# 
# ####  Number of doors: The number of doors on the car.
# 
# ####  Tire size [in]: The tire size, measured in inches.
# 
# ####  Maximum speed [kph]: The top speed of the car, in kilometers per hour.
# 
# #### Boot capacity (VDA) [l]: Trunk or cargo space capacity, measured in liters according to VDA standards.
# 
# ####  Acceleration 0-100 kph [s]: Time taken to accelerate from 0 to 100 kilometers per hour, in seconds.
# 
# #### Maximum DC charging power [kW]: The highest charging power supported when using a DC fast charger, in kilowatts (kW).
# 
# #### Mean - Energy consumption [kWh/100 km]: Average energy consumption per 100 kilometers, in kilowatt-hours (kWh).

# In[10]:


# Lets check the column names present in the dataset
df.columns


# Task 1: A customer has a budget of 350,000 PLN and wants an EV with a minimum range
# of 400 km.
# 
# a) Your task is to filter out EVs that meet these criteria.
# 
# b) Group them by the manufacturer (Make).
# 
# c) Calculate the average battery capacity for each manufacturer.

# In[11]:


# Filter the dataset based on criteria
filtered_evs = df[(df['Minimal price (gross) [PLN]'] <= 350000) & (df['Range (WLTP) [km]'] >= 400)]

# Display the filtered results
filtered_evs.head()


# In[12]:


# Group by 'Make' and calculate the average battery capacity
average_battery_capacity = filtered_evs.groupby('Make')['Battery capacity [kWh]'].mean().reset_index()

# Rename the columns for better readability
average_battery_capacity.columns = ['Make', 'Average Battery Capacity (kWh)']

# Display the results
average_battery_capacity


# In[19]:


# Sort by average battery capacity
average_battery_capacity = average_battery_capacity.sort_values(
    by='Average Battery Capacity (kWh)', 
    ascending=False
)

# Export to CSV
average_battery_capacity.to_csv("Filtered_EVs_Avg_Battery.csv", index=False)


# #### Task 2: You suspect some EVs have unusually high or low energy consumption. Find the  outliers in the mean - Energy consumption [kWh/100 km] column.

# In[23]:


#Calculate Q1 (25th percentile) and Q3 (75th percentile)
Q1 = df['mean - Energy consumption [kWh/100 km]'].quantile(0.25)
Q3 = df['mean - Energy consumption [kWh/100 km]'].quantile(0.75)

# Calculate IQR
IQR = Q3 - Q1

# Define the lower and upper bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Find the outliers
outliers = df[(df['mean - Energy consumption [kWh/100 km]'] < lower_bound) |
              (df['mean - Energy consumption [kWh/100 km]'] > upper_bound)]

# Display the outliers
outliers


# In[26]:


from scipy.stats import zscore

# Calculate Z-scores for the 'mean - Energy consumption [kWh/100 km]' column
df['Z_Score'] = zscore(df['mean - Energy consumption [kWh/100 km]'])

# Find outliers where Z-score > 3 or < -3
outliers_zscore = df[(df['Z_Score'] > 3) | (df['Z_Score'] < -3)]

# Display the outliers
outliers_zscore


# ### Task 3: Your manager wants to know if there's a strong relationship between battery capacity and range.
# 
# #### a) Create a suitable plot to visualize.
# 
# #### b) Highlight any insights.
# 

# In[27]:


import matplotlib.pyplot as plt
import seaborn as sns

# Scatter plot to visualize the relationship between battery capacity and range
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Battery capacity [kWh]', y='Range (WLTP) [km]', data=df)

# Add title and labels
plt.title('Relationship Between Battery Capacity and Range of EVs', fontsize=16)
plt.xlabel('Battery Capacity (kWh)', fontsize=12)
plt.ylabel('Range (WLTP) [km]', fontsize=12)

# Display plot
plt.show()


# In[73]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Create a regression plot
plt.figure(figsize=(10, 6))
sns.regplot(x='Battery capacity [kWh]', y='Range (WLTP) [km]', data=df)
plt.title('Relationship between Battery Capacity and Range')
plt.show()


# ### a) Plot Interpretation: 
# ### Positive Correlation: the scatter plot shows an upward trend, it suggests that as the battery capacity increases, the range also tends to increase, which is expected for EVs.

# In[28]:


# Calculate Pearson correlation coefficient
correlation = df['Battery capacity [kWh]'].corr(df['Range (WLTP) [km]'])
print(f"Pearson correlation coefficient: {correlation}")


# ### b) Insights Based on Correlation:
# 
# #### Strong Positive Correlation: the correlation coefficient is close to 1,  there is a strong positive relationship between battery capacity and range.
# 

# ### Task 4: Build an EV recommendation class. The class should allow users to input theirbudget, desired range, and battery capacity. The class should then return the top three EVsmatching their criteria.

# In[72]:


import pandas as pd

class EVRecommender:
    def __init__(self, data_file):
        self.data = pd.read_csv("D:\\data science\\FEV-data - Auta elektryczne.csv")

    def recommend_evs(self, budget, desired_range, battery_capacity):
        # Filter EVs based on user input
        filtered_evs = self.data[
            (self.data['Minimal price (gross) [PLN]'] <= budget) &
            (self.data['Range (WLTP) [km]'] >= desired_range) &
            (self.data['Battery capacity [kWh]'] >= battery_capacity)
        ]

        # Sort filtered EVs by price
        sorted_evs = filtered_evs.sort_values(by='Minimal price (gross) [PLN]')

        # Return top three EVs or a message if no EVs match the criteria
        if sorted_evs.empty:
            return "No electric vehicles match the specified criteria."
        else:
            return sorted_evs.head(3)


# Create an instance of the EVRecommender class
recommender = EVRecommender('FEV-data - Auta elektryczne.csv')

budget = 9000000
desired_range = 500
battery_capacity = 100

# Now you can use the recommender variable
recommended_evs = recommender.recommend_evs(budget, desired_range, battery_capacity)

print("Recommended EVs:")
print(recommended_evs)


# ### Task 5: Inferential Statistics – Hypothesis Testing: Test whether there is a significant difference in the average Engine power [KM] of vehicles manufactured by two leading manufacturers i.e. Tesla and Audi. What insights can you draw from the test results? Recommendations and Conclusion: Provide actionable insights based on your analysis. (Conduct a two sample t-test using ttest_ind from scipy.stats module)

# In[75]:


import pandas as pd
from scipy.stats import ttest_ind

# Load the data
data = pd.read_csv("D:\\data science\\FEV-data - Auta elektryczne.csv")

# Filter the data for Tesla and Audi vehicles
tesla_data = data[data['Make'] == 'Tesla']
audi_data = data[data['Make'] == 'Audi']

# Perform a two-sample t-test
t_stat, p_value = ttest_ind(tesla_data['Engine power [KM]'], audi_data['Engine power [KM]'])

print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}")

# Interpret the results
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: There is a significant difference in the average engine power of Tesla and Audi vehicles.")
else:
    print("Fail to reject the null hypothesis: There is no significant difference in the average engine power of Tesla and Audi vehicles.")

# Calculate the mean engine power for Tesla and Audi vehicles
tesla_mean_engine_power = tesla_data['Engine power [KM]'].mean()
audi_mean_engine_power = audi_data['Engine power [KM]'].mean()

print(f"Mean engine power for Tesla vehicles: {tesla_mean_engine_power:.2f} KM")
print(f"Mean engine power for Audi vehicles: {audi_mean_engine_power:.2f} KM")


# ### Based on the test results, I can draw the following insights:
# 
# 
# #### - If the p-value is less than the significance level (alpha),  reject the null hypothesis and conclude that there is a significant difference in the average engine power of Tesla and Audi vehicles.
# 
# #### - If the p-value is greater than or equal to the significance level (alpha), fail to reject the null hypothesis and conclude that there is no significant difference in the average engine power of Tesla and Audi vehicles.
# 
# 
# ### Actionable insights and recommendations:
# 
# 
# #### - If there is a significant difference in the average engine power of Tesla and Audi vehicles, manufacturers can focus on optimizing engine power to improve vehicle performance and competitiveness.
# 
# #### - Vehicle buyers can consider engine power as a key factor when comparing Tesla and Audi vehicles.
# 
# #### - Manufacturers can also explore opportunities to improve engine efficiency and reduce emissions while maintaining or improving engine power.

# In[ ]:




