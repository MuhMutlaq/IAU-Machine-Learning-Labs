# Lab11: K-Means Clustering (Credit Card Customer Segmentation)

This project applies K-Means clustering to segment credit card customers based on their usage behavior. By grouping similar customers, we can discover hidden patterns in purchasing and borrowing habits, which can be used to design targeted marketing strategies.

## Final Questions & Answers

**1. Why is this an unsupervised learning problem?**
This is an unsupervised learning problem because the dataset does not contain pre-defined target labels or categories (e.g., "Fraud" or "Not Fraud"). The goal is to explore the data and find natural, hidden groupings among the customers without prior knowledge of what those groups should look like.

**2. Why did we remove the `CUST_ID` column?**
The `CUST_ID` column is purely a unique identifier and contains no behavioral information about the customer. Leaving it in the dataset would negatively distort the distance calculations used by the K-Means algorithm.

**3. Which columns had missing values?**
The `CREDIT_LIMIT` and `MINIMUM_PAYMENTS` columns contained missing values.

**4. How did you handle the missing values?**
The missing values were handled using mean imputation, which involves filling the empty cells with the average value of their respective columns.

**5. Why is scaling important before applying K-Means?**
K-Means clustering relies on distance metrics (like Euclidean distance) to group data points. If the data is not scaled, features with massive numerical ranges (like a `BALANCE` of 5,000) will completely dominate the distance calculations over features with tiny ranges (like a `PURCHASES_FREQUENCY` between 0 and 1). Scaling ensures all features contribute equally to the model.

**6. Which K value did you choose? Explain your answer using the elbow method and silhouette score.**
**K = 3** was chosen. When observing the Elbow curve, the sharp decrease in inertia begins to slow down and flatten out around K=3 and K=4. More importantly, the Silhouette Score peaks strongly at K=3, indicating that three clusters result in the most mathematically distinct and well-separated groupings for this dataset.

**7. Based on the cluster summary table, describe each customer segment in your own words.**

* **Cluster 0 (Cash Advance Reliant):** These customers carry higher balances and frequently rely on taking out large cash advances, but they make relatively few standard purchases.
* **Cluster 1 (High-Value Spenders):** These customers have the highest credit limits and use their cards heavily for purchases. They maintain high balances but primarily use the card for transactions rather than cash advances.
* **Cluster 2 (Frugal/Conservative Users):** This is the largest segment. These customers have low balances, low purchase amounts, and low cash advance usage, indicating they rarely use their credit cards heavily.

**8. Which cluster may represent high-value customers?**
**Cluster 1**, as these customers make the highest average amount of purchases and have the highest credit limits, generating the most transaction revenue.

**9. Which cluster may represent customers who rely more on cash advance?**
**Cluster 0**, as their average cash advance amount is significantly higher than the other clusters.

**10. How can a company use these clusters for marketing strategy?**

* **For Cluster 1 (High Spenders):** The company could offer premium cashback programs, airline miles, or loyalty rewards to encourage even higher transaction volumes and retain their business.
* **For Cluster 0 (Cash Advance Users):** The company could promote lower-interest personal loans or promotional balance transfer rates as a better alternative to expensive cash advance fees.
* **For Cluster 2 (Frugal Users):** The company could launch introductory campaigns or milestone rewards (e.g., "Spend $500 this month, get a $50 bonus") to incentivize them to use their cards more frequently.

---

## What I Learned from this Project

Completing this project provided hands-on experience in building an end-to-end unsupervised machine learning pipeline. Key takeaways include:

* **The Critical Role of Data Preprocessing:** I learned firsthand how sensitive distance-based algorithms like K-Means are to unscaled data and missing values. Implementing `StandardScaler` and mean imputation were crucial steps in preparing the dataset for accurate modeling.
* **Objective Model Evaluation:** Rather than guessing the number of customer segments, I learned how to analytically determine the optimal number of clusters (K) by combining the visual intuition of the **Elbow Method** with the mathematical validation of the **Silhouette Score**.
* **Translating Math into Business Value:** Beyond writing the code, this project demonstrated how to translate mathematical cluster centroids into actionable, real-world business insights. Grouping raw data is only useful if you can interpret the behaviors of those groups (e.g., identifying "High-Value Spenders" vs. "Frugal Users") to drive targeted marketing strategies.
* **Dimensionality Reduction for Visualization:** I gained experience using **Principal Component Analysis (PCA)** to compress high-dimensional data into two components, allowing for clear, 2D visual representation of complex cluster distributions.
