# Chapter 1 Solutions: Introduction to Machine Learning

### Exercise 1.1. Problem Formulation
**Question:** Formulate the hospital readmission prediction task using Mitchell's definition. Discuss why accuracy is a poor choice of $P$.

**Solution:**
*   **Task $T$:** Predicting whether a discharged patient will be readmitted to the hospital within 30 days.
*   **Experience $E$:** A historical dataset of electronic health records, including patient demographics, treatment details, and a binary label indicating if they were readmitted.
*   **Performance Measure $P$:** Accuracy is a poor choice because hospital readmissions are typically imbalanced (e.g., only 5% of patients might be readmitted). A model that always predicts "will not be readmitted" would achieve 95.0% accuracy but is clinically useless. Better alternatives for $P$ include **AUC-ROC**, **Recall** (to minimize costly false negatives/missed interventions), or the **F1-score**.

---

### Exercise 1.2. Supervised vs Unsupervised
**Question:** Classify the following paradigms.

**Solution:**
*   **(a) AlphaGo playing itself:** **Reinforcement Learning**. The agent learns by interacting with the Go environment, optimizing for a delayed scalar reward (win or loss) rather than learning from labeled expert moves.
*   **(b) Grouping 10 million news articles:** **Unsupervised Learning**. Specifically, clustering. There are no pre-existing topic labels provided to the algorithm.
*   **(c) Predicting electricity demand:** **Supervised Learning**. Specifically, regression. The historical consumption patterns serve as the features, and the actual recorded demand serves as the continuous target label.
*   **(d) Training BERT by masking:** **Self-Supervised Learning**. The supervision signal (the masked words) is algorithmically derived directly from the inherent structure of the unlabelled data.
*   **(e) Recommending movies:** **Supervised Learning** (or Semi-Supervised/Collaborative Filtering). The algorithm uses known historical ratings as labels to predict the missing ratings.

---

### Exercise 1.3. NFL Theorem
**Question:** Respond to the claim: "Deep neural networks are the best algorithm; we should always use them."

**Solution:**
*   **(a)** The **No Free Lunch (NFL) Theorem** states that averaged over all possible data-generating distributions, every algorithm has the same expected performance. Therefore, no universally "best" algorithm exists. Deep neural networks perform exceptionally well on natural data (images, text) because their inductive biases (hierarchical representations, spatial locality) align well with the structure of the physical world, but they will fail on datasets where those assumptions do not hold.
*   **(b)** A linear model provably outperforms a deep neural network in settings where $n \ll d$ (small sample size, high dimensionality). In this regime, the high capacity of a neural network leads to severe overfitting (high variance), whereas the strict inductive bias of a linear model (high bias, low variance) prevents catastrophic generalization gaps.

---

### Exercise 1.6. The Metric Trap
**Question:** A credit card fraud detection system has a 0.01% positive rate. A model predicts "not fraud" for everything.

**Solution:**
*   **(a)** 
    *   **Precision:** $0.0$ (There are zero True Positives).
    *   **Recall:** $0.0$ (The model caught zero actual fraud cases).
    *   **F1-score:** $0.0$.
*   **(b)** Let $Cost(FN) = 500.0$ and $Cost(FP) = 10.0$. The optimal decision threshold $\tau^*$ is determined by the ratio of these costs:
    $$\tau^* = \frac{Cost(FP)}{Cost(FP) + Cost(FN)} = \frac{10.0}{10.0 + 500.0} \approx 0.0196$$
    Because the cost of a False Negative is so high, the model should flag a transaction as fraud even if its predicted probability is just $\approx 2.0\%$.

---

### Exercise 1.7. Data Leakage
**Question:** Standardising data before splitting into train/test.

**Solution:**
*   **(a)** Yes, this is data leakage. Standardisation requires computing the global mean $\mu$ and standard deviation $\sigma$. If computed before the split, information about the test set's distribution (its mean and variance) "leaks" into the training features.
*   **(c)** Standardisation must be done by computing $\mu$ and $\sigma$ strictly on the training set, and then applying those exact same parameters to transform the validation and test sets.
