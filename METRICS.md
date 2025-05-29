# Uncertainty Estimation and Model Evaluation Metrics

This README provides definitions, formulas, and use cases for various uncertainty and evaluation metrics used in classification models, especially in ensemble and uncertainty-aware learning.

---

## **1. Model Performance Metrics**
These metrics evaluate the overall predictive performance of the model, including accuracy and ranking-based classification performance.


### **Accuracy (`cls/Acc`)**
Measures the fraction of correctly classified samples.

$$\text{Accuracy} = \frac{\text{Correct Predictions}}{\text{Total Samples}}$$

#### **Intuition:**
- **Higher accuracy** means better classification performance.
- **Does not measure uncertainty or reliability** of predictions.

### **Area Under the Precision-Recall Curve (`AUPR`)**
AUPR is the area under the **precision-recall curve**, which plots **precision (positive predictive value)** against **recall (true positive rate)** across different classification thresholds.

Mathematically, it is computed as:

$$ \text{AUPR} = \int_0^1 \, 	\text{Precision}(r) \, d	\text{Recall} $$

where:
- $\text{Precision} = \frac{TP}{TP + FP}$
- $\text{Recall} = \frac{TP}{TP + FN}$

**Interpretation:** Higher AUPR is better, indicating that the model maintains good precision and recall across thresholds.

### **Area Under the Receiver Operating Characteristic Curve (`AUROC`)**
AUROC measures the model’s ability to distinguish between positive and negative classes across all classification thresholds.

It is computed as:

$$ \text{AUROC} = \int_0^1 \, 	\text{TPR}(f) \, d	\text{FPR} $$

where:
- $\text{True Positive Rate (TPR)} = \frac{TP}{TP + FN}$ (a.k.a $\text{Recall}$)
- $\text{False Positive Rate (FPR)} = \frac{FP}{FP + TN}$

**Intuition:** Higher AUROC is better, meaning the model is better at ranking positive samples above negative samples.

### **False Positive Rate at 95% True Positive Rate (`FPR95`)**
Measures the false positive rate (FPR) when the true positive rate (TPR) is fixed at 95%.

Mathematically, it is defined as:

$$ \text{FPR@95TPR} = \min \left\{ \text{FPR} \, | \, \text{TPR} = 0.95 \right\} $$


**Intuition:** 
- Lower FPR95 is better, meaning the model produces fewer false positives at a high recall threshold.
- This metric is particularly useful for OOD evalutation.

---

## **2. Calibration Metrics**
These metrics measure how well the model's predicted probabilities align with true correctness likelihood.

### **Brier Score (`cls/Brier`)**
Measures the mean squared difference between predicted probabilities and actual labels.

$$ \text{Brier Score} = \frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} (p_{i,c} - y_{i,c})^2 $$

where:
- $p_{i,c}$ is the predicted probability for class $$ c $$.
- $y_{i,c}$ is the true one-hot encoded label.

#### **Intuition:**
- **Lower scores** indicate better-calibrated probabilities.
- A model that always assigns 100% confidence to the correct class has a Brier score of **0**.

### **Negative Log-Likelihood (NLL) (`cls/NLL`)**
Measures how well the predicted probability distribution aligns with the true class labels.

$$ \text{NLL} = - \sum_{i=1}^{N} \log p_{i,y_i} $$

#### **Intuition:**
- **Lower NLL** means better-calibrated confidence.
- Penalizes incorrect but confident predictions more heavily.

### **Expected Calibration Error (ECE) (`cal/ECE`)**
Measures the discrepancy between predicted confidence and actual accuracy.

$$ \text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right| $$

where:
- $B_m$ are confidence bins.
- $\text{acc}(B_m)$ is the empirical accuracy in bin $B_m$.
- $\text{conf}(B_m)$ is the average confidence of predictions in bin $B_m$.

**Interpretation:** Lower ECE means the model’s confidence better aligns with reality.

---

## **3. Selective Classification Metrics**
These metrics measure how well a model can balance coverage and risk when choosing to make predictions selectively.

### **Area Under the Risk-Coverage Curve (AURC) (`sc/AURC`)**
Measures how well a model balances classification risk and coverage.

$$ \text{AURC} = \int_{0}^{1} r(c) \, dc $$

where:
- $c$ (coverage) is the fraction of retained samples.
- $r(c)$ (risk) is the classification error among retained samples.

#### **Risk Definition:**
$$ r(i) = 1 - \text{Accuracy}(i) $$

#### **Coverage Definition:**
$$ c(i) = \frac{\text{Number of retained samples at threshold } i}{\text{Total number of samples}} $$

#### **Intuition:**
- **Lower AURC is better** (less risk at high coverage).
- A good model maintains **low risk across high coverage levels**.

### **Example: Binary Classification with Selective Prediction**
Imagine we have a **binary classification** model (e.g., cat vs. dog) that predicts class probabilities for 10 test samples:

| Sample | True Label | Model Prediction (Cat) | Confidence |
|--------|-----------|----------------------|------------|
| 1      | Cat       | 0.95                 | 95%        |
| 2      | Dog       | 0.92                 | 92%        |
| 3      | Dog       | 0.89                 | 89%        |
| 4      | Cat       | 0.87                 | 87%        |
| 5      | Dog       | 0.80                 | 80%        |
| 6      | Cat       | 0.78                 | 78%        |
| 7      | Dog       | 0.72                 | 72%        |
| 8      | Cat       | 0.60                 | 60%        |
| 9      | Dog       | 0.55                 | 55%        |
| 10     | Cat       | 0.45                 | 45%        |

Now, we define a **confidence threshold** for selective classification. If the model is uncertain (confidence below the threshold), it **rejects** making a prediction.

#### **Coverage at Different Confidence Thresholds**

| Confidence Threshold | Retained Samples | Coverage |
|----------------------|-----------------|------------|
| 90%                 | 2 (Samples 1, 2) | 0.2 |
| 80%                 | 5 (Samples 1-5)  | 0.5 |
| 70%                 | 7 (Samples 1-7)  | 0.7 |
| 50%                 | 9 (Samples 1-9)  | 0.9 |
| 0%                  | 10 (All samples) | 1.0 |

#### **Risk at Different Coverage Levels**

| Confidence Threshold | Retained Samples | Incorrect Predictions | Risk |
|----------------------|-----------------|----------------------|------------|
| 90%                 | 2 (Samples 1, 2) | 0 errors | 0.0 |
| 80%                 | 5 (Samples 1-5)  | 1 error (Sample 2) | 0.2 |
| 70%                 | 7 (Samples 1-7)  | 2 errors (Samples 2, 3) | 0.286 |
| 50%                 | 9 (Samples 1-9)  | 3 errors (Samples 2, 3, 9) | 0.333 |
| 0%                  | 10 (All samples) | 4 errors (Samples 2, 3, 9, 10) | 0.4 |

At **low coverage (0.2)**, the model only makes **high-confidence** predictions, so it has **low risk (0.0)**.
At **higher coverage (0.9)**, the model is forced to predict on less confident samples, so the risk increases.

### **Coverage at 5% Risk (`sc/Cov@5Risk`)**
Measures the fraction of samples retained at a **fixed 5% error rate**.
**Interpetation:** Higher coverage is better since more samples are retained at low risk.

### **Risk at 80% Coverage (`sc/Risk@80Cov`)**
Measures the error rate when retaining **80% of samples**.
**Interpretation:** Lower risk is better at high coverage.

---

## **4. Ensemble-Based Uncertainty Metrics**
These metrics quantify uncertainty in ensemble models by measuring disagreement and confidence dispersion.

### **Ensemble Disagreement (`ens/Disagreement`)**
Measures the disagreement among ensemble members by calculating how often different members predict different classes for the same input.

For each sample $i$, the disagreement is calculated as:

$$ D_i = 1 - \frac{\sum_{c=1}^{C} {n_{i,c} \choose 2}}{{N \choose 2}} $$

where:
- $N$ is the number of ensemble members (estimators)
- $n_{i,c}$ is the number of estimators that predicted class $c$ for sample $i$
- ${n_{i,c} \choose 2} = \frac{n_{i,c}(n_{i,c}-1)}{2}$ is the number of pairs of estimators that agree on class $c$
- ${N \choose 2} = \frac{N(N-1)}{2}$ is the total number of possible pairs of estimators

#### **Interpretation** 
- **High disagreement** means ensemble members predict different classes (indicating high uncertainty)
- **Low disagreement** means ensemble members agree on their predictions (indicating low uncertainty)
- A value of 0 means perfect agreement (all estimators predict the same class)
- A value of 1 means complete disagreement (estimators are maximally divided in their predictions)

### **Ensemble Entropy (Predictive Entropy) (`ens/Entropy`)**
Measures the entropy of the average predictive distribution of the ensemble.

$$ H(y|x) = -\sum_{c} p(y = c) \log p(y = c) $$

where:
- $p(y = c) = \frac{1}{M} \sum_{i=1}^{M} p_i(y = c)$ is the mean ensemble prediction.

#### **Intepretation** 
- **High entropy** means that the ensembles mean output has high entropy (ensemble more uncertain)
- **Low entropy** means that the ensembles mean output has lower entropy (ensemble more confident)


### **Ensemble Mutual Information (Epistemic Uncertainty) (`ens/MI`)**
Mutual information measures epistemic uncertainty, which arises due to a lack of knowledge in the model, typically caused by limited training data. It quantifies the disagreement among ensemble members beyond what is expected from aleatoric (data) uncertainty.

Mathematically, it is defined as:
$$ \text{Mutual Information} = H(y|x) - \frac{1}{M} \sum_{i=1}^{M} H(y|x, \theta_i) $$

where:
- $H(y|x)$ is the **predictive entropy** (a.k.a. ensemble entropy, see above).
- $H(y|x, \theta_i)$ is the entropy of each ensemble member’s prediction.

#### **Intuition:**
- The predictive entropy $H(y|x)$ can be thought of as the total uncertainty.
- Average Individual Model Entropy $\frac{1}{M} \sum_{i=1}^{M} H(y|x, \theta_i)$ measures **only aleatoric uncertainty**: Each model in the ensemble makes a prediction. If all models in the ensemble are confident but predict different classes, the predictive entropy is high, but the individual model entropies are low.

#### **Intepretation** 
- **High MI** corresponds to higher ensemble disargreement and epistemic uncertainty.
- **Low MI** corresponds to lower ensemble disargreement and epistemic uncertainty.
