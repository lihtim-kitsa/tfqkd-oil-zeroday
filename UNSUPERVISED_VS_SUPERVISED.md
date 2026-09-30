# Comparison: Unsupervised Clustering vs Supervised Classification

This document provides a detailed comparison between **Unsupervised Clustering** (e.g., Gaussian Mixture Models, K-Means) and **Supervised Classification** (e.g., XGBoost, Support Vector Machines) specifically within the context of the TF-QKD continuous-wave attack dataset.

## 1. Core Differences

| Feature | Unsupervised Clustering (GMM / K-Means) | Supervised Classification (XGBoost / SVM) |
| :--- | :--- | :--- |
| **Objective** | Group data into $k$ clusters based purely on geometric distance or density without prior knowledge. | Learn decision boundaries that explicitly separate known labeled classes from a training set. |
| **Labels Required?** | **No.** Discovers patterns blind to the true attack type. | **Yes.** Requires a labeled training set of Nominal, FIM, and TWIRL data. |
| **Zero-Day Detection** | Can naturally group unseen anomalies into outlier clusters (if they are geometrically distant from normal data). | Fails completely on unseen attacks unless paired with a dedicated Anomaly Detector (like Deep SVDD). |
| **Best Data Geometry** | Performs well when classes form distinct, isolated "blobs" in the feature space. | Performs well even if classes overlap significantly, provided there are distinct orthogonal splits or complex non-linear boundaries. |

## 2. Performance on the TF-QKD Dataset

In this specific physical simulation dataset, the physical features extracted from the Lang-Kobayashi equations (like Phase Decoherence $\Delta\phi$ and Spectral Sideband Power $P_{sb}$) cause the Nominal and Attack classes to **heavily overlap** in a non-convex manner. 

Because of this complex geometry, the performance gap between the two approaches is massive:

*   **Unsupervised Proxy Accuracy (GMM):** ~36.68%
*   **Supervised Accuracy (XGBoost):** ~88.22%

### Why does Unsupervised fail here?
Algorithms like Gaussian Mixture Models and K-Means attempt to draw spheres or multi-dimensional ellipses around dense regions of data. Because the Nominal state and the early stages of FIM/TWIRL attacks bleed into each other continuously in the feature space, the density algorithms are unable to separate them and instead group them all into one giant continuous block.

### Why does Supervised succeed here?
XGBoost relies on orthogonal decision trees. It is capable of making highly specific, hard splits in the parameter space (e.g., "If $P_{sb} > X$ AND $\Delta\phi < Y$, then it is a TWIRL attack"). This allows it to slice through overlapping continuous data in ways that purely geometric clustering cannot.

## 3. Conclusion

*   **To classify known attacks optimally:** You must use **Supervised Classification** (e.g., XGBoost). The feature overlap in the continuous-wave side-channel space is too severe for blind clustering algorithms to succeed. 
*   **To detect entirely new, unseen attacks (Zero-Days):** Instead of using standard $k$-means clustering, the optimal approach is using a generative One-Class classifier (like the **Deep SVDD** implemented in this repository), trained purely on Nominal data to act as a tight boundary around safe operational limits.
