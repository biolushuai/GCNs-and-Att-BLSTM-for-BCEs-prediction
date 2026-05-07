# A Structure-Based B-cell Epitope Prediction Model Through Combing Local and Global Features
This repository is the implementation of  paper: "A Structure-Based B-cell Epitope Prediction Model Through Combing Local and Global Features", DOI:10.3389/fimmu.2022.890943
# Abstract
B-cell epitopes (BCEs) are a set of specific sites on the surface of an antigen that binds to an antibody produced by B-cell. The recognition of BCEs is a major challenge for drug design and vaccines development. Compared with experimental methods, computational approaches have strong potential for BCEs prediction at much lower cost. Moreover, most of the currently methods focus on using local information around target residue without taking the global information of the whole antigen sequence into consideration. We propose a novel deep leaning method through combing local features and global features for BCEs prediction. In our model, two parallel modules are built to extract local and global features from the antigen separately. For local features, we use Graph Convolutional Networks (GCNs) to capture information of spatial neighbors of a target residue. For global features, Attention-Based Bidirectional Long Short-Term Memory (AttBLSTM) networks are applied to extract information from the whole antigen sequence. Then the local and global features are combined to predict BCEs. The experiments show that the proposed method achieves superior performance over the state-of-the-art BCEs prediction methods on benchmark datasets. Also, we compare the performance differences between data with or without global features. The experimental results show that global features play an important role in BCEs prediction. Our detailed case study on the BCEs prediction for SARS-Cov-2 receptor binding domain confirms that our method is effective for predicting and clustering true BCEs.

## 1. Requirement
* Python = 3.9.10  
* Pytorch = 1.10.2  
* Scikit-learn = 1.0.2

## 2. Citation
Shuai Lu, Yuguang Li, Qiang Ma, Xiaofei Nan*, Shoutao Zhang*. A structure-based B-cell epitope prediction model through combing local and global features[J]. Frontiers in Immunology, 2022, 13: 890943. DOI: 10.3389/fimmu.2022.890943.


## 3. Contact
For questions and comments, feel free to contact : ieslu@zzu.edu.cn.
