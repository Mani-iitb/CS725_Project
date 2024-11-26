ANALYZING DIFFERENT MODELS FOR INTRUSION DETECTION SYSTEMS

Problem Statement
Network intrusion detection mechanisms detect and optionally classify malicious behavior within a networked system.
These range from simple NMAP snooping and DDOS to complex(but obsolete) rootkit attacks. 
Most of these attacks are deterministic procedures and have easy to identify hallmarks, and therefore can be prevented since detection is easy. 
Classification is often challenging due to the large overlap between attacks that are along similar lines or attacks that subsume other attacks. 

----------------------------------------------------------------------------------
Model                                                     | Accuracy | Precision | 
----------------------------------------------------------------------------------
Decision Tree (our own DT code)                           | 0.997    | 0.789     |
DT using Adaboost (SKlearn)                               | 0.997    | 0.805     | 
Random Forest (our own RF code)                           | 0.997    | 0.830     | 
balanced RF (imblearn library)                            | 0.766    | 0.754     | 
Logistic Regression (own code)                            | 0.952    | 0.508     | 
LR using Boosting (SKlearn)                               | 0.938    | 0.472     | 
SVM (our own code)                                        | 0.333    | 0.436     | 
SVM with kernel trick (SKlearn)                           | 0.996    | 0.846     | 
NN (Adam with dropout)                                    | 0.954    |           | 
ADAM with dropout, with data along 2 Principal Components | 0.930    |           |
ADAM with dropout with data along 10 Principal Components | 0.996    |           |
----------------------------------------------------------|----------|-----------|

Important features for DT
![image](https://github.com/user-attachments/assets/23e97877-ea2a-4ef0-b143-ff85cc67be9a)

Sensitivity of the features for DT
![image](https://github.com/user-attachments/assets/a9f5a191-6f20-45cc-a709-53c1daa96c6c)
