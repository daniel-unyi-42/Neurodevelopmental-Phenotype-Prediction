# Neurodevelopmental-Phenotype-Prediction

Official implementation of:

<b>Neurodevelopmental Phenotype Prediction: A State-of-the-Art Deep Learning Model</b>

Dániel Unyi, Bálint Gyires-Tóth

https://arxiv.org/abs/2211.08831

<b>Abstract</b>: A major challenge in medical image analysis is the automated detection of biomarkers from neuroimaging data. Traditional approaches, often based on image registration, are limited in capturing the high variability of cortical organisation across individuals. Deep learning methods have been shown to be successful in overcoming this difficulty, and some of them have even outperformed medical professionals on certain datasets. In this paper, we apply a deep neural network to analyse the cortical surface data of neonates, derived from the publicly available Developing Human Connectome Project (dHCP). Our goal is to identify neurodevelopmental biomarkers and to predict gestational age at birth based on these biomarkers. Using scans of preterm neonates acquired around the term-equivalent age, we were able to investigate the impact of preterm birth on cortical growth and maturation during late gestation. Besides reaching state-of-the-art prediction accuracy, the proposed model has much fewer parameters than the baselines, and its error stays low on both unregistered and registered cortical surfaces.

<b>Instructions to reproduce the results</b>:
1) Download the structural pipeline of the 3rd release of the dHCP dataset: https://biomedia.github.io/dHCP-release-notes/download.html#academic-torrent-download-recommended (once finished, a directory named <i>rel3_dhcp_anat_pipeline</i> should be downloaded, containing 783 subdirectories)
2) Run preprocess.ipynb
3) Run MAIN.ipynb
