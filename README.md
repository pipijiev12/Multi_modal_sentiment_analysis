"# Multimodal_Sentiment_Analysis" 
Contributor: FengXianjie

"#Dataset download"


Dropbox Link: https://www.dropbox.com/s/7z56hf9szw4f8m8/cmumosi_cmumosei_iemocap.zip?dl=0

Containing CMUMOSI, CMUMOSEI and IEMOCAP datasets

Each dataset has the CMU-SDK version and Multimodal-Transformer version (with different input dimensionalities)

"#Model training"

1.Set up the configurations in config/run.ini

2.python run.py -config config/run.ini

"#Research"

1. In the original project, a new network QRSAN was proposed


2. Data sparsity analysis was carried out for the original data set and the model was verified experimentally for data sparsity


3. Perform ablation experiment analysis and hyperparameter analysis on the proposed model
