# Double-Listing project 

The following internship project is done with the Gemeente as a Master's thesis for Data Science at UvA. This project aims to find double listings in an Amsterdam Short Stay data set. An unsupervised bootstrapping of active learning for ER is used as the approach to tackle this project.

# Project folder structure
1. Data --> All the data which do not contain sensitive information can be found here.
2. Code --> All the codes used for this project can be found here.

# Usage 
In the data folder use the jupyter notebooks to follow the pipeline of the project in this order:

1. EDA.ipynb            --> This notebook cleans, transforms, and merges the raw data sets provided by the Gemeente. The specifics functions created can be found in eda_fn.py
2. LaBSE.ipynb          --> Use the data frame which the clean, merged, and reduced (df_merged). This notebook transforms the descriptions of the listings into vectors representing their meaning (this notebook was run on google collab with GPU) 
3. sim_embeddings.ipynb --> Use the data frame which the clean, merged, and reduced (df_red). This notebook creates a similarity vector needed for classification, for this notebook uses a different version of python to use the pre-trained model Hotels 50k (https://github.com/GWUvision/Hotels-50K). Use python version 3.6.5 and TensorFlow 1.13.1. The file similarity_fn.py contains all the functions necessary to run the notebook.
4. Unsupervised         --> Use the data frame which the similarity vectors (df_sim_img_merged_clean). This notebook creates the unsupervised labels needed for the bootstrapping of the model. The file thresholding.py has the necessary functions for running the notebook.
5. active_learning.ipynb--> Use the complete similarity vector data frame (df_unslabel_img_red_hot). This notebook runs the active learning system, the file active_learn.py contains the necessary functions for running the models.

Extras:
1. graphs.ipynb   --> This notebook uses the saved results of the active learning runs, for instance: results_full_rf.plk, to graph the results.
2. labelling.ipynb--> This notebook is used to label the data used as the test set.


# Contact

Victor Salazar, victor12lup@gmail.com

