
# **DCPR** Code Repository

This is a repository to deposit the code and data for **DCPR** model. 
---

The folders in the DCPR repository:

- **Datasets**: 

  a. **Synthetic datasets**: Eighteen synthetic datasets differ in sampling frequency, circadian gene content, and noise conditions.

  b. **Real datasets**: Six datasets from Gene Expression Omnibus with diffrent Species, tissues and Sequencing Platforms and thirteen Alzheimer's disease datasets with different brain regions.

- **DCPR_codes**: Main code file for the DCPR model.

- **Visualization**: The directory contains all graphical outputs generated during analysis, including Cumulative distribution functions (CDFs) for model performance evaluation, Gene expression fitting curves, and bar plots.

- **Supplementary Files**: The detailed results for all the analysis in our study.

- **SOTA**: Comparative methods used in the contrast experiments:

  ​	CYCLOPS/Cyclum: https://github.com/KChen-lab/Cyclum.git

  ​	CHIRAL: https://github.com/naef-lab/CHIRAL.git

---



### **Step-by-step Running:**

## 1. Environment Installation

It is recommended to use the conda environment (python 3.8), mainly installing the following dependencies:

```
conda env create -f environment.yaml  
conda activate tf1            
```
See environment.yaml for details.



## 2. Datasets

Download the datasets from the folder Datasets.



## 3. run

To run the DCPR model, a gene expression matrix is required. Optionally, a time series corresponding to the samples and a list of seed genes can also be provided.

--Command line:

```
python run.py 
```


## 4.  Output

All model outputs are saved in the results_final folder with the following structure:

--Prediction results (saved as predresults.csv)

--Evaluation Metrics (saved as evaluate.csv)

--Two .tiff files are Cumulative distribution functions (CDFs) and Timing comparison plot 



## 5.  Installation

git clone https://github.com/NJAU-CDSIC/DCPR_V1.0.git


