
# **DCPR** Code Repository

This is a repository to deposit the code and data for **DCPR** model. 
---

The folders in the DCPR repository:

- **Datasets**: 

  a. **Synthetic datasets**: Eighteen synthetic datasets differ in sampling frequency, circadian gene content, and noise conditions.

  b. **Real datasets**:
  
     1) Six GEO datasets with diffrent Species, tissues and Sequencing Platforms.
  
     2) Thirteen Alzheimer's disease datasets with different brain regions.

- **DCPR_codes**: Source code for DCPR model, and main.py is the main entrance of the DCPR framework.

- **Visualization**: Scripts for data visualization, including Cumulative distribution functions (CDFs) for model performance evaluation, Gene expression fitting curves, and Bar plots.

- **Supplementary Files**: The detailed results for all the analysis in our study.

- **SOTA**: Three representative models used in the comparative study:

  ​	Cyclum: https://github.com/KChen-lab/Cyclum.git

  ​	CHIRAL: https://github.com/naef-lab/CHIRAL.git

    (The code for CYCLOPS included in Cyclum/cyclum/models)

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



## 3. Run

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


