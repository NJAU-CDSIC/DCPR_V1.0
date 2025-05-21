
# **DCPR** Code Repository

This is a repository to deposit the code and data for **DCPR** model. 
---

The folders in the DCPR repository:

- **Datasets**: 

  a. **Synthetic datasets**: Eighteen synthetic datasets differ in sampling frequency, circadian gene content, and noise conditions.

  b. **Real datasets**:
  
     1) Six GEO datasets with diffrent species, tissues and sequencing platforms.
  
     2) Thirteen Alzheimer's Disease(AD) datasets from different brain regions.

- **DCPR_codes**: Source code for DCPR model.
  
  - main.py: This script is the main entry point of the program.
  
  - base.py: This script is used to prepare features for the model.
  
  - preprocess.py: This script is used to prepare data.
  
  - result.py: This script is used to generates predictive outputs from the DCPR model.
- 

- **Visualization**: Scripts for data visualization, including circadian curves (cosine model fitting), sample plots (predicted and real time),  model performance (CDF curves, and accuracy plots).
  
  - calculate_auc_mederr: The function is used to get AUC and MedAE.
  
  - plot_cdf: The function is used to plot CDF curves.
  
  - plot_prediction: The function is used to plot sample plots (predicted and real time).
  
  - plot_circadian_expression: The function is used to plot circadian curves (cosine model fitting).
  
  - plot_seamless_bar_chart: The function is used to plot accuracy plots.

- **Supplementary Files**: The detailed results for all the analysis in our study.

- **SOTA**: Three representative models used in the comparative study:

  ​	Cyclum: https://github.com/KChen-lab/Cyclum.git

  ​	CHIRAL: https://github.com/naef-lab/CHIRAL.git

    CYCLOPS: https://github.com/cyclops-ui/cyclops.git

    (The python code of CYCLOPS is also provied by Cyclum.)

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

To run the DCPR model, the input parameters include expression matrix X, time stamp vector V, seed gene list S. The output O denotes the target folder for saving the all the results.

--Command line:

```
python run.py 
```


## 4.  Output

All outputs are saved in the folder results_final:

-- Predicted time vector of all input sampless (predresults.csv)

-- Evaluation Metrics: CDF and AUC values (evaluate.csv)

-- Two *.tiff files are CDF curves and sample plots (predicted and real time)



## 5.  Installation

git clone https://github.com/NJAU-CDSIC/DCPR_V1.0.git


