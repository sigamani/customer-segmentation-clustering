# The Kraken

It has been summoned...

The Kraken is a one-stop shop for feature selection, clustering, inference, and metric production
for questionniare data. It relies heavily on Aphrodite to clean and pre-process the data. 

## Getting Started

### Installation 

The Kraken requires anaconda and R (at least 4.2). For local installations please use the 
`environment.yml` file, invoked by the following command:

```  conda env create -f environment.yml  ```

``` conda activate kraken ```

To install R, go to https://cloud.r-project.org/ and select the correct operating system. 

Once this is installed, run `R < requirements.R`

In production or the AWS environment, the `buildspec.yml` is run, which creates the correct environment.

To test the code locally, please navigate to `file_for_local_dev.py` and step through that code.

#### Installation via Windows

If you have a Windows machine and working via PyCharm. You may encounter issues with the package `rpy2`. 
When importing the package, if you encounter 'error 0x7e', you may need to implement the following:

1. In Windows: 'Edit the system environment variables' > 'Advanced' > 'Environment Variables' > 'Path' > 'Edit' > Add 'C:\Program Files\R\R-4.2.2\bin\x64' as a new line.
2. In PyCharm: 'Edit Configurations' > 'Environment Variables' > set the following:
   - R_HOME: C:\Program Files\R\R-4.2.2
   - R_LIB_USER: C:\Program Files\R\R-4.2.2\bin\x64
   - R_USER: C:\Users\{user}\anaconda3\envs\kraken_2\Lib\site-packages\rpy2

Reference:
https://stackoverflow.com/questions/48440228/loadlibrary-failure-with-rpy2


v0.5.2

Change Log:

v0.6
Message Reach Metric
Variability metric refactor
Removed Agglomorative Clustering (never any decent segmentations)

v0.5.2
HOTFIX Checkbox answers missing
HOTFIX limit n of sig features returned
Pass core features back via Chi2 outputs ('selected' column)

v0.5.1
Use essential columns for naming clusterings

v0.5
Amended for preprocessed data
Added Chi2 Signal loss
Amended for new data ETL
Updated test cases
Bug fixes


v0.4
Added Laplacian Score feature selection
Integration into AWS pipeline
Updated test cases
Hierarchical clustering
Bug fixes

v0.3.1 
Added main.py which allows the kraken to be called from external processes
Outputs from main.py save to s3

v0.3
Added Signal Loss metric
Added Baysian Mixture Modelling segmentation method
Added Rules Based segmentation
Added timer to DBScan for large dataset problems
Added feature selection skeleton code, laplacian score, multicluster score, and Chi square tester
Amended social presence to calculate per cluster rather than overall


v0.2
Added Magnitude, Variability, Significant Variables, Spread, and Social Presence Metrics
Refactored tests to use S3 data
Refactored code to reflect S3 stringification of numeric values
Moved LCA metrics from R to python
Changed gower matrix to use unencoded data

v0.1.1
Added inference pipeline
Added Uniqueness and Communicability metrics
Added local conda env set up
Moved to use S3 data ingestion

