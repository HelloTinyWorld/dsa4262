## Cloning the github repository
1. Clone this GitHub repository

Run:
```bash
git clone https://github.com/HelloTinyWorld/dsa4262.git
cd dsa4262
```

1.1
Please install all necessary packages using the code below
```bash
pip install -r requirements.txt
```
1.2 Note that the project uses jupyter notebooks. For instructions on how to use jupyter notebooks refer to this guide by dataquest: https://www.dataquest.io/blog/jupyter-notebook-tutorial/


## Task 1: 

1. Ensure all packages in requirements.txt have been installed and create the empty folder called task1
```bash
mkdir task1
```
2. Next download dataset0.json.gz and data.info.labelled from https://canvas.nus.edu.sg/courses/78679/files/folder/TeamProject_Data/dataset_0
and move them into the folder task1
3. Open task1_data_cleaning.ipynb and click **RUN ALL**
4. Open task1_model_training.ipynb and click **RUN ALL**


## Task 2: Setup and Usage Instructions

Follow these steps to get the project up and running on the Ronin server:

1. Connect to the Ronin server

Open your terminal and connect via SSH using your Ubuntu user credentials. For example:

```bash
ssh -i /path/to/your/private_key.pem ubuntu@<ronin-server-address>
```

Replace `/path/to/your/private_key.pem` with your SSH private key path, and `<ronin-server-address>` with the actual server address.

Before connecting, please increase the Ronin server disk space to at least 100GB and upgrade the machine to have at least 4GB of RAM to ensure smooth operation.




2. Download RNA modification data from AWS

Create the data directory and sync files from the AWS S3 bucket:

```bash
mkdir -p ~/rna_modification_data
cd ~/rna_modification_data

aws s3 sync --no-sign-request \
    s3://sg-nex-data/data/processed_data/m6Anet/ ./
```



This downloads all RNA modification data into `~/rna_modification_data`.

3. Run the data parsing notebook

From the project directory, start Jupyter Notebook, open `task2_data_parsing.ipynb`, and click **Run All**.

4. Run the main analysis notebook

Open `task2.ipynb` and click **Run All** to perform the full analysis and generate predictions.


5. The final folder structure will look like this

The folder structure will look like this after running the data:

```bash
|-- .gitignore
|-- full_xgb_model.pkl
|-- requirements.txt
|-- task1_data_cleaning.ipynb
|-- task1_model_training.ipynb
|-- task2_data_parsing.ipynb
|-- task2.ipynb
|-- task1/
|    |-- files required for task1
|-- predictions/
|    |-- .gitkeep
|    |-- (multiple prediction CSV files)
|-- prepared_data/
|    |-- .gitkeep
|    |-- (multiple CSV files)
|-- rna_modification_data/
|    |-- .gitkeep
|    |-- (multiple cell line subfolders)
```

## Using the model for other datasets

1. Other datasets with the same format as in the sgnex website may be used, just download it into the rna_modification_data folder (create the folder if it does not exist yet)
 

