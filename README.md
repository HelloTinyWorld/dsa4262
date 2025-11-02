## Task 1: 




## Task 2: Setup and Usage Instructions

Follow these steps to get the project up and running on the Ronin server:

1. Connect to the Ronin server

Open your terminal and connect via SSH using your Ubuntu user credentials. For example:

```bash
ssh -i /path/to/your/private_key.pem ubuntu@<ronin-server-address>
```

Replace `/path/to/your/private_key.pem` with your SSH private key path, and `<ronin-server-address>` with the actual server address.

Before connecting, please increase the Ronin server disk space to at least 100GB and upgrade the machine to have at least 4GB of RAM to ensure smooth operation.

2. Clone this GitHub repository

Once logged in, run:
```bash
git clone https://github.com/HelloTinyWorld/dsa4262.git
cd dsa4262
```

The folder structure will look like this after running the data:

```bash
|-- .gitignore
|-- data_parsing.ipynb
|-- full_xgb_model.pkl
|-- task2.ipynb
|-- requirements.txt
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

Note that the folders will not exist yet and will be created in the subsequent steps

2.1 Note that the project uses jupyter notebooks. For instructions on how to use jupyter notebooks refer to this guide by dataquest: https://www.dataquest.io/blog/jupyter-notebook-tutorial/



3. Download RNA modification data from AWS

Create the data directory and sync files from the AWS S3 bucket:

```bash
mkdir -p ~/rna_modification_data
cd ~/rna_modification_data

aws s3 sync --no-sign-request \
    s3://sg-nex-data/data/processed_data/m6Anet/ ./
```



This downloads all RNA modification data into `~/rna_modification_data`.

4. Run the data parsing notebook

From the project directory, start Jupyter Notebook, open `task2_data_parsing.ipynb`, and click **Run All**.

5. Run the main analysis notebook

Open `task2.ipynb` and click **Run All** to perform the full analysis and generate predictions.


