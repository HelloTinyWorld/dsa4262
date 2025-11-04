## Cloning the github repository and setting up ubuntu connection



Follow these steps to get the project up and running on the Ronin server:

1. Go to https://ronin.nus.cloud/ and start your machine
### Important: Before connecting, please increase the Ronin server disk space to at least 100GB and upgrade the machine to have at least 4GB of RAM to ensure smooth operation.

Open your terminal and connect via SSH using your Ubuntu user credentials. For example:

```bash
ssh -i /path/to/your/private_key.pem ubuntu@<ronin-server-address>
```

Replace `/path/to/your/private_key.pem` with your SSH private key path, and `<ronin-server-address>` with the actual server address.


2. Clone this GitHub repository

Open terminal and set working directory to where you want the project to be in and run:
```bash
git clone https://github.com/HelloTinyWorld/team1_TeamProject_dsa4262.git
cd team1_TeamProject_dsa4262/
pip install -r requirements.txt

```

## Task 1: 


1. Download dataset0.json.gz and data.info.labelled from https://canvas.nus.edu.sg/courses/78679/files/folder/TeamProject_Data/dataset_0

### Important: Ensure dataset0.json.gz and data.info.labelled are in your downloads folder, dataset0.json will work fine too. 

2. Open **another** terminal shell and set your working directory to your downloads folder, which should look like this
```bash
cd ~/Downloads
```

3. Run the following command on this new terminal shell, replacing <path_to_key> with the correct path to your .pem SSH key file:
### Important: rename <path_to_key>/dsa4262-2510-team1-YOURNAME.pem to the path to your .pem file, the code here is a placeholder, same for ubuntu@dsa4262-2510-team1-YOURNAME.nus.cloud:/home/ubuntu/dsa4262/task1/
```bash
cd ~/Downloads
scp -i "<path_to_key>/dsa4262-2510-team1-YOURNAME.pem" \
  dataset0.json.gz \
  data.info.labelled \
  ubuntu@dsa4262-2510-team1-YOURNAME.nus.cloud:/home/ubuntu/team1_TeamProject_dsa4262/task1/
```
4. Now on your **original** terminal shell in which you connected to the cloud server, run this command

```bash
python3 task1.py
```


## Task 2: Setup and Usage Instructions

2. Download RNA modification data from AWS

Create the data directory and sync files from the AWS S3 bucket:

```bash
mkdir -p ~/rna_modification_data
cd ~/rna_modification_data

aws s3 sync --no-sign-request \
    s3://sg-nex-data/data/processed_data/m6Anet/ ./
```

This downloads all RNA modification data into `~/rna_modification_data`.

3. Run this command to get task2 predictions
```bash
cd team1_TeamProject_dsa4262/
python3 task2.py

```


5. The final folder structure will look like this

The folder structure will look like this after running the data:

```bash
|-- .gitignore
|-- full_xgb_model.pkl
|-- requirements.txt
|-- task1.py
|-- task2.py
|-- predict_data.py
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

1. Other datasets with the same format as in the sgnex website may be used, just download it into the rna_modification_data folder and run task2.py

2. alternatively, you can also run dataset3.json.gz using the following instructions
   
ssh -i "/path/to/your/key.pem" ubuntu@your-ronin-host "mkdir -p /home/ubuntu/team1_TeamProject_dsa4262/task1/"

scp -i "/path/to/your/key.pem" ~/Downloads/dataset3.json.gz \
    ubuntu@your-ronin-host:/home/ubuntu/team1_TeamProject_dsa4262/task1/

then run predict_data.py



 

