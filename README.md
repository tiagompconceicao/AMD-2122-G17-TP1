## Directory organization:
 /dataset -> directory with the datasets
 /scripts -> directory with the sql source code and the database scripts
 /src -> directory with the python source code
 /orange_workflows -> directory with orange workflow files

### Steps to run a solution:
- Open the command prompt in the directory "src"
- Execute the command "python XMain.py"

Important note: 'X' is to be replaced with "ID3", "NB" or "1R", each script will run a solution with a different classifier.

> ID3Main.py: uses ID3 classifier  
> NBMain.py : uses Naive-Bayes classifier  
> 1RMain.py : uses One-Rule classifier  


### Steps to run the script that evaluate the 3 classifiers:

- Open the command prompt in the directory "src"
- Execute the command "python EvaluationMain.py"

### Steps to run the script that applies the One-Rule classifier in the Mushroom dataset:
- Open the command prompt in the directory "src"
- Execute the command "python 1RMushroomDataset.py"
- The output file (oneR_OUTPUT.txt) is present in the root directory of the project

