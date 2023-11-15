# Multimodal Sentiment Analysis
COMP/DATA/CSEC 5703 Group-Based Capstone Project. 

## Getting Started

This project aims to create three multimodality sentiment analysis models and test them on three multimodal datasets with texts and images using Python.

### Prerequisites

Python 3.x version

Check requirement.txt for all the packages that needs to be installed.

Please open a issue if there is any problem.

### Installing

Clone the whole github respository

```
git clone https://github.com/MaySsssss/Capstone-5703.git
```

Install all the required packages using

```
pip install -r requirements.txt
```

Download the following three datasets from the websites and put the zip files into the folder Capstone-5703/CS15_2_virtual/DATASET

```
-Memotion
https://www.kaggle.com/datasets/williamscott701/memotion-dataset-7k
Rename the zip file as memotion_dataset_7k.zip

-MVSA-single
https://www.kaggle.com/datasets/vincemarcs/mvsasingle
Rename the zip file as MVSA_single.zip

-MVSA-multi
https://www.kaggle.com/datasets/vincemarcs/mvsamultiple
Rename the zip file as MVSA.zip
```

## Running the tests

#### Linux/MacOS:

Using termial to navigate to the respository folder and run the following codes

```
make run_script clean
```

#### Windows:

Physically unzip three datasets to folder Capstone-5703/CS15_2_virtual/DATASET:

If using terminal 
```
tar -xf memotion_dataset_7k.zip
tar -xf MVSA.zip
tar -xf MVSA_single.zip
```

If using zip tool, please select **'Extract Here'**


Then run
```
python script.py
```

## Results

After the program is done, there will be files logging the results in the `output` folder. Please send this output folder back to developer team. 

## Authors

* **May Li** - *Code Integration* - (https://github.com/MaySsssss)
* **Kehao Chen** - *Model establishment* - (https://github.com/beholder91)
* **Taoxu Zhao** - *Fine tuning and testing* - 
* **Yufan Lin** - *Version control* - 
* **Liye Wang** - *File management* - 
* **Xucheng Zhou** - *Team management* - (https://github.com/aden1350)
