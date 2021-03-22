# sound-anomaly-detection
This is a work in progress.


## Table of Contents

- [Introduction](#introduction)

- [Installation](#installation)

- [Data sources](#data-sources)

- [Instructions](#instructions)

- [Architecture](#architecture)

- [Next steps](#next-steps)

---

## Introduction
### Description
This is a python project for *Codit Belgium*.
_ADD_MORE_DESCRIPTION_


The program performs the following actions:
1. **Cleans** and **preprocesses the dataset** for a machine learning model
2. **Trains** a machine learning model
3. **Predicts** whether a machine will fail and classifies the failures based on sound
4. **Evaluates** the model performance
5. **Export** data insights for dashboarding.

### Objectives
- Create a machine learning model to predict when a machine will fail based on sound
- Classify the failures based on sound in order to do more targeted maintenance.

### When?
It is a 2 weeks project.
The deadline to complete it is scheduled to `02/04/2021 at 9 a.m.`.

### Visuals
![image title to add](core/assets/image.png)


## Installation
To run the program, you need:
- To install the libraries below
- To download the *MIMII* dataset (see [Data sources](#data-sources) for information).

### Install the libraries
| Library       | Used to                                        |
| ------------- | :----------------------------------------------|
| Numpy         | To handle Numpy arrays                         |
| Pandas        | To store and access info in a DataFrame        |
| librosa       | To analyse audio                               |

Follow these instructions to install the required libraries: on terminal
1. Open your terminal;
2. cd to the directory where the `requirements.txt` file is located;
3. Create and activate your virtual environment.
4. Run the command: 
```pip3 install -r requirements.txt```

## Data Sources
_TO_PARAPHRAZE_
"The MIMII Dataset is a sound dataset for malfunctioning industrial machine investigation and inspection. It contains the sounds generated from four types of industrial machines, i.e. valves, pumps, fans, and slide rails. Each type of machine includes multiple individual product models, and the data for each model contains normal and anomalous sounds. To resemble a real-life scenario, various anomalous sounds were recorded. Also, the background noise recorded in multiple real factories was mixed with the machine sounds.

The MIMII Dataset can be downloaded at: https://zenodo.org/record/3384388

Harsh Purohit, Ryo Tanabe, Kenji Ichige, Takashi Endo, Yuki Nikaido, Kaori Suefusa, and Yohei Kawaguchi, “MIMII Dataset: Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection,” arXiv preprint arXiv:1909.09347, 2019. URL: https://arxiv.org/abs/1909.09347"

## Instructions
### How to run the program
- Run `main.py` to start the program.
Or
- On your terminal:
```python3 main.py```

### Usage example
#### Example of input the user can give:
_ToDo_

#### Output of the example:
_ToDo_


## Architecture
The project is structured as follows:

```
sound-anomaly-detection
│   README.md               :explains the project
│   main.py                 :script to run in order to start the program
│   requirements.txt        :packages to install to run the program
│   .gitignore              :specifies which files to ignore when pushing to the GitHub repository
│
└───core                    :directory contains all the core scripts of the program
│   │   __init__.py
│   │
│   └───assets              :contains the datasets and images
```

### Roadmap
- [x] Download the dataset
- [X] Explore the dataset
- [X] Clean dataset: no missing values (remove or fill in missing values)
- [ ] Clean dataset: Consolidate values
- [ ] Clean dataset: Make sure data format is correct
- [ ] Clean dataset: Trim blank spaces
- [ ] Prepare dataset for machine learning model: If necessary, add new useful features based on existing features in the dataset
- [ ] Prepare dataset for machine learning model: feature selection
- [ ] Prepare dataset for machine learning model: feature engineering
- [ ] Prepare dataset for machine learning model: feature normalization
- [ ] Prepare dataset for machine learning model: feature resampling

Depending on progress in roadmap above:
- [X] Export data into `data_clean_GOOD_ENOUGH.csv`
- [ ] Export data into `data_clean_GOOD.csv`
- [ ] Export data into `data_clean_PRECISE.csv`

Finally:
- [ ] Build dashboard to display data insights
- [ ] Prepare presentation (15 minutes + 5 minutes Q&A)
- [ ] 


### Contributing
Open to contributions.
Requirements to be defined.
Instructions on how to contribute will be described in this section.


### Author(s) and acknowledgment
This project is carried out by:
- **Louan Mastrogiovanni**
- **Van Frausum Derrick** 
from Theano 2.27 promotion at BeCode.

We would like to thank:
- **Codit Belgium** for this opportuniy to work on a use-case
- Harsh Purohit, Ryo Tanabe, Kenji Ichige, Takashi Endo, Yuki Nikaido, Kaori Suefusa, and Yohei Kawaguchi for the *MIMII* Dataset
- and our colleagues and coaches at BeCode for their help and guidance.


## Next steps
- Determine requirements for contributions
- Add instructions on how to contribute
- Progress in roadmap: continue cleaning and preparing data
