<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Thanks again! Now go create something AMAZING! :D
-->

<!-- PROJECT LOGO -->
<br />
  <h3 align="center">Continuous-time Representation learning on Temporal Heterogeneous Graph</h3>

<!-- TABLE OF CONTENTS -->
<!--<details open="open">-->
<!--  <summary>Table of Contents</summary>-->
<!--  <ol>-->
<!--    <li>-->
<!--      <a href="#about-the-project">About The Project</a>-->
<!--    </li>-->
<!--    <li>-->
<!--      <a href="#project-structure">Project Structure</a>-->
<!--    </li>-->
<!--    <li><a href="#usage">Usage</a></li>-->
<!--    <li><a href="#data">Used datasey</a></li>-->
<!--  </ol>-->
<!--</details>-->



<!-- ABOUT THE PROJECT -->
## About The Project

This repo focuses on continues-time Dynamic Heterogeneous Graph Embedding. We focus on developing an inductive representation learning method on temporal HIN. 

<!-- Project Structure -->
## Project Structure
```bash 
├── README.md                                 Read me file 
├── Dataset                                   Jupyter Notebooks for data processing
├── loaders.py                                Data loader for training and testing, including neighbor sampling methods on temporal graph
├── model.py                                  The model structure. 
├── modules.py                                Submodules
├── utils.py                                  early stop function etc.   
├── eval_metric.py                            Evaluation metrics   
├── main.py                                   The main model training and testing script.    
├── train_test_split.py                       Data split.     
└── .gitignore                                gitignore file
```

<!-- USAGE -->
## Usage
``` shell 
python main.py --data acm --prefix xx --training True --time_comb mul --dynamic h_decay --pop_node both --pop_pred null --bs 1024 --edge_event
```
Find the details about the arguements in main.py 

<!-- DATA -->
## Data
ACM and DBLP: https://www.aminer.cn/citation\#b541

IMDB: https://www.imdb.com/interfaces/
