<img src="figures/dataset-cover.png">



<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

Table of Content:


- [1. Executive Summary](#1-executive-summary)
- [2. Introduction](#2-introduction)
- [3. Analysis Pathway:](#3-analysis-pathway)
- [4. Exploratory Data Analysis](#4-exploratory-data-analysis)
- [5. Walk Through the Project](#5-walk-through-the-project)
  - [5.1. Dataset](#51-dataset)
  - [5.2. Data Preprocessing](#52-data-preprocessing)
  - [5.3. Modeling](#53-modeling)
  - [5.4. Evaluation](#54-evaluation)
  - [5.5. Topic Results & Interpretation](#55-topic-results--interpretation)
- [6. Next Step](#6-next-step)
- [7.Project Structure](#7project-structure)
- [8.Reference](#8reference)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->




## 1. Executive Summary



**Project Goal:**

The goal of this project is to reveal the topics from the massive amount of medical papers, and build a program that allows users to search an article title, then returns the most relevant papers' information (along with their confidence score) from the dataset. 

**Project Results:**

Using LDA model, I discovered 15 topics and the top 7 keywords for each topic, as below:

|      topic1 |    topic2 |      topic3 |     topic4 |    topic5 |        topic6 |       topic7 |      topic8 |     topic9 |      topic10 |   topic11 |     topic12 |   topic13 |   topic14 |      topic15 |
| ----------: | --------: | ----------: | ---------: | --------: | ------------: | -----------: | ----------: | ---------: | -----------: | --------: | ----------: | --------: | --------: | -----------: |
|         cov |  patients |       virus |      cells |       pcr |        health |          ace |         air |      covid | participants |   protein |         new | treatment |     model | transmission |
|        sars |     covid |     viruses |       cell |   samples |         covid | inflammatory |        high | population |      anxiety |   binding |    diseases |  clinical |      data |        masks |
|    sars_cov |  hospital |   influenza | expression |  positive |      pandemic |           il | temperature |       risk |        women |  proteins | development |   studies |  analysis |         mask |
| coronavirus | mortality |     vaccine |         cd |   testing |          care |         lung |        time |   pandemic |     symptoms |     spike |        many |   patient |    models |     exposure |
|   infection |      risk |       viral |     immune | detection |        public |       immune |      method |      among |       stress | antiviral |       human |  patients |      time |         face |
|     disease |   disease |  infections |       host |      test |    healthcare |      disease |    compared |  countries |         self |  activity |   potential |  evidence | different |    bacterial |
| respiratory |  clinical | respiratory |      viral |        rt | public_health |        blood |     methods |   measures |   associated |      drug |    research |    cancer |  learning |         hand |

**Recommendation Function:**

When a user search a paper title (eg. "Logistics of community smallpox control through contact tracing and ring vaccination: a stochastic network model"), the program will output the top N related papers to the user's query, and score them based on the confidence score (prop_topic):

| prop_topic |                                             title |                                          abstract | publish_time |                                           authors |                                               url |
| ---------: | ------------------------------------------------: | ------------------------------------------------: | -----------: | ------------------------------------------------: | ------------------------------------------------: |
|   0.867478 |     Equitable d-degenerate Choosability of Graphs | let formula see text class d-degenerate graphs... |   2020-04-30 | Drgas-Burchardt, Ewa; Furmańczyk, Hanna; Sidor... | https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7... |
|   0.826024 | Transition Property for [Formula: see text]-Po... | 1985 restivo salemi presented list five proble... |   2020-05-26 |                                  Rukavicka, Josef | https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7... |
|   0.815862 |       Edge-Disjoint Branchings in Temporal Graphs | temporal digraph formula see text triple formu... |   2020-04-30 | Campos, Victor; Lopes, Raul; Marino, Andrea; S... | https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7... |
|   0.815116 | Some asymptotic properties of kernel regressio... | present consider nonparametric regression mode... |   2020-08-17 |                    Bouzebda, Salim; Didi, Sultana | https://doi.org/10.1007/s13163-020-00368-6; ht... |
|   0.810232 | Tuning the overlap and the cross-layer correla... | properties potential overlap networks formula ... |   2018-03-09 |                       Juher, David; Saldaña, Joan | https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7... |



## 2. Introduction

In response to the COVID-19 pandemic, the White House and a coalition of leading research groups have prepared the COVID-19 Open Research Dataset (CORD-19). CORD-19 is a resource of over 300,000 scholarly articles (by 2021 April), about COVID-19, SARS-CoV-2, and related coronaviruses. This freely available dataset is provided to the global research community to apply recent advances in natural language processing and other AI techniques to generate new insights in support of the ongoing fight against this infectious disease. There is a growing urgency for these approaches because of the rapid acceleration in new coronavirus literature, making it difficult for the medical research community to keep up.





## 3. Analysis Pathway:

1. EDA & Data Preprocessing
2. Modeling approach 1: TF-IDF + PCA to create features -> K-means to find clusters -> Most frequent words on each cluster to reveal topics
3. Modeling approach 2: Topic Model directly used on the whole dataset, topic model experimented:
   1. Latent Dirichlet Allocation (LDA)
   2. (HDP)
4. Performance Evaluation
5. Presenting results



## 4. Exploratory Data Analysis

Link for EDA: https://georgehua.github.io/covid19-research-paper-topic-modeling/EDA

Key Takeaways:

- The dataset contains duplicates and missing abstract, need to be cleaned first
- In 2020, the medical paper around the pandemic increase exponentially (more than 250,000) compare to 2019 (less than 25,000)

- Dataset total entries: 536,817
- After remove duplicates and missing abstract paper: 344,711
- English papers kept for further modeling: **338,442**



<img src="figures/wordcloud.png">

From the word cloud generated above, we start to see some levels of research directions in the papers:

- Covid, cov, coronavirus, sar
- Virus, infection, cell, protein, treatment (virus study)
- Mortality, death (death rate study)
- Respiratory, age, lung, blood, protein, risk (risk factor study and impact analysis)



## 5. Walk Through the Project



### 5.1. Dataset

```
kaggle datasets download -d allen-institute-for-ai/CORD-19-research-challenge -f metadata.csv

unzip metadata.csv.zip -d data/
```

Instruction about how to use Kaggle API: https://www.kaggle.com/docs/api

Or you can download the dataset manually from: https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge?select=metadata.csv, then store metadata.csv into data/ folder.

```
Data columns (total 19 columns):
 #   Column            Non-Null Count   Dtype  
---  ------            --------------   -----  
 0   cord_uid          536817 non-null  object 
 1   sha               180066 non-null  object 
 2   source_x          536817 non-null  object 
 3   title             536570 non-null  object 
 4   doi               295050 non-null  object 
 5   pmcid             191123 non-null  object 
 6   pubmed_id         254917 non-null  object 
 7   license           536817 non-null  object 
 8   abstract          390379 non-null  object 
 9   publish_time      536598 non-null  object 
 10  authors           522082 non-null  object 
 11  journal           501591 non-null  object 
 12  mag_id            0 non-null       float64
 13  who_covidence_id  222762 non-null  object 
 14  arxiv_id          6996 non-null    object 
 15  pdf_json_files    180066 non-null  object 
 16  pmc_json_files    147047 non-null  object 
 17  url               315683 non-null  object 
 18  s2_id             488029 non-null  float64
```



### 5.2. Data Preprocessing

```
python src/preproc.py -i <INPUT_FILE_NAME> -dir <DATA_DIRECTORY>

# arg -i and -dir has default values, you can just run
python src/preproc.py
```

The program perform the following preprocessing tasks for the dataset:

Data Cleaning:

- Drop NA
- Drop duplicates
- Drop all non-English research paper
- Remove HTML tags & replace HTML character codes with ASCII equivalent
- Remove URLs, new line and line breaks characters and punctuations
- Replace extra white spaces with one space

Features:

- Lemmatization
- Tokenize each paper's abstract text into a list of words
- Add Bi-gram and Tri-gram to the list of words
- Defined and remove stop words and the words with only 2 letters or less
- Save tokenized list of words into `data/docs.npy` and `data/df_cleaned.csv` for topic modeling



### 5.3. Modeling



**Approach 1: K-Means:** (Notebook Link: https://georgehua.github.io/covid19-research-paper-topic-modeling/K-Means)

- Turn each document instance into a feature vector using Term Frequency–inverse Document Frequency (TF-IDF).
- Apply Dimensionality Reduction to each feature vector using t-Distributed Stochastic Neighbor Embedding (t-SNE) to cluster similar research articles in the two dimensional plane embedding.
- Use Principal Component Analysis (PCA) to project down the dimensions to a number of dimensions that will keep .95 variance while removing noise and outliers in embedding.
- Apply k-means clustering, where k is 17, to label each cluster on.
- Find the most frequent words in the cluster as the topic of the cluster.
- Investigate the clusters visually on the plot, zooming down to specific articles as needed, and via classification using Stochastic Gradient Descent (SGD). 



By applying K-means, we use elbow method to estimate the optimal number of clusters.  we can see in the plot above that the decline of the sum of squared errors (or distortion) becomes considerably less after k=17. Since the turning point is at k = 17, we will use 17 as the number of clusters for the KMeans model below.

<img src="/figures/elbow.png"></img>

The K-means generate most frequent words for each cluster

```
Cluster 1
acute, virus, severe, syndrome, respiratory, infection, covid, coronavirus, sars, cov

Cluster 2
admission, icu, clinical, risk, severe, disease, hospital, mortality, covid, patients

Cluster 3
cell, cells, host, replication, binding, virus, viral, rna, proteins, protein

Cluster 4
practice, services, patient, medical, health, healthcare, patients, pandemic, covid, care

Cluster 5
spread, model, crisis, distancing, measures, economic, countries, pandemic, covid, social

Cluster 6
impact, india, health, march, measures, air, pandemic, period, covid, lockdown

Cluster 7
resection, performed, technique, operative, complications, postoperative, surgical, patients, laparoscopic, surgery

Cluster 8
covid, protease, activity, inhibitors, cov, sars, antiviral, compounds, drugs, drug

Cluster 9
method, disease, models, analysis, different, new, time, research, data, model

Cluster 10
sensitivity, test, positive, testing, detection, sars, cov, samples, rt, pcr

Cluster 11
trial, stroke, therapy, risk, trials, patient, studies, clinical, treatment, patients

Cluster 12
case, pandemic, acute, infection, respiratory, severe, coronavirus, disease, patients, covid

Cluster 13
adults, disease, parents, age, infection, years, pediatric, respiratory, covid, children

Cluster 14
pathogens, detected, infection, human, infections, viral, respiratory, viruses, virus, influenza

Cluster 15
inflammatory, ifn, induced, infection, il, mice, immune, expression, cell, cells

Cluster 16
university, pandemic, covid, medical, student, teaching, online, education, learning, students

Cluster 17
care, stress, depression, psychological, pandemic, anxiety, public, mental, covid, health
```

____________________

**Approach 2 Topic Models Only:**

(Notebook Link for LDA:) https://georgehua.github.io/covid19-research-paper-topic-modeling/LDA

(Notebook Link for HDP:) https://georgehua.github.io/covid19-research-paper-topic-modeling/HDP

- Apply topic models on the whole document sets
- Evaluate with Coherence score
- Visualize topics
- Build a function that research dataset with user query (paper title), then returns most related papers based on confidence score.



**Short Introduction and comparison of the 2 topic models: (LDA, HDP)**

The LDA model is guided by two principles:

- Each document is a mixture of topics. In a 3 topic model we could assert that a document is 70% about topic A, 30 about topic B, and 0% about topic C.
- Every topic is a mixture of words. A topic is considered a probabilistic distribution over multiple words.

<img src="/figures/LDA_usage.jpg"></img>

HDP is an extension of LDA, designed to address the case where the number of mixture components (the number of "topics" in document-modeling terms) is not known a priori. For HDP (applied to document modeling), one also uses a Dirichlet process to capture the uncertainty in the number of topics. So a common base distribution is selected which represents the countably-infinite set of possible topics for the corpus, and then the finite distribution of topics for each document is sampled from this base distribution.

As far as pros and cons, HDP has the advantage that the maximum number of topics can be unbounded and learned from the data rather than specified in advance. Though it is more complicated to implement, and unnecessary in the case where a bounded number of topics is acceptable.





### 5.4. Evaluation



**K-means cluster verify with SGD classifier: accuracy: 0.90**

K-means is fast to run and provide a "hard cluster" among the dataset, but our goal is to create a navigation system that allows the users to search and find similar articles. K-means, in this case, cannot fulfill the mission. 



Since topic Modeling is unsupervised, accuracy score is not  applicable for evaluating the model. Instead, we look at the coherence  score, which is an statistical measure of the topic model performance. A topic has a higher score of coherence if the words defining a topic  have a high probability of co-occurring cross documents.

**LDA Coherence Score: 0.58686**

**HDP Coherence Score: 0.3932**

- Topic Coherence measures score a single topic by measuring the degree of semantic similarity between high scoring words in the topic.
- C_v measure is based on a sliding window, one-set segmentation of the top words and an indirect confirmation measure that uses normalized pointwise mutual information (NPMI) and the cosine similarity
- LDA outperforms in this instance





### 5.5. Topic Results & Interpretation



LDA Topic Viz with pyLDAvis:

<img src="figures/LDA_topic_viz.png">



LDA top 7 keywords for each topic:

|      topic1 |    topic2 |      topic3 |     topic4 |    topic5 |        topic6 |       topic7 |      topic8 |     topic9 |      topic10 |   topic11 |     topic12 |   topic13 |   topic14 |      topic15 |
| ----------: | --------: | ----------: | ---------: | --------: | ------------: | -----------: | ----------: | ---------: | -----------: | --------: | ----------: | --------: | --------: | -----------: |
|         cov |  patients |       virus |      cells |       pcr |        health |          ace |         air |      covid | participants |   protein |         new | treatment |     model | transmission |
|        sars |     covid |     viruses |       cell |   samples |         covid | inflammatory |        high | population |      anxiety |   binding |    diseases |  clinical |      data |        masks |
|    sars_cov |  hospital |   influenza | expression |  positive |      pandemic |           il | temperature |       risk |        women |  proteins | development |   studies |  analysis |         mask |
| coronavirus | mortality |     vaccine |         cd |   testing |          care |         lung |        time |   pandemic |     symptoms |     spike |        many |   patient |    models |     exposure |
|   infection |      risk |       viral |     immune | detection |        public |       immune |      method |      among |       stress | antiviral |       human |  patients |      time |         face |
|     disease |   disease |  infections |       host |      test |    healthcare |      disease |    compared |  countries |         self |  activity |   potential |  evidence | different |    bacterial |
| respiratory |  clinical | respiratory |      viral |        rt | public_health |        blood |     methods |   measures |   associated |      drug |    research |    cancer |  learning |         hand |



Example output for searching the title: "Logistics of community smallpox control through contact tracing and ring vaccination: a stochastic network model"

System output:

Topic 4 key words (rank in order): model, data, analysis, models, time, different, learning, information, show, abstract

|                                             title |                                          abstract | publish_time |                                           authors |                                               url | prop_topic |
| ------------------------------------------------: | ------------------------------------------------: | -----------: | ------------------------------------------------: | ------------------------------------------------: | ---------: |
|     Equitable d-degenerate Choosability of Graphs | let formula see text class d-degenerate graphs... |   2020-04-30 | Drgas-Burchardt, Ewa; Furmańczyk, Hanna; Sidor... | https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7... |   0.867478 |
| Transition Property for [Formula: see text]-Po... | 1985 restivo salemi presented list five proble... |   2020-05-26 |                                  Rukavicka, Josef | https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7... |   0.826024 |
|       Edge-Disjoint Branchings in Temporal Graphs | temporal digraph formula see text triple formu... |   2020-04-30 | Campos, Victor; Lopes, Raul; Marino, Andrea; S... | https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7... |   0.815862 |
| Some asymptotic properties of kernel regressio... | present consider nonparametric regression mode... |   2020-08-17 |                    Bouzebda, Salim; Didi, Sultana | https://doi.org/10.1007/s13163-020-00368-6; ht... |   0.815116 |
| Tuning the overlap and the cross-layer correla... | properties potential overlap networks formula ... |   2018-03-09 |                       Juher, David; Saldaña, Joan | https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7... |   0.810232 |






## 6. Next Step

- Build a graphical interface to allow better user experience to interact with the program

- Implement a search engine to allow user search keywords instead of the whole paper title



## 7.Project Structure



    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── metadata.csv   <- Data source
    │   ├── df_cleaned.csv <- Preprocessed data file
    │
    ├── docs               <- Github Pages documents
    |
    ├── figures            <- Markdown figures
    │
    │
    ├── notebooks          <- Jupyter notebooks for EDA and experiments
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   │
    │   ├── preproc.py     <- Script for preprocessing data



## 8.Reference

[Shashank Kapadia, Evaluate Topic Models: Latent Dirichlet Allocation (LDA)](https://towardsdatascience.com/evaluate-topic-model-in-python-latent-dirichlet-allocation-lda-7d57484bb5d0)

[Kaggle, COVID-19 Open Research Dataset Challenge (CORD-19), n AI challenge with AI2, CZI, MSR, Georgetown, NIH & The White House](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge/code?datasetId=551982&sortBy=voteCount)

[Selva Prabhakaran, Topic modeling visualization – How to present the results of LDA models?](https://www.machinelearningplus.com/nlp/topic-modeling-visualization-how-to-present-results-lda-models/)

