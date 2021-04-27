<img src="figures/dataset-cover.png">

In response to the COVID-19 pandemic, the White House and a coalition of leading research groups have prepared the COVID-19 Open Research Dataset (CORD-19). CORD-19 is a resource of over 300,000 scholarly articles (by 2021 April), about COVID-19, SARS-CoV-2, and related coronaviruses. This freely available dataset is provided to the global research community to apply recent advances in natural language processing and other AI techniques to generate new insights in support of the ongoing fight against this infectious disease. There is a growing urgency for these approaches because of the rapid acceleration in new coronavirus literature, making it difficult for the medical research community to keep up.



**The goal of this project is to reveal the latent topics from the massive amount of papers, and build a program that allows the user to search article title, then the system can return the most relevant research papers (along with their confidence score) in the dataset.** 



**Analysis Pathway:**

1. EDA & Data Preprocessing
2. Modeling approach 1: TF-IDF + PCA to create features -> K-means to find clusters -> LDA on each cluster to reveal topics
3. Modeling approach 2: Topic Model directly used on the whole dataset, topic model experimented:
   1. Latent Dirichlet Allocation (LDA)
   2. (HDP)
   3. (NFM)
4. Performance Evaluation
5. Presenting results



## Exploratory Data Analysis

Link for EDA:

Key Takeaways:

<img src="figures/wordcloud.png" height="400px">

## Walk Through



### 1. Dataset

```
kaggle datasets download -d allen-institute-for-ai/CORD-19-research-challenge -f metadata.csv

unzip metadata.csv.zip -d data/
```

Instruction about how to use Kaggle API: https://www.kaggle.com/docs/api

Or you can download the dataset manually from: https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge?select=metadata.csv, then store metadata.csv into data/ folder.



### 2. Data Preprocessing

```
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



### 3. Modeling



**Approach 1: K-Means + LDA:** 

- Turn each document instance $d_i$ into a feature vector $X_i$ using Term Frequency–inverse Document Frequency (TF-IDF).
- Apply Dimensionality Reduction to each feature vector $X_i$ using t-Distributed Stochastic Neighbor Embedding (t-SNE) to cluster similar research articles in the two dimensional plane $X$ embedding $Y_1$.
- Use Principal Component Analysis (PCA) to project down the dimensions of $X$ to a number of dimensions that will keep .95 variance while removing noise and outliers in embedding $Y_2$.
- Apply k-means clustering on $Y_2$, where $k$ is 20, to label each cluster on $Y_1$.
- Apply Topic Modeling on $X$ using Latent Dirichlet Allocation (LDA) to discover keywords from each cluster. 
- Investigate the clusters visually on the plot, zooming down to specific articles as needed, and via classification using Stochastic Gradient Descent (SGD). 



____________________

**Approach 2 Topic Models:**

- Apply LDA on the whole document sets
- Evaluate with Coherence score
- Visualize topics
- Build a function that research dataset with user query (paper title), then returns most related papers based on confidence score.



The LDA model is guided by two principles:

- Each document is a mixture of topics. In a 3 topic model we could assert that a document is 70% about topic A, 30 about topic B, and 0% about topic C.
- Every topic is a mixture of words. A topic is considered a probabilistic distribution over multiple words.

<img src="/figures/LDA_usage.jpg"></img>



### 4. Results



LDA Topic Viz with pyLDAvis:

<img src="figures/LDA_topic_viz.png">



LDA top 7 keywords for each topic:

<img src="figures/LDA_top_k_topics.png">




### 5. Next Step

- Build a graphical interface to allow better user experience to interact with the program

- Implement a search engine to allow user search keywords instead of the whole paper title





## Reference

https://towardsdatascience.com/evaluate-topic-model-in-python-latent-dirichlet-allocation-lda-7d57484bb5d0

https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge/code?datasetId=551982&sortBy=voteCount

