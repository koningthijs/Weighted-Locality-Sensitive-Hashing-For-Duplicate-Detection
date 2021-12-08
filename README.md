# Weighted-Locality-Sensitive-Hashing-For-Duplicate-Detection
## Intro

This project is made as part of the course computer science for business analytics, given by Flavius Frasincar

The data that is used consists of 1624 TV's offered by 4 different webshops. The goal of the algorithm is to identify duplicate products across different web shops.

## The code

### Hyper parameters
Before running the code the hyper parameters have to be set. 
The key readingthreshold is later used to identify similar keys within the features map
Under model word choice different settings for choosing restrictions on which words are used to create the input matrix
Under LSH different settings for the LSH algorithm can be set such as the amount of rows within a band or the fraction of the amount e=of rows the signature matrix should counsist of
Under clustering only the threshold of the agglomerative clustering algorithm is set

### Load data 
In this section the data is loaded to python

### Cleaning
In this sectio the data is cleaned to create as much similarity between products. This is done by removing some special caraters and capitals. Also units are transformed to one universal unit representation

### Reading and matching keys
In this section some key data is extracted for later use such as the potential model ID's , the brand of a product and the shop it belongs to.
Also the key's of the key value pairs are matched by a simple algorithm

### Fill dataset
In this section the dataset is created in which the data is listed 

### Chosing words
In this section the words that are used for creating binary representations are arecollected and chosen. 
Also some words are added extra times to create the weights of the WLSH

### Creating input matrix
In this section the binary representations are created by using the words chosen before

### Creating signature matrix
The rows of the input matrix are hashed by random hash functions which create unique hashes. These together form signatires that are placed in the signature matrix

### LSH
LSH is applied on the signature matrix and the products are placed into buckets

### find potential pairs from potmodelID
A list of pairs that can be linked by looking at the potential model ID's found within the titles f the products is created

### Define candidate pairs
Buckets are extracted and potential pairs are created 

### remove same shop and different brand 
Here a logical filter on brands and shops is applied to remove candidate pairs that do not make any sense
    
### find true pairs
For further evaluation a list of true pairs is also created. this make evaluation easier later on

### evalueate   
Here the performance of LSH is evaluated and printed as output. 

### CLustering:
CLustering is applied to find clusters of products. These products are then transpfrmed to pairs of two to find the final result.
   
### Performance
This last step is used to calculate the total performance of the algorithm.
Tis will eventualy be printed.
