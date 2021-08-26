# Identifying unreported links between ClinicalTrials.gov trial registrations and their published results

This module facilitates the ranking of candidates given query files according to the predicted probabilities from the model. 

This is the demo package for discoviering the unestablished links between clnical trial registraions and PubMed articles.

This package includes source codes for training and inference.

## Packagge Requirements

All the codes are written using Python 3 and the experiments are run on Mac.
Necessary pacakges are:
* scikit-learn==0.23.2
* numpy==1.19.1
* scipy==1.3.3

## How to run
You need to prepare:
* A folder contains the trained model
* A foler contains the preprocessed candidate text, vectorizers for the candidate text, and vectorizers for the query text. 
* A file containing the query text. Each line in the file refers to the contatenated raw text.

```python
python inference.py --model_path DIR_OF_MODEL --query_file QUERY_FILE --search_path DIR_OF_SEARCH_FILES --output DIR_OF_OUTPUT
```

## Data Description

    |-data
        |-c2p -> the preprocessed dataset fold to rank PubMed articles given clinical trial registrations. c for clnical trial registration, and p for PubMed articles
            |-query_uni_tf_matrix.npz -> the vectorized clinical trial registrations. These only contains those on the ground truth list. The vectors are based on the token frequency of unigram tokens.
            |-query_uni_tfidf_matrix.npz -> the vectorized clinical trial registrations. These only contains those on the ground truth list. The vectors are based on the token frequency-inverse document frequency of unigram tokens.
            |-query_bi_tf_matrix.npz -> the vectorized clinical trial registrations. These only contains those on the ground truth list. The vectors are based on the token frequency of bigram tokens.
            |-query_bi_tfidf_matrix.npz -> the vectorized clinical trial registrations. These only contains those on the ground truth list. The vectors are based on the token frequency-inverse document frequency of bigram tokens.
            |-search_uni_tf_matrix.npz -> the vectorized clinical trial registrations. These contains all the PubMed articles. The vectors are based on the token frequency of unigram tokens.
            |-search_uni_tfidf_matrix.npz -> the vectorized clinical trial registrations. These contains all the PubMed articles. The vectors are based on the token frequency-inverse document frequency of unigram tokens.
            |-search_bi_tf_matrix.npz -> the vectorized clinical trial registrations. These contains all the PubMed articles. The vectors are based on the token frequency of bigram tokens.
            |-search_bi_tfidf_matrix.npz -> the vectorized clinical trial registrations. These contains all the PubMed articles. The vectors are based on the token frequency-inverse document frequency of bigram tokens.
            |-ground.npz -> the vector storage of ground pairs.
            |-cidx2pidx.csv -> the automatically generaget ground truth of clnical nctid index and PubMed publish id index pairs. NOTE: these are represented with index tranformed from original nctid or publish id.
            |-query_bi_tf_vectorizer.pk -> the vectorizer to generate bigram token frequency vector for input clinical trial registration text
            |-query_bi_tfidf_vectorizer.pk -> the vectorizer to generate bigram token frequency-inverse document frequncy vector for input clinical trial registration text
            |-query_uni_tf_vectorizer.pk -> the vectorizer to generate unigram token frequency vector for input clinical trial registration text
            |-query_uni_tfidf_vectorizer.pk -> the vectorizer to generate unigram token frequency-inverse document frequncy vector for input clinical trial registration text
            |-search_bi_tf_vectorizer.pk -> the vectorizer to generate bigram token frequency vector for candidate PubMed article text
            |-search_bi_tfidf_vectorizer.pk -> the vectorizer to generate bigram token frequency-inverse document frequncy vector for candidate PubMed article text
            |-search_uni_tf_vectorizer.pk -> the vectorizer to generate unigram token frequency vector for candidate PubMed article text
            |-search_uni_tfidf_vectorizer.pk -> the vectorizer to generate unigram token frequency-inverse document frequncy vector for candidate PubMed article text
        |-p2c -> the preprocessed dataset fold to rank clinical trial registrations given PubMed articles
            |-query_uni_tf_matrix.npz -> the vectorized PubMed articles. These only contains those on the ground truth list. The vectors are based on the token frequency of unigram tokens.
            |-query_uni_tfidf_matrix.npz -> the vectorized PubMed articles. These only contains those on the ground truth list. The vectors are based on the token frequency-inverse document frequency of unigram tokens.
            |-query_bi_tf_matrix.npz -> the vectorized PubMed articles. These only contains those on the ground truth list. The vectors are based on the token frequency of bigram tokens.
            |-query_bi_tfidf_matrix.npz -> the vectorized PubMed articles. These only contains those on the ground truth list. The vectors are based on the token frequency-inverse document frequency of bigram tokens.
            |-search_uni_tf_matrix.npz -> the vectorized PubMed articles. These contains all the clinical trial registrations. The vectors are based on the token frequency of unigram tokens.
            |-search_uni_tfidf_matrix.npz -> the vectorized PubMed articles. These contains all the clinical trial registrations. The vectors are based on the token frequency-inverse document frequency of unigram tokens.
            |-search_bi_tf_matrix.npz -> the vectorized PubMed articles. These contains all the clinical trial registrations. The vectors are based on the token frequency of bigram tokens.
            |-search_bi_tfidf_matrix.npz -> the vectorized PubMed articles. These contains all the clinical trial registrations. The vectors are based on the token frequency-inverse document frequency of bigram tokens.
            |-ground.npz -> the vector storage of ground pairs.
            |-pidx2cidx.csv -> the automatically generaget ground truth of PubMed publish id index and clnical nctid index pairs. NOTE: these are represented with index tranformed from original nctid or publish id.
            |-query_bi_tf_vectorizer.pk -> the vectorizer to generate bigram token frequency vector for input PubMed article text
            |-query_bi_tfidf_vectorizer.pk -> the vectorizer to generate bigram token frequency-inverse document frequncy vector for input PubMed article text
            |-query_uni_tf_vectorizer.pk -> the vectorizer to generate unigram token frequency vector for input PubMed article text
            |-query_uni_tfidf_vectorizer.pk -> the vectorizer to generate unigram token frequency-inverse document frequncy vector for input PubMed article text
            |-search_bi_tf_vectorizer.pk -> the vectorizer to generate bigram token frequency vector for candidate clinical trial registrations text
            |-search_bi_tfidf_vectorizer.pk -> the vectorizer to generate bigram token frequency-inverse document frequncy vector for candidate clinical trial registrations text
            |-search_uni_tf_vectorizer.pk -> the vectorizer to generate unigram token frequency vector for candidate clinical trial registrations text
            |-search_uni_tfidf_vectorizer.pk -> the vectorizer to generate unigram token frequency-inverse document frequncy vector for candidate clinical trial registrations text
        |-clinic_db.csv -> the extracted clinical trial registrations. format: nctid, concatenated_text, date
        |-clnic_nctid2idx.csv -> processed transformation of nctid to index. format: nctid, index
        |-pubmed_db.csv -> the extracted PubMed articles. format:publishid, nctid (nan if does not exist), concatenated_text, date
        |-pubmed_pubid2idx.csv  -> processed transformation of publishid to index. format: publishid, index

## Source Code Description
    |-src
        |- __init__.py
        |-model
            |- __init__.py
            |-multi_distance_models.py -> the designed model with feature generation, model fit, prediction, etc. functionalities
            |-multi_distance_inference_models.py -> the designed model with feature generation, prediction, etc. functionalities
        |-utils
            |- __init__.py
            |-filters.py -> has filter text by date functions
            |-nlp.py -> has stopwords, tokenization related functions
            |-utils.py -> has flatten array function
        |-inference.py -> the main file for inference ranking candidates given input text
        
    |-eval_multi_distance_models_hpc.py -> the entrance to train and evaluate the model
    
## Model Description
    
    |-exps
        |-eval_multi_distance_model/
            |-eval_c2p_rf_5_1000_single/
                |-model.pk -> the trained random forest model to return likelihood given vectorized clinical trial registration text
                |-params.pk -> the dumped parameters to train the model
            |-eval_p2c_rf_5_1000_single/
                |-model.pk -> the trained random forest model to return likelihood given vectorized PubMed article text
                |-params.pk -> the dumped parameters to train the model
        
