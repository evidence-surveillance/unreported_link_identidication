import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import pairwise_distances
import json
import os


class multi_distance_inference_model(object):
    def __init__(self, query_uni_tf_vectorizer, query_uni_tfidf_vectorizer, query_bi_tf_vectorizer, query_bi_tfidf_vectorizer,
                 search_uni_tf_matrix, search_uni_tfidf_matrix, search_bi_tf_matrix, search_bi_tfidf_matrix, search_id2idx):
        """
        the inference model for the multi-distance ranking model.
        :param query_uni_tf_vectorizer: vectorizer for query text.
        transform to token frequency vector for unigram tokens.
        :param query_uni_tfidf_vectorizer: vectorizer for query text.
        transform to token frequency - inverse document frequency vector for unigram tokens.
        :param query_bi_tf_vectorizer: vectorizer for query text.
        transform to token frequency vector for bigram tokens.
        :param query_bi_tfidf_vectorizer: vectorizer for query text.
        transform to token frequency - inverse document frequency vector for bigram tokens.
        :param search_uni_tf_matrix: preprocessed vectors for search text.
        vectors are token frequency vectors for unigram token.
        :param search_uni_tfidf_matrix: preprocessed vectors for search text.
        vectors are token frequency vectors - inverser document frequency for unigram token.
        :param search_bi_tf_matrix: preprocessed vectors for search text.
        vectors are token frequency vectors for bigram token.
        :param search_bi_tfidf_matrix: preprocessed vectors for search text.
        vectors are token frequency vectors - inverser document frequency for bigram token.
        :param search_id2idx: the original id of the search documents.
        """
        self.model = make_pipeline(
                                MinMaxScaler(),
                                RandomForestClassifier(n_jobs=-1, criterion='entropy',
                                                       min_samples_split=4, min_samples_leaf=2)
                                )

        self.query_uni_tf_vectorizer = query_uni_tf_vectorizer
        self.query_uni_tfidf_vectorizer = query_uni_tfidf_vectorizer
        self.query_bi_tf_vectorizer = query_bi_tf_vectorizer
        self.query_bi_tfidf_vectorizer = query_bi_tfidf_vectorizer

        self.search_uni_tf_matrix = search_uni_tf_matrix
        self.search_uni_tfidf_matrix = search_uni_tfidf_matrix
        self.search_uni_bi_matrix = (search_uni_tf_matrix > 0).astype(int)
        self.search_bi_tf_matrix = search_bi_tf_matrix
        self.search_bi_tfidf_matrix = search_bi_tfidf_matrix
        self.search_bi_bi_matrix = (search_bi_tf_matrix > 0).astype(int)

        self.search_id2idx = search_id2idx
        # 1 -> link 0 -> detach

    def build_batch_features(self, text):
        """
        multi-distance feature generation for all the candidate documents given the query text.
        :param text: the concatenated text of the query.
        :return: the multi-distance features for all the candidate documents. shape: (num_of_candidates, 6)
        """
        assert isinstance(text, str)

        uni_query_tf = self.query_uni_tf_vectorizer.transform([text])
        bi_query_tf = self.query_bi_tf_vectorizer.transform([text])
        uni_query_tfidf = self.query_uni_tfidf_vectorizer.transform([text])
        bi_query_tfidf = self.query_bi_tfidf_vectorizer.transform([text])
        uni_query_bi = (uni_query_tf > 0).astype(int)
        bi_query_bi = (bi_query_tf > 0).astype(int)

        uni_search_tf = self.search_uni_tf_matrix
        bi_search_tf = self.search_bi_tf_matrix
        uni_search_tfidf = self.search_uni_tfidf_matrix
        bi_search_tfidf = self.search_bi_tfidf_matrix
        uni_search_bi = self.search_uni_bi_matrix
        bi_search_bi = self.search_bi_bi_matrix
        # uni-gram distance
        uni_tf_cosine_distance = pairwise_distances(uni_search_tf, uni_query_tf, metric='cosine', n_jobs=-1)
        uni_tfidf_cosine_distance = pairwise_distances(uni_search_tfidf, uni_query_tfidf, metric='cosine', n_jobs=-1)
        uni_bi_cosine_similarity = 1 - pairwise_distances(uni_search_bi, uni_query_bi, metric='cosine', n_jobs=-1)
        # bi-gram distance
        bi_tf_cosine_distance = pairwise_distances(bi_search_tf, bi_query_tf, metric='cosine', n_jobs=-1)
        bi_tfidf_cosine_distance = pairwise_distances(bi_search_tfidf, bi_query_tfidf, metric='cosine', n_jobs=-1)
        bi_bi_cosine_similarity = 1 - pairwise_distances(bi_search_bi, bi_query_bi, metric='cosine', n_jobs=-1)

        features = np.hstack([uni_tf_cosine_distance, uni_tfidf_cosine_distance, uni_bi_cosine_similarity,
                              bi_tf_cosine_distance, bi_tfidf_cosine_distance, bi_bi_cosine_similarity])
        return features

    def predict(self, texts):
        """
        predict the ranked candidates for the series of text
        :param texts: a list of the concatenated text as query.
        :return: a list of dictionaries.
        each dictionary contain
        query_id: the appearance id of the input,
        text:the query text,
        search_id_rank: the search id rank list,
        search_doc_idx_rank: the transformed original candidate id list

        """
        query_result = []
        for i, text in enumerate(texts):
            features = self.build_batch_features(text)
            ans = self.model.predict_proba(features)[:, 1].argsort()[::-1]
            id_ans = [self.search_id2idx[idx] for idx in ans]
            query_result.append({'query_id': i,
                                 'text': text,
                                 'search_id_rank': ans,
                                 'search_doc_idx_rank': id_ans})
        return query_result

    def predict_score(self, texts, output=None):
        """
        predict the likelihoods of candidates for the seires of text
        :param texts: a series of the concatenated text as query.
        :param output: the output folder path
        :return: a list of tuples.
        each tuple contain i: the appearance id of the input, text: the query text,
        ans: the search id rank list
        """
        query_result = []
        for i, text in enumerate(texts):
            features = self.build_batch_features(text)
            ans = self.model.predict_proba(features)[:, 1].tolist()
            if output is not None:
                with open(os.path.join(output, '{}.json'.format(str(i))), 'w') as fout:
                    json.dump(ans, fout)
            query_result.append((i, text, ans))
        return query_result


