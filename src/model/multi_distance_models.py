import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import paired_distances
from scipy.sparse import hstack
import json
import time
import os

class multi_distance_models(object):
    def __init__(self, query_uni_tf_matrix, query_uni_tfidf_matrix, query_bi_tf_matrix, query_bi_tfidf_matrix,
                 search_uni_tf_matrix, search_uni_tfidf_matrix, search_bi_tf_matrix, search_bi_tfidf_matrix,
                 num_model=1):

        self.query_uni_tf_matrix = query_uni_tf_matrix
        self.query_uni_tfidf_matrix = query_uni_tfidf_matrix
        self.query_uni_bi_matrix = (query_uni_tf_matrix > 0).astype(int)
        self.query_bi_tf_matrix = query_bi_tf_matrix
        self.query_bi_tfidf_matrix = query_bi_tfidf_matrix
        self.query_bi_bi_matrix = (query_bi_tf_matrix > 0).astype(int)

        self.search_uni_tf_matrix = search_uni_tf_matrix
        self.search_uni_tfidf_matrix = search_uni_tfidf_matrix
        self.search_uni_bi_matrix = (search_uni_tf_matrix > 0).astype(int)
        self.search_bi_tf_matrix = search_bi_tf_matrix
        self.search_bi_tfidf_matrix = search_bi_tfidf_matrix
        self.search_bi_bi_matrix = (search_bi_tf_matrix > 0).astype(int)

        self.num_model = num_model

        self.models = [make_pipeline(
                                MinMaxScaler(),
                                RandomForestClassifier(n_jobs=-1, criterion='entropy',
                                                       min_samples_split=4, min_samples_leaf=2)
                                     )
                       for _ in range(self.num_model)]
        # 1 -> link 0 -> detach

    def fit(self, xs, ys):
        for i, (model, x, y) in enumerate(zip(self.models, xs, ys)):
            model.fit(x, y)

            print(model.named_steps['randomforestclassifier'].n_classes_)
            print(model.named_steps['randomforestclassifier'].n_features_)
            print(model.named_steps['randomforestclassifier'].n_outputs_)
            print(model.named_steps['randomforestclassifier'].feature_importances_)

            print('the {}th model fitted'.format(str(i)))
        print('all models fitted')

    def build_pair_features(self, x):
        assert (isinstance(x, list) and (isinstance(x[0], tuple) or isinstance(x[0], list)) and len(x[0]) == 2) \
               or (isinstance(x, np.ndarray) and x.ndim == 2)
        if isinstance(x, list):
            x = np.array(x)
        qids, sids = x[:, 0], x[:, 1]
        uni_query_tf = self.query_uni_tf_matrix[qids]
        bi_query_tf = self.query_bi_tf_matrix[qids]
        uni_query_tfidf = self.query_uni_tfidf_matrix[qids]
        bi_query_tfidf = self.query_bi_tfidf_matrix[qids]
        uni_query_bi = self.query_uni_bi_matrix[qids]
        bi_query_bi = self.query_bi_bi_matrix[qids]

        uni_search_tf = self.search_uni_tf_matrix[sids]
        bi_search_tf = self.search_bi_tf_matrix[sids]
        uni_search_tfidf = self.search_uni_tfidf_matrix[sids]
        bi_search_tfidf = self.search_bi_tfidf_matrix[sids]
        uni_search_bi = self.search_uni_bi_matrix[sids]
        bi_search_bi = self.search_bi_bi_matrix[sids]

        # uni-gram similarity
        uni_tf_cosine_distance = paired_distances(uni_query_tf, uni_search_tf, metric='cosine').reshape(-1, 1)
        uni_tfidf_cosine_distance = paired_distances(uni_query_tfidf, uni_search_tfidf, metric='cosine').reshape(-1, 1)
        uni_bi_cosine_similarity = 1 - paired_distances(uni_query_bi, uni_search_bi, metric='cosine').reshape(-1, 1)
        # bi-gram similarity
        bi_tf_cosine_distance = paired_distances(bi_query_tf, bi_search_tf, metric='cosine').reshape(-1, 1)
        bi_tfidf_cosine_distance = paired_distances(bi_query_tfidf, bi_search_tfidf, metric='cosine').reshape(-1, 1)
        bi_bi_cosine_similarity = 1 - paired_distances(bi_query_bi, bi_search_bi, metric='cosine').reshape(-1, 1)

        features = np.hstack([uni_tf_cosine_distance, uni_tfidf_cosine_distance, uni_bi_cosine_similarity,
                              bi_tf_cosine_distance, bi_tfidf_cosine_distance, bi_bi_cosine_similarity])
        return features

    def build_batch_features(self, qids):
        assert isinstance(qids, int)
        qids = np.array([qids])

        uni_query_tf = self.query_uni_tf_matrix[qids]
        bi_query_tf = self.query_bi_tf_matrix[qids]
        uni_query_tfidf = self.query_uni_tfidf_matrix[qids]
        bi_query_tfidf = self.query_bi_tfidf_matrix[qids]
        uni_query_bi = self.query_uni_bi_matrix[qids]
        bi_query_bi = self.query_bi_bi_matrix[qids]

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

    def predict(self, x):
        query_result = []
        for i, qid in enumerate(x):
            features = self.build_batch_features(qid)
            ans = [model.predict_proba(features)[:, 1].argsort()[::-1] for model in self.models]
            query_result.append((qid, ans))
            if i < 10:
                print(query_result[-1])
        return query_result

    def predict_score(self, x, output=None):
        query_result = []
        for i, qid in enumerate(x):
            features = self.build_batch_features(qid)
            ans = [model.predict_proba(features)[:, 1].tolist() for model in self.models]
            if output is not None:
                with open(os.path.join(output, '{}.json'.format(str(qid))), 'w') as fout:
                    json.dump(ans, fout)
            query_result.append((qid, ans))
            if i < 10:
                print(query_result[-1])
        return query_result

    def evaluate(self, x, outpath=None):
        query_result = []
        recall1_counter = np.zeros(self.num_model)
        num_eval = len(x)
        last_second = time.time()
        if outpath:
            if os.path.exists(outpath):
                with open(outpath, 'r') as f:
                    line_count = sum(1 for _ in f)
            else:
                line_count = 0
            with open(outpath, 'w', encoding='utf-8', buffering=1000) as fout:
                for i, (qid, gid) in enumerate(x):
                    if i < line_count: continue
                    if i > 0 and i % 1000 == 0:
                        current_second = time.time()
                        predict_finish_time = (current_second - last_second) / 1000 * (num_eval - i) / 3600
                        last_second = current_second
                        print('current time', time.strftime("%H:%M:%S", time.localtime()),
                              'hours to run %.2f' % predict_finish_time, flush=True)
                        print('recall@1', recall1_counter/i, flush=True)
                        fout.flush()
                    features = self.build_batch_features(i)
                    # anss = [model.decision_function(features) for model in self.models]
                    # print([model.predict_proba(features) for model in self.models])
                    anss = [model.predict_proba(features)[:, 1] for model in self.models]

                    gscores = [ans[gid] for ans in anss]
                    ranks = [str((ans > gscore).astype(int).sum() + 1) for ans, gscore in zip(anss, gscores)]
                    recall1_counter += (np.array(ranks) == '1').astype(int)
                    fout.write('{},{},{}\n'.format(qid, gid, ','.join(ranks)))
                    # query_result.append((qid, gid, ranks))
                    # ans = self.model.decision_function(features).argsort()[::-1]
                    # query_result.append((qid, gid, np.where(ans == gid)[0][0]+1))
                    if i < 10:
                        print(qid, gid, ranks, gscores, flush=True)
                        print('current time', time.strftime("%H:%M:%S", time.localtime()), flush=True)
                fout.flush()
        else:
            for i, (qid, gid) in enumerate(x):
                if i > 0 and i % 1000 == 0:
                    current_second = time.time()
                    predict_finish_time = (current_second-last_second)/1000*(num_eval-i)/3600
                    last_second = current_second
                    print('current time', time.strftime("%H:%M:%S", time.localtime()),
                          'hours to run %.2f' % predict_finish_time, flush=True)
                    print('recall@1', recall1_counter / i, flush=True)
                features = self.build_batch_features(i)
                # anss = [model.decision_function(features) for model in self.models]
                anss = [model.predict_proba(features)[:, 1] for model in self.models]
                gscores = [ans[gid] for ans in anss]
                ranks = [(ans > gscore).astype(int).sum()+1 for ans, gscore in zip(anss, gscores)]
                recall1_counter += (np.array(ranks) == '1').astype(int)
                query_result.append((qid, gid, ranks))
                # ans = self.model.decision_function(features).argsort()[::-1]
                # query_result.append((qid, gid, np.where(ans == gid)[0][0]+1))
                if i < 10:
                    print(qid, gid, ranks, flush=True)
                    print('current time', time.strftime("%H:%M:%S", time.localtime()),
                          'hours to run %.2f' % predict_finish_time, flush=True)
        return query_result

