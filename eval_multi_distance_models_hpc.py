from src.model.multi_distance_models import multi_distance_models
from sklearn import preprocessing
import pickle as pk
import numpy as np
import scipy as sp
import argparse
import os
import sys
import random

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='evaluate the token based tf-idf vector cosine '
                                                 'similarity document ranking method')
    parser.add_argument('--data_path', help='the path of tf matrix',
                        default='/project/RDS-FMH-CouplingCTgov-RW/data/tf_matrix_', type=str)

    parser.add_argument('--n_neighbours', help='the number of neighbours queried from nn',
                        nargs="+", type=int)
    parser.add_argument('--n_trains', help='the number of training examples',
                        nargs="+", type=int)
    parser.add_argument('--output', help='the path of output result folder', type=str)
    parser.add_argument('--name', help='the name of evaluation', type=str)
    parser.add_argument('--override', help='whether override the existing folder', action='store_true')

    args = parser.parse_args()
    print(args)
    data_path = args.data_path

    n_neighbours = args.n_neighbours
    n_trains = args.n_trains
    output = args.output
    name = args.name
    override = args.override

    if not (os.path.exists(output) and os.path.isdir(output)):
        os.makedirs(output)

    if not os.path.exists(os.path.join(output, name)):
        os.makedirs(os.path.join(output, name))
    elif not args.override:
        print('Folder exists! Exist')
        sys.exit()

    with open(os.path.join(output, name, 'params.pk'), 'wb') as fout:
        pk.dump(args, fout)

    query_uni_tf_matrix = sp.sparse.load_npz(os.path.join(data_path, 'query_uni_tf_matrix.npz'))
    query_uni_tfidf_matrix = sp.sparse.load_npz(os.path.join(data_path, 'query_uni_tfidf_matrix.npz'))
    query_bi_tf_matrix = sp.sparse.load_npz(os.path.join(data_path, 'query_bi_tf_matrix.npz'))
    query_bi_tfidf_matrix = sp.sparse.load_npz(os.path.join(data_path, 'query_bi_tfidf_matrix.npz'))

    search_uni_tf_matrix = sp.sparse.load_npz(os.path.join(data_path, 'search_uni_tf_matrix.npz'))
    search_uni_tfidf_matrix = sp.sparse.load_npz(os.path.join(data_path, 'search_uni_tfidf_matrix.npz'))
    search_bi_tf_matrix = sp.sparse.load_npz(os.path.join(data_path, 'search_bi_tf_matrix.npz'))
    search_bi_tfidf_matrix = sp.sparse.load_npz(os.path.join(data_path, 'search_bi_tfidf_matrix.npz'))

    models = multi_distance_models(query_uni_tf_matrix, query_uni_tfidf_matrix,
                                   query_bi_tf_matrix, query_bi_tfidf_matrix,
                                   search_uni_tf_matrix, search_uni_tfidf_matrix,
                                   search_bi_tf_matrix, search_bi_tfidf_matrix,
                                   num_model=len(n_trains)*len(n_neighbours))
    print('model', models)
    ground_link_pair = np.load(os.path.join(data_path, 'ground.npz'))['arr_0']
    assert len(ground_link_pair) > max(n_trains)

    if os.path.exists(os.path.join(output, name, 'model.pk')):
        models.models = pk.load(open(os.path.join(output, name, 'model.pk'), 'rb'))
        print('model loaded')
    else:
        train_nns = np.load(os.path.join(data_path, 'neighbours.npz'))['arr_0']
        le = preprocessing.LabelEncoder()
        le.fit(['detach', 'link'])
        train_features = []
        train_labels = []
        for n_train in n_trains:
            for nn in n_neighbours:
                print('n_train', n_train, 'n_neighbour', nn)
                train_id_pairs = []
                train_label = []
                if nn == -1:
                    for i, (qid, gid) in enumerate(ground_link_pair[:n_train]):
                        for j in range(search_uni_tf_matrix.shape[0]):
                            train_id_pairs.append((i, j))
                            if j == gid:
                                train_label.append('link')
                            else:
                                train_label.append('detach')
                else:
                    for i, (qid, gid) in enumerate(ground_link_pair[:n_train]):
                        train_id_pairs.append((i, gid))
                        train_label.append('link')
                        c = 0
                        randomlist = random.sample(range(search_uni_tf_matrix.shape[0]), 2*nn+1)
                        for rid in randomlist:
                            if rid != gid and c < nn:
                                train_id_pairs.append((i, rid))
                                train_label.append('detach')
                                c += 1
                train_feature = models.build_pair_features(train_id_pairs)
                train_label = le.transform(train_label)
                print('shape of train features', train_feature.shape, train_label.shape)
                train_features.append(train_feature)
                train_labels.append(train_label)
        print('model fitting...')
        models.fit(train_features, train_labels)
        del train_features, train_labels
        with open(os.path.join(output, name, 'model.pk'), 'wb') as fout:
            pk.dump(models.models, fout)
        print('model fitted')
    print('model evaluating...')
    query_result = models.evaluate(ground_link_pair, os.path.join(output, name, 'query_result.csv'))
    # with open(os.path.join(output, name, 'query_result.pk'), 'wb') as fout:
    #     pk.dump(query_result, fout)
    print('model evaluation done.')
