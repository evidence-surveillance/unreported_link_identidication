from src.model.multi_distance_inference_models import multi_distance_inference_model
import pickle as pk
import scipy as sp
import argparse
import os


def file_reader(path):
    files = []
    with open(path, 'r') as fin:
        for line in fin:
            files.append(line.strip())
    return files


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='inference the ')
    parser.add_argument('--model_path', help='the path of model folder', type=str)
    # each document is only preprocessed by concatenating the targeted domain of free text.
    # For example, in PubMed, the title and abstract of an article are extracted and concatenated
    # For trials, the brief title, official title, brief summarie, and description are used
    parser.add_argument('--query_path', help='the path of query files. each line indicates one preprocessed document.', type=str)
    parser.add_argument('--search_path', help='the path of search candidate folder.', type=str)
    parser.add_argument('--output', help='the path of output result folder', type=str)

    args = parser.parse_args()
    print(args)

    print('model folder checking...')
    assert os.path.isdir(args.model_path)
    assert os.path.isfile(os.path.join(args.search_path, 'query_bi_tf_vectorizer.pk'))
    assert os.path.isfile(os.path.join(args.search_path, 'query_bi_tfidf_vectorizer.pk'))
    assert os.path.isfile(os.path.join(args.search_path, 'query_uni_tf_vectorizer.pk'))
    assert os.path.isfile(os.path.join(args.search_path, 'query_uni_tfidf_vectorizer.pk'))
    assert os.path.isfile(os.path.join(args.search_path, 'search_bi_tf_vectorizer.pk'))
    assert os.path.isfile(os.path.join(args.search_path, 'search_bi_tfidf_vectorizer.pk'))
    assert os.path.isfile(os.path.join(args.search_path, 'search_uni_tf_vectorizer.pk'))
    assert os.path.isfile(os.path.join(args.search_path, 'search_uni_tfidf_vectorizer.pk'))
    assert os.path.isfile(os.path.join(args.search_path, 'search_bi_tf_matrix.npz'))
    assert os.path.isfile(os.path.join(args.search_path, 'search_bi_tfidf_matrix.npz'))
    assert os.path.isfile(os.path.join(args.search_path, 'search_uni_tf_matrix.npz'))
    assert os.path.isfile(os.path.join(args.search_path, 'search_uni_tfidf_matrix.npz'))
    assert os.path.isfile(os.path.join(args.search_path, 'search_uni_tfidf_matrix.npz'))
    assert os.path.isfile(os.path.join(args.search_path, 'search_id2idx.csv'))
    assert os.path.isfile(os.path.join(args.model_path, 'model.pk'))
    print('model folder check done.')

    print('model loading...')
    with open(os.path.join(args.search_path, 'query_uni_tf_vectorizer.pk'), 'rb') as fin:
        query_uni_tf_vectorizer = pk.load(fin)
    with open(os.path.join(args.search_path, 'query_uni_tfidf_vectorizer.pk'), 'rb') as fin:
        query_uni_tfidf_vectorizer = pk.load(fin)
    with open(os.path.join(args.search_path, 'query_bi_tf_vectorizer.pk'), 'rb') as fin:
        query_bi_tf_vectorizer = pk.load(fin)
    with open(os.path.join(args.search_path, 'query_bi_tfidf_vectorizer.pk'), 'rb') as fin:
        query_bi_tfidf_vectorizer = pk.load(fin)
    search_uni_tf_matrix = sp.sparse.load_npz(os.path.join(args.search_path, 'search_uni_tf_matrix.npz'))
    search_uni_tfidf_matrix = sp.sparse.load_npz(os.path.join(args.search_path, 'search_uni_tfidf_matrix.npz'))
    search_bi_tf_matrix = sp.sparse.load_npz(os.path.join(args.search_path, 'search_bi_tf_matrix.npz'))
    search_bi_tfidf_matrix = sp.sparse.load_npz(os.path.join(args.search_path, 'search_bi_tfidf_matrix.npz'))
    search_id = []
    with open(os.path.join(args.search_path, 'search_id2idx.csv'), 'r') as fin:
        for line in fin:
            search_id.append(line.strip().split(',')[0])
    with open(os.path.join(args.model_path, 'model.pk'), 'rb') as fin:
        model = pk.load(fin)[0]
    print('model loading done')

    # load text document
    query_files = file_reader(args.query_path)
    # model load
    md_model = multi_distance_inference_model(query_uni_tf_vectorizer, query_uni_tfidf_vectorizer,
                                              query_bi_tf_vectorizer, query_bi_tfidf_vectorizer,
                                              search_uni_tf_matrix, search_uni_tfidf_matrix,
                                              search_bi_tf_matrix, search_bi_tfidf_matrix,
                                              search_id)
    md_model.model = model

    # # predict the probability of links between a pair of query file and search file,
    # # and output to the folder in json format.
    # # following sorting is required
    # # multiple candidate files may have the same probabilities thus they should ranked equally
    # query_result = md_model.predict_score(range(len(query_files)), args.output)

    # predicted the ranked result based on the probability of being a link pair
    # it should be noted several candidate files may have the same probabilities
    # thus they may be sorted in various orders
    query_result = md_model.predict(query_files)

