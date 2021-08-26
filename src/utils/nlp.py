import spacy
import string


def get_stopwords():
    pubmed_stopwords = 'a, about, again, all, almost, also, although, always, among, an, and, another, any, are, as, ' \
                       'at, be, because, been, before, being, between, both, but, by,can, could,did, do, does, done, ' \
                       'due, during,each, either, enough, especially, etc, for, found, from, further, had, has, ' \
                       'have, having, here, how, however, i, if, in, into, is, it, its, itself, just, kg, km, made, ' \
                       'mainly, make, may, mg, might, ml, mm, most, mostly, must, nearly, neither, no, nor, ' \
                       'obtained, of, often, on, our, overall, perhaps, pmid, quite, rather, really, regarding, ' \
                       'seem, seen, several, should, show, showed, shown, shows, significantly, since, so, some, ' \
                       'such, than, that, the, their, theirs, them, then, there, therefore, these, they, this, ' \
                       'those, through, thus, to, upon, various, very, was, we, were, what, when, which, while, ' \
                       'with, within, without, would'.replace(' ', '').split(',')
    nlp = spacy.load("en_core_web_sm")
    stopwords = list(nlp.Defaults.stop_words)
    punctuations = list(string.punctuation)
    single_chars = ['b', 'c', 'd', 'e', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
                    'w', 'x', 'y', '‘', '’']
    allstopwords = stopwords + punctuations + single_chars + pubmed_stopwords
    allstopwords = list(set(allstopwords))
    return allstopwords


def spacy_tokenization(nlp, text):
    if not isinstance(text, str):
        print(text)
    doc = nlp(text)
    return [[token.text.strip() for token in sent if token.text.strip()] for sent in doc.sents]


def tokenization_func(x):

    return [t.strip() for t in x if t.strip()]


def bracket_replace(text):
    return text.replace('"', '').replace('NULL', '')\
        .replace('{', '').replace('}', '')\
        .replace('[', '').replace(']', '')\
        .replace('(', '').replace(')', '')
