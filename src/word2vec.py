#!/usr/bin/env python
import jieba
import numpy as np
import pandas as pd
from gensim.models.word2vec import Word2Vec

def main():
    userdict_pth   = '../userdict.txt'  # defined by user for precisely word splitting
    stopwords_pth  = '../baidu_stopwords.txt'  # refer to [https://github.com/goto456/stopwords/]
    data_pth       = '../data/data.csv'
    keywords_pth   = '../data/keywords.csv'
    topK           = 3
    vec_dim        = 100
    window_len     = 3
    min_cnt        = 1
    model_pth      = '../model/word2vec/dim%s_window%s_cnt%s.model' % (vec_dim, window_len, min_cnt)

    df, sentence_lst = split_word(data_pth, userdict_pth, stopwords_pth)
    chose_by_tfidf(df, outpth=keywords_pth, topK=topK)
    model = train_model(sentences_lst=sentence_lst, model_pth=model_pth)#train word2vec model (CBOW or Skip-Gram)
    generate_array(data_pth=keywords_pth, model=model, model_pth='../model/word2vec/dim100_window3_cnt1.model', mode='stack', vec_dim=vec_dim, num_words=topK)
    generate_array(data_pth=keywords_pth, model=model, model_pth='../model/word2vec/dim100_window3_cnt1.model', mode='padding', vec_dim=vec_dim, num_words=topK)
    generate_array(data_pth=keywords_pth, model=model, model_pth='../model/word2vec/dim100_window3_cnt1.model', mode='sum', vec_dim=vec_dim, num_words=topK)

def split_word(data_pth, userdict_pth, stopwords_pth, column_name='商品名称'):
    '''
    split word at the column_name based on the costume dictionary and stopwords.
    '''
    def _tokenizer(text, userdict_pth, stopwords_pth):
        jieba.load_userdict(userdict_pth)  # load custom dictionary
        with open(stopwords_pth, encoding='utf-8') as f:
            stopwords_lst = [x.strip() for x in f.readlines()]
        return ' '.join([tag for tag in jieba.cut(text) if tag not in stopwords_lst])

    df = pd.read_csv(data_pth, encoding='utf-8')
    df['tokens'] = df.loc[:, column_name].apply(_tokenizer, args=(userdict_pth, stopwords_pth))

    return df, [text.split() for text in df.tokens]

def chose_by_tfidf(df, outpth, topK=3):
    '''
    chose topK keywords from tokens by tf-idf.
    '''
    def _tfidf(text, freq_all, topK):
        freq = {}
        idx = {}
        tf_idf = {}
        for w in text.split():
            freq[w] = freq.get(w, 0.) + 1.
        for w in text.split():
            tf_idf[w] = freq[w] / len(text.split()) * np.log(len(df) / (freq_all[w] + 1))
        w_lst = [x[0]+':'+str(x[1]) for x in sorted(tf_idf.items(), reverse=True)[:topK]]
        idx = np.argsort([text.split().index(w.split(':')[0]) for w in w_lst])
        return ' '.join([w_lst[i] for i in idx])

    freq_all = {}
    for text in df.tokens:
        for w in set(text.split()):
            freq_all[w] = freq_all.get(w, 0.) + 1.
    df['keywords'] = df.loc[:, 'tokens'].apply(_tfidf, args=(freq_all, topK))

    df.to_csv(outpth, index=False)

def train_model(sentences_lst, vec_dim=100, window_len=3, min_cnt=1, model_pth=None):
    model = Word2Vec(sentences_lst,
                     size=vec_dim,
                     window=window_len,
                     min_count=min_cnt,
                     sg=1,
                     hs=0,
                     negative=5,)
    model.init_sims(replace=True)
    if model_pth is not None:
        model.save(model_pth) # load it later using Word2Vec.load()
    return model


def generate_array(data_pth, model_pth=None, model=None, mode='stack', vec_dim=100, num_words=3):
    '''
    generate array from column keywords[keywords,tf-idf] in a csv file.
    ---
    mode = ['stack', 'padding', 'sum']
    'stack': stack word vectors. Size is N_i * vec_dim.
    'padding': stack word vectors with padding when number of words is short than num_words. Size is num_words * vec_dim.
    'sum': weighted sum word vectors by original tf-idf values, Size is also vec_dim.
    '''
    if model_pth is None and model is None:
        print('[WARNING!] model_pth and model can not be None at the same time!')
        exit(0)
    elif model is not None:
        pass
    elif model_pth is not None:
        model = Word2Vec.load(model_pth)
    print('---mode is', mode)
    df = pd.read_csv(data_pth)
    x_all = []
    for i in range(len(df)):
        keywords_lst = [x.split(':')[0] for x in df.iloc[i,:][['keywords']].values[0].split(' ')]
        tfidf_lst    = [float(x.split(':')[1]) for x in df.iloc[i,:][['keywords']].values[0].split(' ')]
        if mode == 'stack':
            tmp_vec = []
            for keyword, tfidf in zip(keywords_lst, tfidf_lst):
                try:
                    tmp_vec = np.hstack((tmp_vec, model.wv[keyword] * tfidf))
                except:
                    tmp_vec = np.hstack((tmp_vec, np.zeros(vec_dim))) # for words that do not occur in dictionary
        elif mode == 'padding':
            tmp_vec = []
            for keyword, tfidf in zip(keywords_lst, tfidf_lst):
                try:
                    tmp_vec = np.hstack((tmp_vec, model.wv[keyword] * tfidf))
                except:
                    tmp_vec = np.hstack((tmp_vec, np.zeros(vec_dim)))  # for words that do not occur in dictionary
            if len(keywords_lst) < num_words:
                tmp_vec = np.hstack((tmp_vec, np.zeros((num_words - len(keywords_lst)) * vec_dim)))
        elif mode == 'sum':
            tmp_vec = np.zeros(vec_dim)
            for keyword, tfidf in zip(keywords_lst, tfidf_lst):
                try:
                    tmp_vec += model.wv[keyword] * tfidf
                except:
                    pass # for words that do not occur in dictionary
        x_all.append(tmp_vec)
    x = np.array(x_all)
    y = df.loc[:,'商品编码'].values - 101
    np.savez('../data/mode_%s.npz'%mode, x=x, y=y)
    print('\nx shape: %s'
          '\ny shape: %s'%(x.shape, y.shape))

    return x, y

if __name__ == '__main__':
    main()