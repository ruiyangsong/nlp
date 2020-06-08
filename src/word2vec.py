import jieba
from numpy import log, argsort
import pandas as pd
from gensim.models.word2vec import Word2Vec

def main():
    userdict_pth   = '../userdict.txt'  # defined by user for precisely word splitting
    stopwords_pth  = '../baidu_stopwords.txt'  # refer to [https://github.com/goto456/stopwords/]
    model_dir      = '../model/word2vec'
    data_pth       = '../data/data.csv'
    keywords_pth = '../data/keywords.csv'

    df, sentence_lst = split_word(data_pth, userdict_pth, stopwords_pth)
    model = train_model(sentences_lst=sentence_lst, model_dir=model_dir)
    chose_by_tfidf(df, outpth=keywords_pth, topK=3)


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
            tf_idf[w] = freq[w] / len(text.split()) * log(len(df) / (freq_all[w] + 1))
        w_lst = [x[0]+':'+str(x[1]) for x in sorted(tf_idf.items(), reverse=True)[:topK]]
        idx = argsort([text.split().index(w.split(':')[0]) for w in w_lst])
        return ' '.join([w_lst[i] for i in idx])

    freq_all = {}
    for text in df.tokens:
        for w in set(text.split()):
            freq_all[w] = freq_all.get(w, 0.) + 1.
    df['keywords'] = df.loc[:, 'tokens'].apply(_tfidf, args=(freq_all, topK))

    df.to_csv(outpth, index=False)



def train_model(sentences_lst, vec_dim=100, window_len=3, min_cnt=1, model_dir=None):
    model = Word2Vec(sentences_lst,
                     size=vec_dim,
                     window=window_len,
                     min_count=min_cnt,
                     sg=1,
                     hs=0,
                     negative=5,)
    model.init_sims(replace=True)
    if model_dir is not None:
        model.save('%s/dim%s_window%s_cnt%s.model'%(model_dir, vec_dim, window_len, min_cnt)) # load it later using Word2Vec.load()
    return model

if __name__ == '__main__':
    main()