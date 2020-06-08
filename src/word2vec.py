import jieba
import pandas as pd
from gensim.models.word2vec import Word2Vec

def main():
    userdict_pth   = '../userdict.txt'  # defined by user for precisely word splitting
    stopwords_pth  = '../baidu_stopwords.txt'  # refer to [https://github.com/goto456/stopwords/]
    data_pth       = '../data/data.csv'
    data_token_pth = '../data/data_token.csv'

    model = train_model(sentences_lst=word_split(data_pth, userdict_pth, stopwords_pth, outpth=data_token_pth), model_dir='../model')


def word_split(data_pth, userdict_pth, stopwords_pth, outpth, cloumn_name='商品名称'):
    def tokenizer(text, userdict_pth, stopwords_pth):
        jieba.load_userdict(userdict_pth)  # load custom dictionary
        with open(stopwords_pth, encoding='utf-8') as f:
            stopwords_lst = [x.strip() for x in f.readlines()]
        return ' '.join([tag for tag in jieba.cut(text) if tag not in stopwords_lst])

    df = pd.read_csv(data_pth, encoding='utf-8')
    df['tokens'] = df.loc[:, cloumn_name].apply(tokenizer, args=(userdict_pth, stopwords_pth))
    df.to_csv(outpth, index=False)
    return [tok.split() for tok in df.tokens]


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