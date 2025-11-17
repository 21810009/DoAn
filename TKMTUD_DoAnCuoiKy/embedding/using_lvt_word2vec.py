from gensim.models import Word2Vec

# Tải model đã lưu
model = Word2Vec.load("lvt.word2vec.model")

# Sử dụng như bình thường
print(model.wv.most_similar("hiếu", topn=5))