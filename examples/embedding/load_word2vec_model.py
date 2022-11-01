from gensim.models import Word2Vec

# 모델 로딩
model = Word2Vec.load('nvmc.model')
print('corpus_total_word', model.corpus_total_words)

# '사랑'이라는 단어로 생성된 단어 임베딩 벡터 -> 200개
print('사랑:', model.wv['사랑']) # Word Vector

# 단어 유사도 계산 -> 벡터 공간에서 가장 가까운 거리에 있는 단어를 반환(0~1)
print('일요일과 월요일의 유사도 \t', model.wv.similarity(w1='일요일', w2='월요일'))
print('안성기와 배우의 유사도 \t', model.wv.similarity(w1='안성기', w2='배우'))
print('대기업과 삼성의 유사도 \t', model.wv.similarity(w1='대기업', w2='삼성'))
print('히어로와 마블의 유사도 \t', model.wv.similarity(w1='히어로', w2='마블'))
print('해리와 포터의 유사도 \t', model.wv.similarity(w1='해리', w2='포터'))

# 가장 유사한 단어 추출
print(model.wv.most_similar('안성기', topn=10))
print(model.wv.most_similar('시리즈', topn=10))
print(model.wv.most_similar('얼굴', topn=10))