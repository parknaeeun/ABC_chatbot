from gensim.models import Word2Vec
from konlpy.tag import Komoran
import time

# 네이버 영화 리뷰 데이터를 읽어오는 사용자 정의 함수
def read_review_data(filename):
    with open(filename, encoding='UTF-8') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
        data = data[1:]  #header 제거
    return data

# 측정 시작
start = time.time()

print('1) 말뭉치 데이터 읽기 시작')
review_data = read_review_data('ratings.txt')
print('ratings count', len(review_data)) #리뷰 데이터 전체 개수
print('1) 말뭉치 데이터 읽기 완료', time.time()-start)

print('2) 형태소에서 명사만 추출 시작')
komoran = Komoran()
docs = [komoran.nouns(sentence[1]) for sentence in review_data]
print('2) 형태소에서 명사만 추출 완료', time.time()-start)

print('3) word2vec 모델 학습 시작')
# sentences: 모델 학습에 필요한 데이터, vector_size: 단어 임베딩 차원,
# window: 주변 단어 윈도우 크기 sg= 0 CBOW, 1 skip-gram
model = Word2Vec(sentences=docs, vector_size=200, window=4, sg=1)
print('3) word2vec 모델 학습 완료', time.time()-start)

print('4) 학습된 모델 저장 시작')
model.save('nvmc.model')
print('4) 학습된 모델 저장 완료', time.time()-start)

print('말뭉치 전체 단어 개수', model.corpus_count)
print('학습된 말뭉치 개수', model.corpus_total_words)
