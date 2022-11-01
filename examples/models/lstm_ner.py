import tensorflow as tf
from tensorflow.keras import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# file reader; 사용자 정의 함수
def read_file(file_name):
    sents = []
    with open(file_name, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for idx, l in enumerate(lines):
            if l[0] == ';' and lines[idx + 1][0] == '$':
                this_sent = []
            elif l[0] == '$' and lines[idx - 1][0] == ';':
                continue
            elif l[0] == '\n':
                sents.append(this_sent)
            else:
                this_sent.append(tuple(l.split()))
    return sents

# 1) 학습 데이터 불러 오기(장문 텍스트 2줄은 저장 x, id, 단어, 형태소, BIO)
corpus = read_file('train.txt')
print('corpus', corpus[:10])


# 2) 말뭉치 데이터 -> 단어, BIO 태그만 불러와서 데이터셋 생성
## 말뭉치 -> 단어, BIO 태그 추출 -> 시퀀스로 변환
sentences, tags = [], []
for t in corpus: #[[], [], []]
    tagged_sentence = []
    sentence, bio_tag = [], []
    for w in t: #[[(), (), ()]]
        tagged_sentence.append((w[1], w[3]))
        sentence.append(w[1])
        bio_tag.append(w[3])

    sentences.append(sentence)
    tags.append(bio_tag)

print('샘플 크기:', len(sentences))
print('0번째 샘플 문장 시퀀스:', sentences[0])
print('0번째 샘플 문장 BIO 태그:', tags[0])

print('샘플 문장 시퀀스 최대 길이', max(len(l) for l in sentences))
print('샘플 문장 시퀀스 평균 길이', sum(map(len, sentences)) / len(sentences)) #map(함수, 리스트): 리스트 각 요소에 함수 적용


# 3) 단어 단위의 시퀀스 부여(Out Of Vocabulary)
sent_tokenizer = preprocessing.text.Tokenizer(oov_token='OOV') #첫 번째 인덱스는 OOV 사용
sent_tokenizer.fit_on_texts(sentences)
tag_tokenizer = preprocessing.text.Tokenizer(lower=False) #대문자 그대로 사용
tag_tokenizer.fit_on_texts(tags)

vocab_size = len(sent_tokenizer.word_index) + 1
tag_size = len(tag_tokenizer.word_index) + 1

print('단어 사전 크기:', vocab_size)
print('BIO 태그 크기:', tag_size)
print(tag_tokenizer.word_index)


# 4) 시퀀스를 적용한 x_train, y_train 생성
x_train = sent_tokenizer.texts_to_sequences(sentences)
y_train = tag_tokenizer.texts_to_sequences(tags)

# 검증을 위한 단어 사전
index_to_word = sent_tokenizer.index_word
index_to_ner = tag_tokenizer.index_word
index_to_ner[0] = 'PAD' #key(0):value('PAD')
print('index_to_ner', index_to_ner)


# 5) 문장 패딩 처리
max_len = 40 #평균 처리
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=max_len, padding='post')
y_train = preprocessing.sequence.pad_sequences(y_train, maxlen=max_len, padding='post')


# 6) 학습 데이터(80), 테스트 데이터(20) 분리
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train,
                                                    test_size=0.2, random_state=0)

# 7) BIO 태그(1~7) -> onehot 인코딩으로 변경
y_train = tf.keras.utils.to_categorical(y_train, num_classes=tag_size)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=tag_size)

print('학습 샘플 시퀀스 shape', x_train.shape)
print('학습 샘플 레이블 shape', y_train.shape)
print('학습 샘플 레이블 y_train[0]', y_train[0])

print('테스트 샘플 시퀀스 shape', x_test.shape)
print('테스트 샘플 레이블 shape', y_test.shape)
print('테스트 샘플 레이블 y_train[0]', y_test[0])


# 8) 모델 정의
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional #양방향(문장<->태그) 예측
from tensorflow.keras.optimizers import Adam

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=30, input_length=max_len, mask_zero=True))
model.add(Bidirectional(LSTM(200, return_sequences=True, dropout=0.5, recurrent_dropout=0.25))) #양방향 LSTM
model.add(TimeDistributed(Dense(tag_size, activation='softmax')))


# 9) 모델 설정
model.compile(loss='categorical_crossentropy', optimizer=Adam(0.01), metrics=['accuracy'])


# 10) 모델 학습
model.fit(x_train, y_train, batch_size=128, epochs=10)


# 11) 모델 평가
print('모델 평가:', model.evaluate(x_test, y_test)[1])


# 12) 모델 예측
## 시퀀스를 NER 태그로 변환
def sequences_to_tag(sequences):
    result = []
    for sequence in sequences:
        temp = []
        for pred in sequence:
            pred_index = np.argmax(pred)
            temp.append(index_to_ner[pred_index].replace('PAD', '0'))
        result.append(temp)
    return result

## 예측하기
y_pred = model.predict(x_test)
pred_tags = sequences_to_tag(y_pred)
test_tags = sequences_to_tag(y_test)

from seqeval.metrics import classification_report
print(classification_report(test_tags, pred_tags))


# 13) 새로운 유형의 문장 NER 예측
word_to_index = sent_tokenizer.word_index #학습한 문장의 단어 사전
new_sentence = '삼성전자 출시 스마트폰 오늘 애플 도전장 내민다.'.split()
new_x = []
for w in new_sentence:
    try:
        new_x.append(word_to_index.get(w, 1))
    except KeyError:
        #단어사전에 없는 단어가 출현했을 경우
        new_x.append(word_to_index['OOV'])

print('새로운 유형의 시퀀스:', new_x)
new_padding_seqs = preprocessing.sequence.pad_sequences([new_x], padding='post', value=0, maxlen=max_len)

# NER 예측
p = model.predict(np.array([new_padding_seqs[0]]))
p = np.argmax(p, axis=-1)
print('{:10}{:5}'.format('단어', '예측된 NER'))
print('-'*50)

for w, pred in zip(new_sentence, p[0]):
    print('{:10}{:5}'.format(w, index_to_ner[pred]))