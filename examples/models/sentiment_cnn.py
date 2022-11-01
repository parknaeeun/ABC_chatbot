import pandas as pd
import tensorflow as tf
from tensorflow.keras import preprocessing
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, Dropout, Conv1D, GlobalMaxPool1D, concatenate


# 1) 데이터 읽어오기
data = pd.read_csv('chatbot_data.csv', delimiter = ',')
features = data['Q'].tolist()
labels = data['label'].tolist()

print('feature ', features[:10])
print('label ', labels[:10])


# 2) 말뭉치 만들기
corpus = [preprocessing.text.text_to_word_sequence(text) for text in features]
# 테스트
print('corpus', corpus[:10])


# 3) 시퀀스 벡터 만들기
tokenizer = preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(corpus)
sequences= tokenizer.texts_to_sequences(corpus)
word_index = tokenizer.word_index # {'12시':4646, '땡':4647 ...}

print('sequences', sequences[:10])
print('sequence max', max([len(text) for text in sequences]))


# 4) 문장 길이 맞추기 padding
MAX_SEQ_LEN = max([len(text) for text in sequences]) # 15 | 교체될 일 없는 변수는 대문자로
padded_seqs = preprocessing.sequence.pad_sequences(sequences, maxlen=MAX_SEQ_LEN, padding='post') #post: sequence 뒤를 0으로 채움

print('padded_seqs', padded_seqs[:10])


# 5) 학습(70), 검증(20), 테스트셋(10) 생성
ds = tf.data.Dataset.from_tensor_slices((padded_seqs, labels))
ds = ds.shuffle(len(features)) # feature의 개수 사이에서 랜덤하게 추출해서 다시 담기

train_size = int(len(padded_seqs) * 0.7)
val_size = int(len(padded_seqs) * 0.2)
test_size = int(len(padded_seqs) * 0.1)

train_ds = ds.take(train_size).batch(20)
val_ds = ds.skip(train_size).take(val_size).batch(20)
test_ds = ds.skip(train_size+val_size).take(test_size).batch(20)


# 6) 모델 구성
## 하이퍼 파라미터 설정
dropout_prob = 0.5
EMB_SIZE = 128  #embedded size
EPOCH = 5
VOCAB_SIZE = len(word_index) + 1  #13398건

## CNN 모델 정의
input_layers = Input(shape=(MAX_SEQ_LEN, ))
embedding_layer = Embedding(VOCAB_SIZE, EMB_SIZE, input_length=MAX_SEQ_LEN)(input_layers) #층 연결
dropout_emb = Dropout(rate=dropout_prob)(embedding_layer)

## CNN 모델 병렬 처리
conv1 = Conv1D(filters=128, kernel_size=3, padding='valid', activation=tf.nn.relu)(dropout_emb)
pool1 = GlobalMaxPool1D()(conv1)

conv2 = Conv1D(filters=128, kernel_size=4, padding='valid', activation=tf.nn.relu)(dropout_emb)
pool2 = GlobalMaxPool1D()(conv2)

conv3 = Conv1D(filters=128, kernel_size=5, padding='valid', activation=tf.nn.relu)(dropout_emb)
pool3 = GlobalMaxPool1D()(conv3)

## 3, 4, 5 특징 추출한 CNN 합치기
concat = concatenate([pool1, pool2, pool3])
hidden = Dense(128, activation=tf.nn.relu)(concat)
dropout_hidden = Dropout(rate=dropout_prob)(hidden)
logits = Dense(3, name='logits')(dropout_hidden)  #0, 1, 2 세 개로 예측
predictions = Dense(3, activation=tf.nn.softmax)(logits)


# 7) 모델 생성 및 컴파일
model = Model(inputs=input_layers, outputs=predictions)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) #인코딩하지 않은 넘버링 값을 조정해 주는 함수


# 8) 모델 학습
model.fit(train_ds, validation_data=val_ds, epochs=EPOCH, verbose=1) #verbose: 출력 형식


# 9) 모델 평가
loss, accuracy = model.evaluate(test_ds, verbose=1)
print('accuracy: %f' % (accuracy*100))
print('loss: %f' % (loss))


# 10) 모델 저장
model.save('cnn_model.h5')