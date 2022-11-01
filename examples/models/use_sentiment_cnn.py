import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import preprocessing


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


# 5) 테스트용 데이터셋 생성
ds = tf.data.Dataset.from_tensor_slices((padded_seqs, labels))
ds = ds.shuffle(len(features))

test_ds = ds.take(2000).batch(20)


# 6) CNN 모델 불러 오기
model =load_model('cnn_model.h5')
model.summary()
model.evaluate(test_ds, verbose=2)


# 7) 테스트용 데이터셋 확인
picks = 929
print('말뭉치:', corpus[picks])
print('단어 시퀀스:', padded_seqs[picks])
print('문장 분류:', labels[picks])


# 8) 모델 예측
picks = [929]
predict = model.predict(padded_seqs[picks])
predict_class = tf.math.argmax(predict, axis=1)
print('감정 예측 점수:', predict)
print('감정 예측 클래스:', predict_class.numpy())