from config.DatabaseConfig import *
from utils.Database import Database
from utils.Preprocess import Preprocess

# 1) 전처리 객체 생성(단어 사전 준비)
p = Preprocess(word2index_dic='../train_tools/dict/chatbot_dict.bin',
               userdic='../utils/user_dic.tsv')

# 2) 질문/답변 학습 DB 연결 객체 생성 (DB 접속 준비)
db = Database(host=DB_HOST, user=DB_USER,
              password=DB_PASSWORD, db_name=DB_NAME)
db.connect()

# 3) 원문 준비
query = '탕수육 1개 주세요'

# 4) 의도 파악
from models.intent.intentModel import IntentModel
intent = IntentModel(model_name='../models/intent/intent_model.h5', preprocess=p)

predict = intent.predict_class(query)
intent_name = intent.labels[predict]

# 5) 개체명 인식
from models.ner.NerModel import NerModel
ner = NerModel(model_name='../models/ner/ner_model.h5', preprocess=p)
predicts = ner.predict(query)
ner_tags = ner.predict_tags(query)

print('질문:', query)
print('=' * 100)
print('의도 파악:', intent_name)
print('개체명 인식:', predicts)
print('답변 검색에 필요한 NER:', ner_tags)


# 6) 답변 검색
from utils.FindAnswer import FindAnswer

try:
    f = FindAnswer(db)
    answer_text, answer_image = f.search(intent_name, ner_tags)
    answer = f.tag_to_word(predicts, answer_text)

except:
    answer = '죄송해요 무슨 말인지 모르겠어요'

print('답변:', answer)
db.close()