from konlpy.tag import Komoran

komoran = Komoran(userdic='user_dic.tsv')
text = '우리 챗봇은 엔엘피를 좋아해.'

# 형태소와 품사 태그 추출
pos = komoran.pos(text)
print(pos)