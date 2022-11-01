from konlpy.tag import Kkma

kkma = Kkma()
text = '아버지가 방에 들어갑니다.'

# 형태소 추출
morphs = kkma.morphs(text)
print(morphs)

# 형태소와 품사 태그 추출
pos = kkma.pos(text)
print(pos)

# 명사만 추출
nouns = kkma.nouns(text)
print(nouns)

# 문장 분리
s = kkma.sentences(text)
print(s)