class FindAnswer:
    def __init__(self, db):
        self.db = db

    def _make_query(self, intent_name, ner_tags):
        sql = 'SELECT * FROM chatbot_train_data'

        if intent_name != None and ner_tags == None:
            sql = sql + " WHERE intent = '{}'".format(intent_name)

        elif intent_name != None and ner_tags != None:
            where = ' WHERE intent = "%s"' % intent_name
            if len(ner_tags) > 0:
                where += ' AND ('
                for ne in ner_tags:
                    where += " ner LIKE '%{}%' OR ".format(ne)
                where = where[:-3] + ')'
            sql = sql + where
        # 동일한 답변이 2개 이상인 경우 랜덤으로 선택하도록
        sql = sql + 'ORDER BY rand() LIMIT 1'

        return sql


    def search(self, intent_name, ner_tags):

        sql = self.make_query(intent_name, ner_tags)
        answer = self.db.select_one(sql)

        #검색되는 답변이 없으면 의도명만 검색
        if answer is None:
            sql = self._make_query(intent_name, None)
            answer = self.db.select_one(sql)

        return (answer['answer'], answer['answer_image'])


    # NER 태그를 실제 입력된 단어로 변환
    def tag_to_word(self, ner_predicts, answer):
        for word, tag in ner_predicts:
            if tag == 'B_FOOD' or tag == 'B_DT' or tag == 'B_TI':
                answer = answer.replace(tag, word)
        answer = answer.replace('{', '')
        answer = answer.replace('}', '')

        return answer