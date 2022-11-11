import pymysql
import pymysql.cursors
import logging

# 데이터 베이스 제어 모듈
class Database:
    def __init__(self, host, user, password, db_name, charset='utf8'):
        self.host = host
        self.user = user
        self.password = password
        self.db_name = db_name
        self.charset = charset
        self.conn = None


    def connect(self):
        if self.conn != None:
            return

        self.conn = pymysql.connect(
            host = self.host,
            user = self.user,
            password = self.password,
            db = self.db_name,
            charset = self.charset
        )


    # DB 연결 종료
    def close(self):
        if self.conn is None:
            return

        if not self.conn.open:
            self.conn = None
            return

        self.conn.close()
        self.conn = None


    # SQL 구문 실행
    def execute(self, sql):
        last_row_id = -1

        try:
            with self.conn.cursor() as cursor:
                cursor.execute(sql)
            self.conn.commit()
            last_row_id = cursor.lastrowid

        except Exception as ex:
            logging.error(ex)

        finally:
            return last_row_id


    # SELECT 구문 실행 후에 1개의 데이터 row만 불러옴
    def select_one(self, sql):
        result = None
        try:
            with self.conn.cursor(pymysql.cursors.DictCursor) as cursor:
                cursor.execute(sql)
                result = cursor.fetchone()

        except Exception as ex:
            logging.error(ex)

        finally:
            return last_row_id


    # SELECT 구문 실행 후에 전체 데이터 row를 불러옴
    def select_all(self, sql):
        result = None
        try:
            with self.conn.cursor(pymysql.cursors.DictCursor) as cursor:
                cursor.execute(sql)
                result = cursor.fetchall()

        except Exception as ex:
            logging.error(ex)

        finally:
            return last_row_id