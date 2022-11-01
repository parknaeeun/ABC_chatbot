import pymysql

# DB_HOST = 127.0.0.1
# DB_PORT = 3306
# DB_DATABASE = homestead
# DB_USERNAME = homestead
# DB_PASSWORD = secret


db = None
try:
    db = pymysql.connect(
        host='127.0.0.1',
        user='homestead',
        password='secret',
        db='homestead',
        charset='utf8'
    )
    print('DB 연결 성공')

except Exception as e:
    print(e)

finally:
    if db is not None:
        db.close()
        print('DB 연결 닫기 성공')