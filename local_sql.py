# from mysql import connector
import mysql.connector
import pandas as pd
import os

# pip install mysql-connector-python==8.0.21 高版本有问题

def connect_as_root(host, user, password):
    connection = None
    try:
        connection = mysql.connector.connect(host=host, user=user, passwd=password)
        print("SUccessfully connected to mysql as root.")
    except Exception as err:
        print(f"Error: '{err}'")


def create_db_connection(host_name, user_name, user_password, db_name):
    connection = None
    try:
        # 配置信息
        # config = {
        #     'host': host_name,
        #     'port': 3306,
        #     'user': user_name,
        #     'password': user_password,
        #     'database': db_name,
        #     'charset': 'utf8'
        # }
        # 连接数据库
        # con=mysql.connector.connect(host='localhost',port=3306,user='root',
        #                         password='root',database='test',charset='utf8')
        # connection = connector.connect(**config)
        # connection =mysql.connector.connect(host=host_name,user=user_name, password=user_password, database=db_name)
        connection = mysql.connector.connect(
            host=host_name,
            user=user_name,
            password=user_password,
            # auth_plugin='caching_sha2_password',
            # caching_sha2_password=user_password,mysql_native_password
            database=db_name,
            charset='utf8',
            buffered=True
        )
        print("MySQL Database connection successful")
    except Exception as err:
        print(f"Error: '{err}'")

    return connection


def execute_query(connection, query):
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        connection.commit()
        print("Query successful")
    except Exception as err:
        print(f"Error: '{err}'")


def read_csv_into_mysql(connection, path, tablename):
    df = pd.read_csv(path)

    cols = df.columns

    drop_query = "DROP TABLE IF EXISTS " + tablename

    execute_query(connection, drop_query)

    query = "CREATE TABLE IF NOT EXISTS " + tablename + " ("
    for i in range(len(cols)):
        if i != len(cols) - 1:
            query += (cols[i] + " varchar(128) DEFAULT NULL,")
        else:
            query += (cols[i] + " varchar(128) DEFAULT NULL);")

    execute_query(connection, query)

    for i in range(df.shape[0]):
        insert_query = "INSERT INTO " + tablename + "("
        for j in range(len(cols)):
            if j != len(cols) - 1:
                insert_query += (cols[j] + ",")
            else:
                insert_query += (cols[j] + ")")

        insert_query += " VALUES ("
        cur_row = df.iloc[[i]]
        for j in range(len(cols)):
            print(str(cur_row[cols[j]].values[0]))
            if j != len(cols) - 1:
                insert_query += ("'" + str(cur_row[cols[j]].values[0]) + "'" + ",")
            else:
                insert_query += ("'" + str(cur_row[cols[j]].values[0]) + "'" + ");")
        execute_query(connection, insert_query)


def read_table_from_sql(connection, table):
    select_query = "SELECT * FROM " + table

    mycursor = connection.cursor()

    mycursor.execute(select_query)

    myresult = mycursor.fetchall()

    return myresult


connection = create_db_connection("192.168.1.151", "root", "CtMysql@123", "mysql")

this_path = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(this_path, 'datasets/anomaly/raw_data/yahoo_sub_5.csv')

# read_csv_into_mysql(connection, path, "yahoo_sub_5")

yahoo_sub_5_data = read_table_from_sql(connection, "yahoo_sub_5")

for i in yahoo_sub_5_data:
    print(i)