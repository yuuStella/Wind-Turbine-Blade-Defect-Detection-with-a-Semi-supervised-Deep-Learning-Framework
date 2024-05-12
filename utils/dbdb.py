import pymysql.cursors

# Connect to the database
connection = pymysql.connect(host='localhost',
                             user='root',
                             password='123456',
                             database='jg_dataset',
                             cursorclass=pymysql.cursors.DictCursor)


def insert_data(image_name, location, state):
    with connection:
        with connection.cursor() as cursor:
            # Create a new record
            sql = "INSERT INTO `jg_info` (`imagename`, `location`, `state`) VALUES (%s, %s, %s)"
            cursor.execute(sql, (image_name, location, state))
        connection.commit()


if __name__ == '__main__':
    insert_data("11", "22", "33")
