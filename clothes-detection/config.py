import mysql.connector
from datetime import datetime

# Insert into database
import mysql.connector
from datetime import datetime

def insert_into_database(item_name, color, image_path):
    try:
        mydb = mysql.connector.connect(
            host="localhost",
            user="anthea",
            password="anthea.rpi",
            database="project"
        )
        cursor = mydb.cursor()
        query = """
        INSERT INTO clothes (Clothing_Item, Colour, Date_Added, image_path)
        VALUES (%s, %s, %s, %s)
        """
        cursor.execute(query, (item_name, color, datetime.now().date(), image_path))
        mydb.commit()
        print(f"Inserted: {item_name}, {color}, {image_path}")
    except mysql.connector.Error as err:
        print(f"Database error: {err}")
    finally:
        if mydb.is_connected():
            cursor.close()
            mydb.close()



def fetch_data():
    try:
        mydb = mysql.connector.connect(
            host="localhost",
            user="anthea",
            password="anthea.rpi",
            database="project"
        )
        cursor = mydb.cursor()
        cursor.execute("SELECT `Index`, Clothing_Item, Colour, Date_Added, image_path FROM clothes")
        data = cursor.fetchall()
        return data
    except mysql.connector.Error as err:
        print(f"Database error: {err}")
        return []
    finally:
        if mydb.is_connected():
            cursor.close()
            mydb.close()


# Remove from database
def remove_from_database(item_index):
    try:
        mydb = mysql.connector.connect(
            host="localhost",
            user="anthea",
            password="anthea.rpi",
            database="project"
        )
        cursor = mydb.cursor()
        query = "DELETE FROM clothes WHERE `Index` = %s"
        cursor.execute(query, (item_index,))
        mydb.commit()
        print(f"Removed item with Index: {item_index}")
    except mysql.connector.Error as err:
        print(f"Error removing item: {err}")
    finally:
        if mydb.is_connected():
            cursor.close()
            mydb.close()