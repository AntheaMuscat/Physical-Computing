import mysql.connector

mydb = mysql.connector.connect(
  host="localhost",
  user="anthea",
  password="anthea.rpi"
)

print(mydb)