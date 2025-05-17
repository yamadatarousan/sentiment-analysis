import mysql.connector

connection = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="text_analysis_db"
)
cursor = connection.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS analysis_history (
        id INT AUTO_INCREMENT PRIMARY KEY,
        text TEXT,
        sentiment VARCHAR(50),
        topic VARCHAR(50),
        sentiment_confidence FLOAT,
        topic_confidence FLOAT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
""")
cursor.execute(
    "INSERT INTO analysis_history (text, sentiment, topic, sentiment_confidence, topic_confidence) VALUES (%s, %s, %s, %s, %s)",
    ("テスト", "ポジティブ", "映画", 0.95, 0.90)
)
connection.commit()
cursor.execute("SELECT * FROM analysis_history")
print(cursor.fetchall())
cursor.close()
connection.close()