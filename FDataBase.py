import sqlite3
import time
import math
from flask import url_for
import re


class FDataBase:
    # db - link to db and create cur to work with tables
    def __init__(self, db):
        self.__db = db
        self.__cur = db.cursor()

    # take all records from menu
    # we need try block to avoid errors
    def getMenu(self):
        sql = """SELECT * FROM mainmenu"""
        try:
            # if all good then exe sql reqs and save it in res obj
            self.__cur.execute(sql)
            res = self.__cur.fetchall()
            if res:
                return res
        except:
            print('Read db error')
        return []

    def addPost(self, title, text, url):
        try:
            self.__cur.execute(f"SELECT COUNT() as 'count' FROM posts WHERE url LIKE '{url}'")
            res = self.__cur.fetchone()
            if res["count"] > 0:
                print("Post with such url already exists")
                return False

            base = url_for("static", filename='images_html')

            text = re.sub(r"(?P<tag><img\s+[^>]*src=)(?P<quote>[\"'])(?P<url>.+?)(?P=quote)>",
                          "\\g<tag>" + base + "/\\g<url>>", text)

            # take an adding post time
            tm = math.floor(time.time())
            self.__cur.execute("INSERT INTO posts VALUES(NULL, ?, ?, ?, ?)", (title, text, url, tm))
            self.__db.commit()
        except sqlite3.Error as e:
            print("Add post error in db "+str(e))
            return False
        return True

    def getPost(self, alias):
        try:
            self.__cur.execute(f"SELECT title, text FROM posts WHERE url LIKE '{alias}' LIMIT 1")
            res = self.__cur.fetchone()
            if res:
                return res
        except sqlite3.Error as e:
            print("Taking post from bd error"+str(e))

        return (False, False)

    def getPostsAnonce(self):
        try:
            self.__cur.execute(f"SELECT id, title, text, url FROM posts ORDER BY time DESC")
            res = self.__cur.fetchall()
            if res:
                return res
        except sqlite3.Error as e:
            print("taking posts from bd error"+str(e))

        return []
