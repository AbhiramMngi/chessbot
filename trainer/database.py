from peewee import *

db = SqliteDatabase("positions.db")

class Evaluations(Model):
    id = IntegerField()
    fen = TextField()
    binary = BlobField()
    eval = FloatField()

    class Meta:
        database = db
        # table_name = "evaluations"s

db.connect()
db_close = lambda : db.close()