from pymongo import MongoClient
from .config import Config

client = MongoClient(Config.MONGO_URI)
db = client.get_database()   # database name is taken from the URI path