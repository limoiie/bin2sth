from src.preprocess import *
from src.database import *
from pymongo import MongoClient
from src.ida.code_elements import *


client = MongoClient('localhost', 27017)
db = client.test_database

args = BinArgs(['lua'], [['5.2.3']], ['gcc'], [[5]],
               [Arch('metapc', 'b32', 'le')], ['O0'], ['sub3'])

data = load_cbow_data_end(db, args, args, 3)
