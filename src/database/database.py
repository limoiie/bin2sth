from pymongo import MongoClient

from src.utils.logger import get_logger

logger = get_logger('database')


def get_database_client():
    client = MongoClient('localhost', 27017)
    return client


def get_database():
    return get_database_client().test_database
