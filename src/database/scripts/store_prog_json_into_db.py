import os
import re

import fire
from gridfs import GridFS

from src.database.database import get_database_client, get_database
from src.database.program_dao import ProgramDAO
from src.ida.as_json import AsJson
from src.ida.code_elements import Program
from src.utils.logger import get_logger

logger = get_logger('store into db')


def store_into_db(path):
    db = get_database()
    if os.path.isfile(path):
        store_file_into_db(db, path)
        return

    for filename in os.listdir(path):
        if re.match('.*.tmp.json', filename):
            fullpath = os.path.join(path, filename)
            store_file_into_db(db, fullpath)


def store_file_into_db(db, filepath):
    logger.info(F'Storing proginfo at {filepath} into db...')
    prog = AsJson.load(filepath, Program)
    prog_dao = ProgramDAO(db, GridFS(db))
    prog_dao.store(prog)
    logger.info(F'Finish Storing')


if __name__ == '__main__':
    fire.Fire(store_into_db)
