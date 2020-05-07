import fire
from gridfs import GridFS

from src.database.database import get_database_client
from src.database.program_dao import ProgramDAO
from src.ida.as_json import AsJson
from src.ida.code_elements import Program


def store_into_db(prog_file):
    prog = AsJson.load(prog_file, Program)
    db = get_database_client().test_database
    prog_dao = ProgramDAO(db, GridFS(db))
    prog_dao.store(prog)


if __name__ == '__main__':
    fire.Fire(store_into_db)
