import fire
from IPython import embed
from gridfs import GridFS

# noinspection PyUnresolvedReferences
import src.training.args

from src.database.dao import Dao
from src.database.database import get_database


def start_console(bean):
    db = get_database()
    dao = Dao.instance(bean, db, GridFS(db))
    print(f'Welcome to starting {dao}, {dao.count({})}!')
    embed()
    print(f'Thanks for using~')


if __name__ == '__main__':
    fire.Fire(start_console)
