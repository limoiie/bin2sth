from gridfs import GridFS
import pandas as pd

from src.database.database import get_database_client
from src.database.program_dao import ProgramDAO
from src.ida.code_elements import Function, Block, Arch, Program


tmp_folder = 'src/training/.tmp/nmt_inspired'

db = get_database_client().test_database
prog_dao = ProgramDAO(db, GridFS(db))


def load_from(folder):
    TRAIN_CSV = f'{folder}/train_set_O2.csv'
    TEST_CSV = f'{folder}/test_set_O2.csv'

    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    return pd.concat([train_df, test_df])


# NOTE: tmp code!!! Used to transform nmt official data into our db
def make_x86(blks, x86, eq):
    funcs = []
    arch_name = 'x86' if x86 else 'arm'
    for i, blk in enumerate(blks):
        func = Function()
        func.label = f'func-{i:06}'
        if eq.iat[i] != 1:
            func.label += arch_name
        func.segm = '.text'
        stmts = str(blk).upper().split()
        block = Block()
        block.label = 'i'
        block.src = stmts
        func.blocks.append(block)
        funcs.append(func)

    arch = Arch(arch_name, 'b32', 'be')
    b = Program(
        ('tmp_prog', '0.1'), ('gcc', '5'), arch, 'O3', None, funcs, None)
    prog_dao.store(b)


def convert_and_store():
    df = load_from(tmp_folder)
    make_x86(df['x86_bb'], True, df['eq'])
    make_x86(df['arm_bb'], False, df['eq'])
