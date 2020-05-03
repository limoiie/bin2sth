import os

import fire


def append(folder, recurse, postfix):
    for folder, dirs, files in os.walk(folder):
        for f in files:
            f = os.path.join(folder, f)
            print(f'rename file {f} as {f}{postfix}')
            os.rename(f, f + postfix)
        if not recurse:
            return


if __name__ == '__main__':
    fire.Fire(append)
