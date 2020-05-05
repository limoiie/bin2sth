import json

from gridfs import GridFS

encoding = 'utf-8'


class Dao:
    def __init__(self, db, fs: GridFS):
        self.db, self.fs = db, fs

    def _put_into_fs(self, obj):
        return self.fs.put(json.dumps(obj), encoding=encoding)

    def _get_from_fs(self, f_id):
        return json.loads(self.fs.get(f_id).read().decode(encoding=encoding))
