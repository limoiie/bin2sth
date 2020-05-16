#!/usr/bin/env bash

# parse .tmp.json file and store into database
python src/database/scripts/store_prog_json_into_db.py --path \
  "$HOME/Downloads/opensource/asm2vec_rebuild/bin/"
