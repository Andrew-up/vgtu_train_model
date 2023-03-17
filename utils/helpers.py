
import os
import time
from datetime import datetime

from definitions import MODEL_H5_PATH

file_info = {
    'name_file': None,
    'date': None,
    'path': None
}


def delete_legacy_models_and_zip(max_files_legacy: int):
    list_h5: file_info = []
    list_zip: file_info = []
    sum_file_delete = 0
    for root, dirs, files in os.walk(MODEL_H5_PATH):
        for file in files:
            if file.endswith('.h5'):
                a = os.stat(os.path.join(MODEL_H5_PATH, file))
                created = time.ctime(a.st_atime)
                # print(type(created))
                list_h5.append({'name_file': file, 'date': datetime.strptime(created, '%c'),
                                'path': os.path.join(MODEL_H5_PATH, file)})

            if file.endswith('.zip'):
                a = os.stat(os.path.join(MODEL_H5_PATH, file))
                created = time.ctime(a.st_atime)
                # print(type(created))
                list_zip.append({'name_file': file, 'date': datetime.strptime(created, '%c'),
                                 'path': os.path.join(MODEL_H5_PATH, file)})

    newlist_h5 = sorted(list_h5, key=lambda d: d['date'], reverse=False)
    newlist_zip = sorted(list_zip, key=lambda d: d['date'], reverse=False)
    if len(newlist_h5) > max_files_legacy:
        summ_deletefiles = len(newlist_h5) - max_files_legacy
        for i in newlist_h5[0:summ_deletefiles]:
            if os.path.exists(i['path']):
                os.remove(i['path'])
                sum_file_delete += 1
                print(f"REMOVE FILE: {i['path']}")

    if len(newlist_zip) > max_files_legacy:
        summ_deletefiles = len(newlist_zip) - max_files_legacy
        for i in newlist_zip[0:summ_deletefiles]:
            print(i['path'])
            if os.path.exists(i['path']):
                os.remove(i['path'])
                sum_file_delete += 1
                print(f"REMOVE FILE: {i['path']}")
    return sum_file_delete
