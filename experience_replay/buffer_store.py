import pickle
import os, shutil 
 
import logging

logger = logging.getLogger("buffer_store")

logger.setLevel(logging.DEBUG)
format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

class BufferStore:
    def __init__(self, buffer_path, file_name ='buffer.pickle'):
        self.file_name  = os.path.join(buffer_path, file_name)
        self.backup_file_name = os.path.join(buffer_path, "backup_"+file_name)
 
    def file_backup(self):
        if os.path.isfile(self.file_name):
            shutil.copy(self.file_name, self.backup_file_name)
        

    def save(self, code_object, backup=True):
        if backup: self.file_backup()
        with open(self.file_name, 'wb') as handle:
            pickle.dump(code_object, handle)
        logger.info(f"buffer saved at {self.file_name}")