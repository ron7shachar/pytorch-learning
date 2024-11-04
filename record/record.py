import pickle

class Record:
    def __init__(self,name):
        self.name = name
        self.file_path = f'./record/{name}.pkl'

    def set_record(self,record):
        with open(self.file_path, 'wb') as f:
            pickle.dump(record, f)

    def get_record(self):
        try:
            # ניסיון לטעון את האובייקט מהקובץ
            with open(self.file_path, 'rb') as f:
                return pickle.load(f)


        except FileNotFoundError:
            return None


