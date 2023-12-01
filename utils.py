import pickle

def save_pickle(py_object, file_path):
    pickle_file = open(file_path, 'ab')
    pickle.dump(py_object, pickle_file)                    
    pickle_file.close()

def load_pickle(file_path):
    pickle_file = open(file_path, 'rb')    
    py_object = pickle.load(pickle_file)
    pickle_file.close()
    return py_object