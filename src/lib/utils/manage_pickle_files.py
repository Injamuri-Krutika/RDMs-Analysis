import pickle


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f, encoding='latin1')


def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
