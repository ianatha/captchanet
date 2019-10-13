from keras.models import model_from_json

def load_model(path):
    json_file = open(path + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    model = model_from_json(loaded_model_json)
    model.load_weights(path + '.h5')

    return model