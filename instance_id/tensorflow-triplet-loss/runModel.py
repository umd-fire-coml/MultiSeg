import pickle
import model.input_fn as input_fn
from model.utils import Params
import os

model_file = open('model_file', 'rb') 
estimator = pickle.load(model_file)
model_file.close()

dir = "data/mnist"
json_path = os.path.join("experiments/base_model", 'params.json')
params = Params(json_path)
res = estimator.evaluate(lambda: input_fn.train_input_fn(dir, params))
for key in res:
	print("{}: {}".format(key, res[key]))
