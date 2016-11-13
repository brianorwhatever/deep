#path = "data/state-farm/"
path = "data/state-farm/sample/"
import utils; reload(utils)
from utils import *

batch_size=64
from vgg16 import Vgg16
vgg = Vgg16()

batches = vgg.get_batches(path+'train', batch_size=batch_size)
val_batches = vgg.get_batches(path+'valid', batch_size=batch_size*2)
vgg.finetune(batches)

# Setting all dense layers to trainable
def set_trainable():
    layers = vgg.model.layers
    # Get the index of the first dense layer...
    first_dense_idx = [index for index,layer in enumerate(layers) if type(layer) is Dense][0]
    # ...and set this and all subsequent layers to trainable
    for layer in layers[first_dense_idx:]:
        layer.trainable=True

def run():
    vgg.model.optimizer.lr = 0.00001
    vgg.fit(batches, val_batches, nb_epoch=5)
    vgg.model.optimizer.lr = 0.0001
    vgg.fit(batches, val_batches, nb_epoch=5)
    vgg.model.optimizer.lr = 0.001
    vgg.fit(batches, val_batches, nb_epoch=5)
    vgg.model.optimizer.lr = 0.01
    vgg.fit(batches, val_batches, nb_epoch=5)
    vgg.model.optimizer.lr = 0.1
    vgg.fit(batches, val_batches, nb_epoch=5)

def save(file):
    vgg.model.save_weights(file)

# init
set_trainable()
run()
save(path+'results/aviary_vgg_003.h5')
