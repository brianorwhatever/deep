path = "data/state-farm/"
#path = "data/state-farm/sample/"
import utils; reload(utils)
from utils import *

batch_size=64
from vgg16 import Vgg16
vgg = Vgg16()

batches = vgg.get_batches(path+'train', batch_size=batch_size)
val_batches = vgg.get_batches(path+'valid', batch_size=batch_size*2)
vgg.finetune(batches)

layers = vgg.model.layers
# Get the index of the first dense layer...
first_dense_idx = [index for index,layer in enumerate(layers) if type(layer) is Dense][0]
# ...and set this and all subsequent layers to trainable
for layer in layers[first_dense_idx:]:
    layer.trainable=True

vgg.model.optimizer.lr = 0.0007
vgg.fit(batches, val_batches, nb_epoch=5)
vgg.model.optimizer.lr = 0.003
vgg.fit(batches, val_batches, nb_epoch=5)

vgg.model.save_weights(path+'results/first_attempt.h5')
