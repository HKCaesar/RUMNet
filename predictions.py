import models
from data import gen, test_data
import matplotlib.pyplot as plt

shape = (256,256)


# Load model
print('\n')
print('-'*30)
print('Loading model...')
print('-'*30)  
model = models.unet(shape, models.res_block_basic, models.Activation('relu'), 0, False)   
#model = models.get_unet(shape)
model.load_weights('./weights/weights.hdf5')

# Look at sample predictions
print('\n')
print('-'*30)
print('Sample predictions...')
print('-'*30)
Xt = test_data(shape)
Yt = model.predict(Xt, verbose=1)

for i in range(Xt.shape[0]):
    plt.subplot(1,2,1)
    plt.imshow(Xt[i,0,:,:])
    plt.subplot(1,2,2)
    plt.imshow(Yt[i,0,:,:])
    plt.show()