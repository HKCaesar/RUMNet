import models
from data import gen, test_data
from keras.callbacks import ModelCheckpoint, EarlyStopping


shape = (128,128)
batch_size = 1


# Load model
print('\n')
print('-'*30)
print('Loading model...')
print('-'*30)  
model = models.unet(shape, models.res_block_basic, models.Activation('relu'), 0, False)  
#model = models.get_unet(shape)    
callbacks = [EarlyStopping(monitor='val_loss', patience=3, verbose=0),
                ModelCheckpoint('./weights/weights.hdf5', 
                                monitor='val_loss', save_best_only=True)]

# Training    
print('\n')
print('-'*30)
print('Begin training...')
print('-'*30)
dgen = gen(shape, batch_size)
model.fit_generator(generator=dgen, 
                    samples_per_epoch=batch_size*50, nb_epoch=25,
                    validation_data=gen(shape, 100),
                    nb_val_samples=100,
                    verbose=1, callbacks=callbacks)