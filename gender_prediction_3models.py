import mat73
import numpy as np # linear algebra
import keras
from keras.layers import Input
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, AveragePooling2D, MaxPooling2D
from tensorflow.keras.layers import Conv1D, Conv2D, SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization, LayerNormalization, Flatten 
from tensorflow.keras.layers import Add, Concatenate, Lambda, Input, Permute
from tensorflow.keras.constraints import max_norm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
import math
import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense
from tensorflow.keras.layers import multiply, Permute, Concatenate, Conv2D, Add, Activation, Lambda
from tensorflow.keras.layers import Dropout, MultiHeadAttention, LayerNormalization, Reshape
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import LearningRateScheduler


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K
from sklearn.metrics import confusion_matrix
import math
import matplotlib.pyplot as plt

from keras.layers import Concatenate, Dense, LSTM, Input, concatenate

# Gender labels array contain the gender data for each subject
# 0 = male, 1 = female
gender_labels = [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0]


#Temporal convolutional (TC) block used in the ATCNet model
def TCN_block(input_layer,input_dimension,depth,kernel_size,filters,dropout,activation='relu'):
       
    block = Conv1D(filters,kernel_size=kernel_size,dilation_rate=1,activation='linear',
                   padding = 'causal',kernel_initializer='he_uniform')(input_layer)
    block = BatchNormalization()(block)
    block = Activation(activation)(block)
    block = Dropout(dropout)(block)
    block = Conv1D(filters,kernel_size=kernel_size,dilation_rate=1,activation='linear',
                   padding = 'causal',kernel_initializer='he_uniform')(block)
    block = BatchNormalization()(block)
    block = Activation(activation)(block)
    block = Dropout(dropout)(block)
    if(input_dimension != filters):
        conv = Conv1D(filters,kernel_size=1,padding='same')(input_layer)
        added = Add()([block,conv])
    else:
        added = Add()([block,input_layer])
    out = Activation(activation)(added)
    
    for i in range(depth-1):
        block = Conv1D(filters,kernel_size=kernel_size,dilation_rate=2**(i+1),activation='linear',
                   padding = 'causal',kernel_initializer='he_uniform')(out)
        block = BatchNormalization()(block)
        block = Activation(activation)(block)
        block = Dropout(dropout)(block)
        block = Conv1D(filters,kernel_size=kernel_size,dilation_rate=2**(i+1),activation='linear',
                   padding = 'causal',kernel_initializer='he_uniform')(block)
        block = BatchNormalization()(block)
        block = Activation(activation)(block)
        block = Dropout(dropout)(block)
        added = Add()([block, out])
        out = Activation(activation)(added)
    return out

# Function to define EEGNeX model
def EEGNeX_8_32(n_timesteps, n_features, n_outputs):
    input_main = Input((n_timesteps, n_features, 1))
    block1s = Permute((3, 2, 1))(input_main)
    block1 = Conv2D(filters=8, kernel_size=(1, 32), use_bias=False, padding='same', data_format="channels_first")(block1s)
    block1 = LayerNormalization()(block1)
    block1 = Activation(activation='elu')(block1)

    block2 = Conv2D(filters=32, kernel_size=(1, 32), use_bias=False, padding='same', data_format="channels_first")(block1)
    block2 = LayerNormalization()(block2)
    block2 = Activation(activation='elu')(block2)

    block3 = DepthwiseConv2D(kernel_size=(n_features, 1), depth_multiplier=2, use_bias=False, depthwise_constraint=max_norm(1.), data_format="channels_first")(block2)
    block3 = LayerNormalization()(block3)
    block3 = Activation(activation='elu')(block3)
    block3 = AveragePooling2D(pool_size=(1, 4), padding='same', data_format="channels_first")(block3)
    block3 = Dropout(0.5)(block3)

    block4 = Conv2D(filters=32, kernel_size=(1, 16), use_bias=False, padding='same', dilation_rate=(1, 2), data_format='channels_first')(block3)
    block4 = LayerNormalization()(block4)
    block4 = Activation(activation='elu')(block4)

    block5 = Conv2D(filters=8, kernel_size=(1, 16), use_bias=False, padding='same', dilation_rate=(1, 4), data_format='channels_first')(block4)
    block5 = LayerNormalization()(block5)
    block5 = Activation(activation='elu')(block5)
    block5 = Dropout(0.5)(block5)
    block5 = Lambda(lambda x: x[:,:,-1,:])(block5)

    # ADDED TCN
    block5 = TCN_block(input_layer = block5, input_dimension = 8, depth = 1,
                    kernel_size = 4, filters = 24, 
                    dropout = 0.3, activation = 'elu')

      
    
        # Get feature maps of the last sequence
    block5= Lambda(lambda x: x[:,-1,:])(block5)
    block5 = Flatten()(block5)
    block5 = Dense(n_outputs, kernel_constraint=max_norm(0.25))(block5)
    block5 = Activation(activation='softmax')(block5)
    return Model(inputs=input_main, outputs=block5)

# Load and preprocess data
vibration_dict = mat73.loadmat('/scratch/ac9429/gender_classification/dataUrg_sub_200Hz.mat')
vibration_data = vibration_dict['data']['data']
print(np.shape(vibration_data), 'x', len(vibration_data))

def scheduler(epoch, lr):
    if epoch < 50:
        return lr
    if epoch < 100 and epoch >= 50:
        return lr / 2
    if epoch >= 100:
        return lr / 10

nSub = 31  # number of subjects
bs_t = 64  # batch size
epochs = 400
lr = 0.0009
nb_classes = 2
chans = 60
samples = 400

urgency_categories = ['low', 'medium', 'high']
confx_all = {category: np.zeros((nSub, nb_classes, nb_classes)) for category in urgency_categories}
scores_all = {category: [] for category in urgency_categories}


# misclassification of majority class comes with a penalty of 0.738
# misclassification of minority class comes with a penalty of 1.55
# formula used to calculate class weights: (total number of samples)/(number of classes*number of samples in class)

class_weights = {0: 0.738, 1: 1.55}

for category_idx, category in enumerate(urgency_categories):
    print(f'Training for {category} urgency category')

    # Extract data for the current urgency category
    urgency_data = [vibration_data[i][category_idx][:, :, :36] for i in range(nSub)]

    # Creating gender labels for each trial
    gender_trials = []
    for idx, trials in enumerate(urgency_data):
        gender_label = np.full((1, trials.shape[2]), gender_labels[idx])
        gender_trials.append(gender_label)

    for sub in range(nSub):
        print(f'Subject: #{sub}')
        # Split train and test - time
        X_test_t = urgency_data[sub]

        x_train_t = urgency_data[:sub] + urgency_data[sub + 1:]  # all other subjects as train data
        x_train_t = np.concatenate(x_train_t, axis=2)

        # Concatenate the gender labels for all trials of the training set
        y_train_t = gender_trials[:sub] + gender_trials[sub + 1:]
        y_train_t = np.concatenate(y_train_t, axis=1).flatten()

        # Extract gender labels for the test subject
        Y_test_t = gender_trials[sub].flatten()

        X_test_t = np.swapaxes(X_test_t, 0, 2)
        X_test_t = np.swapaxes(X_test_t, 1, 2)

        X_train_t = np.swapaxes(x_train_t, 0, 2)
        X_train_t = np.swapaxes(X_train_t, 1, 2)

        # Add kernels dimension to X_train_t & X_test_t => np.newaxis / np.expand_dims
        X_test_t = np.expand_dims(X_test_t, axis=3)  # (80, 59, 1400, 1)
        X_train_t = np.expand_dims(X_train_t, axis=3)  # (2487, 59, 1400, 1)

        Y_test_t = to_categorical(Y_test_t, nb_classes)  # Updated for gender classification
        Y_train_t = to_categorical(y_train_t, nb_classes)

        x_train_nex = np.swapaxes(X_train_t, 1, 2)
        x_test_nex = np.swapaxes(X_test_t, 1, 2)

        X_train_t = np.swapaxes(X_train_t, 1, 3)
        X_train_t = np.swapaxes(X_train_t, 2, 3)

        X_test_t = np.swapaxes(X_test_t, 1, 3)
        X_test_t = np.swapaxes(X_test_t, 2, 3)

        print(np.shape(x_train_nex))
        print(np.shape(x_test_nex))

        model_nex = EEGNeX_8_32(samples, chans, nb_classes)

        opt_nex = Adam(learning_rate=lr)

        LR_callback = LearningRateScheduler(scheduler)

        model_nex.compile(loss='categorical_crossentropy', optimizer=opt_nex, metrics=['accuracy'])

        callbacks = [LR_callback]
        print(f'Epochs from 1 to 200')

        history_nex = model_nex.fit(
            x_train_nex, 
            Y_train_t, 
            batch_size=bs_t, 
            epochs=epochs, 
            verbose=0, 
            class_weight=class_weights, 
            callbacks=callbacks
        )

        probs_nex = model_nex.predict(x_test_nex)
        preds_nex = probs_nex.argmax(axis=-1)
        acc_nex = np.mean(preds_nex == Y_test_t.argmax(axis=-1))
        print(f'Nex: {acc_nex} %')

        confx_all[category][sub, :, :] = confusion_matrix(Y_test_t.argmax(axis=-1), preds_nex)

        scores_all[category].append(acc_nex)

for category in urgency_categories:
    cfx_ = np.squeeze(np.mean(confx_all[category], axis=0))
    print(f'Confusion matrix for {category} urgency:')
    print(cfx_)
    sum = np.sum(cfx_, axis=1)
    sum = np.repeat(sum[:, None], 2, axis=1)

    cfx_ = 100 * (cfx_ / sum)

    np.save(f'cfx_{category}_EGX3', confx_all[category], allow_pickle=True)
    np.save(f'scores_{category}_EGX3', scores_all[category], allow_pickle=True)

    disp = ConfusionMatrixDisplay(confusion_matrix=cfx_)
    disp.plot()
    plt.show()
    plt.savefig(f'cfx_{category}_EGX3.pdf')

    print(f'Mean score for {category} urgency: {np.mean(scores_all[category])}')
