import mat73
import numpy as np
import keras
from keras.layers import Input, Reshape
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Activation, AveragePooling2D
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, BatchNormalization, LayerNormalization, Flatten
from tensorflow.keras.layers import Permute, Lambda, Add, MultiHeadAttention
from tensorflow.keras.constraints import max_norm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import backend as K

# Function to duplicate female subjects in training data and include self-reported labels
def duplicate_females_in_train(data, labels, categories, self_reported, test_subject_idx):
    augmented_data = []
    augmented_labels = []
    augmented_categories = []
    augmented_self_reported = []

    for i in range(len(data)):
        if i == test_subject_idx:  # Skip the test subject
            continue
        augmented_data.append(data[i])
        augmented_labels.append(labels[i])
        augmented_categories.append(categories[i])
        augmented_self_reported.append(self_reported[i])


        if labels[i] == 0:  # Duplicate female subjects
            for _ in range(4):  # Eight times duplication
                augmented_data.append(apply_augmentation(data[i]))
                augmented_labels.append(labels[i])
                augmented_categories.append(categories[i])
                augmented_self_reported.append(self_reported[i])
       
        if labels[i] == 1:  # Duplicate female subjects
            for _ in range(4):  # Eight times duplication
                augmented_data.append(apply_augmentation(data[i]))
                augmented_labels.append(labels[i])
                augmented_categories.append(categories[i])
                augmented_self_reported.append(self_reported[i])
        

    return augmented_data, augmented_labels, augmented_categories, augmented_self_reported





# Function for data augmentation
def apply_augmentation(data):
    # Time shifting
    shift = np.random.randint(-5, 5)
    data_shifted = np.roll(data, shift, axis=1)

    # Scaling
    scale = np.random.uniform(0.9, 1.1)
    data_scaled = data_shifted * scale

    # Adding noise
    noise = np.random.normal(0, 0.05, data_scaled.shape)  # You can adjust the noise level
    data_noisy = data_scaled + noise

    return data_noisy






# Adjust dimensions of test_categories and train_categories
def adjust_category_shape(categories, n_timesteps, n_trials):
    return np.tile(categories[:, None, :], (1, n_timesteps, 1)).reshape(n_trials, n_timesteps, -1)

# Attention mechanism
def attention_block(inputs):
    attention_probs = Dense(inputs.shape[-1], activation='softmax')(inputs)
    attention_mul = tf.keras.layers.multiply([inputs, attention_probs])
    return attention_mul

# Transformer encoder block
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = Dropout(dropout)(x)
    res = Add()([x, inputs])

    # Feed Forward Part
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(ff_dim, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    return Add()([x, res])

# Function to define EEGNeX + Transformer model
def EEGNeX_Transformer(n_timesteps, n_features, n_extra_features, n_outputs):
    input_main = Input((n_timesteps, n_features + n_extra_features, 1))
    block1s = Permute((3, 2, 1))(input_main)
    block1 = Conv2D(filters=8, kernel_size=(1, 32), use_bias=False, padding='same', data_format="channels_first")(block1s)
    block1 = LayerNormalization()(block1)
    block1 = Activation(activation='elu')(block1)

    block2 = Conv2D(filters=32, kernel_size=(1, 32), use_bias=False, padding='same', data_format="channels_first")(block1)
    block2 = LayerNormalization()(block2)
    block2 = Activation(activation='elu')(block2)

    block3 = DepthwiseConv2D(kernel_size=(n_features + n_extra_features, 1), depth_multiplier=2, use_bias=False, depthwise_constraint=max_norm(1.), data_format="channels_first")(block2)
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

    # Adding Transformer Encoder layers
    block5 = Lambda(lambda x: tf.keras.backend.permute_dimensions(x, (0, 2, 1)))(block5)  # Permute dimensions to fit Transformer input format
    block5_shape = K.int_shape(block5)
    block5 = Reshape((block5_shape[1], block5_shape[2]))(block5)
    block5 = transformer_encoder(block5, head_size=64, num_heads=4, ff_dim=128, dropout=0.1)
    block5 = transformer_encoder(block5, head_size=64, num_heads=4, ff_dim=128, dropout=0.1)

    block5 = Flatten()(block5)
    
    # Adding Attention mechanism
    block5 = attention_block(block5)
    
    block5 = Dense(128, activation='relu')(block5)
    block5 = Dropout(0.5)(block5)
    block5 = Dense(n_outputs, kernel_constraint=max_norm(0.25))(block5)
    block5 = Activation(activation='softmax')(block5)
    return Model(inputs=input_main, outputs=block5)

# Load and preprocess data
vibration_dict = mat73.loadmat('/scratch/ac9429/gender_classification/dataUrg_sub_200Hz.mat')
vibration_list = vibration_dict['data']
print(np.shape(vibration_list[0]), 'x', len(vibration_list))

vibration_tr_con = []
categories = []

# Create category labels
for i in range(len(vibration_list)):
    vibration_arr = vibration_list[i]

    vibration_arr[0] = vibration_arr[0][:, :, :36]
    vibration_arr[1] = vibration_arr[1][:, :, :36]
    vibration_arr[2] = vibration_arr[2][:, :, :36]

    vibration_temp = np.concatenate((np.array(vibration_arr[0]), np.array(vibration_arr[1]), np.array(vibration_arr[2])), axis=2)
    vibration_tr_con.append(vibration_temp)

    # Create category labels (0 for NVP, 1 for VUP, 2 for VVUP)
    categories.append(np.concatenate([
        np.zeros(36, dtype=int),
        np.ones(36, dtype=int),
        np.full(36, 2, dtype=int)
    ]))

# One-hot encoding the categories
one_hot_categories = [to_categorical(cat, num_classes=3) for cat in categories]

# Creating gender labels for each trial
gender_labels = [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0]
gender_trials = []
for idx, trials in enumerate(vibration_tr_con):
    gender_label = np.full((1, trials.shape[2]), gender_labels[idx])
    gender_trials.append(gender_label)

# Load and preprocess the self-reported labels
self_reported_labels = pd.read_csv('/scratch/ac9429/gender_classification/self_reported_labels.csv')
self_reported_labels = self_reported_labels.iloc[:, 1:].values

# Create self-reported labels for each subject and trial
self_reported_trials = []
for idx, trials in enumerate(vibration_tr_con):
    self_reported_label = np.full((1, trials.shape[2]), self_reported_labels[idx])
    self_reported_trials.append(self_reported_label)

def scheduler(epoch, lr):
    if epoch < 50:
        return lr
    if epoch < 100 and epoch >= 50:
        return lr / 2
    if epoch >= 100:
        return lr / 10

nSub = len(gender_labels)  # number of subjects (after duplication)
bs_t = 64  # batch size
epochs = 200
lr = 0.0009
scores = []
nb_classes = 2
chans = 60
samples = 400
extra_features = 3  # Number of additional features (category information)

confx = np.zeros((nSub, nb_classes, nb_classes))
shap_values_all = []
y_test_all = []
y_pred_all = []

# Calculate class weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(gender_labels), y=gender_labels)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

for sub in range(nSub):
    print(f'Subject: #{sub}')
    # Split train and test - time
    X_test_t = vibration_tr_con[sub]

    # Duplicate females in the remaining training data
    x_train_t, train_labels, train_categories, train_self_reported = duplicate_females_in_train(vibration_tr_con, gender_labels, one_hot_categories, self_reported_trials, sub)
    x_train_t = np.concatenate(x_train_t, axis=2)

    # Concatenate the gender labels for all trials of the training set
    y_train_t = []
    for label in train_labels:
        y_train_t.append(np.full((1, trials.shape[2]), label))
    y_train_t = np.concatenate(y_train_t, axis=1).flatten()

    # Concatenate the categories for all trials of the training set
    train_categories = np.concatenate(train_categories, axis=0)

    # Concatenate the self-reported labels for all trials of the training set
    train_self_reported = np.concatenate(train_self_reported, axis=0)

    print(f"Shape of X_train_t before upsampling: {x_train_t.shape}")
    print(f"Shape of y_train_t before upsampling: {y_train_t.shape}")
    print(f"Shape of train_categories: {train_categories.shape}")
    print(f"Shape of train_self_reported: {train_self_reported.shape}")

    # Extract gender labels for the test subject
    Y_test_t = gender_trials[sub].flatten()

    # Extract categories for the test subject
    test_categories = one_hot_categories[sub]

    # Extract self-reported labels for the test subject
    test_self_reported = self_reported_trials[sub].flatten()

    # Adjust shapes for concatenation
    test_categories = adjust_category_shape(test_categories, samples, 108)
    train_categories = adjust_category_shape(train_categories, samples, len(y_train_t))

    test_self_reported = np.repeat(test_self_reported[:, np.newaxis], samples, axis=1).reshape(108, samples, 1)
    train_self_reported = np.repeat(train_self_reported[:, np.newaxis], samples, axis=1).reshape(len(y_train_t), samples, 1)

    print(f"Shape of test_categories: {test_categories.shape}")
    print(f"Shape of train_categories: {train_categories.shape}")
    print(f"Shape of test_self_reported: {test_self_reported.shape}")
    print(f"Shape of train_self_reported: {train_self_reported.shape}")

    X_test_t = np.swapaxes(X_test_t, 0, 2)
    X_test_t = np.swapaxes(X_test_t, 1, 2)

    X_train_t = np.swapaxes(x_train_t, 0, 2)
    X_train_t = np.swapaxes(X_train_t, 1, 2)

    print(f"Shape of X_test_t after swapaxes: {X_test_t.shape}")
    print(f"Shape of X_train_t after swapaxes: {X_train_t.shape}")

    # Append the categories and self-reported labels to the input features
    test_categories = np.swapaxes(test_categories, 0, 1)  # shape (108, 400, 3)
    train_categories = np.swapaxes(train_categories, 0, 1)  # shape (4320, 400, 3)
    
    test_categories = np.swapaxes(test_categories, 0, 1)
    test_categories = np.swapaxes(test_categories, 1, 2)
    
    train_categories = np.swapaxes(train_categories, 0, 1) 
    train_categories = np.swapaxes(train_categories, 1, 2) 
    
    train_self_reported = np.swapaxes(train_self_reported,1,2)
    
    test_self_reported = np.swapaxes(test_self_reported, 1,2)

    print(f"Shape of test_categories after swapaxes: {test_categories.shape}")
    print(f"Shape of train_categories after swapaxes: {train_categories.shape}")

    print(f"Shape of X_test_t before concatenation: {X_test_t.shape}")
    print(f"Shape of test_categories before concatenation: {test_categories.shape}")
    print(f"Shape of test_self_reported before concatenation: {test_self_reported.shape}")
    X_test_t = np.concatenate([X_test_t, test_categories, test_self_reported], axis=1)
    print(f"Shape of X_test_t after concatenation: {X_test_t.shape}")

    print(f"Shape of X_train_t before concatenation: {X_train_t.shape}")
    print(f"Shape of train_categories before concatenation: {train_categories.shape}")
    print(f"Shape of train_self_reported before concatenation: {train_self_reported.shape}")
    X_train_t = np.concatenate([X_train_t, train_categories, train_self_reported], axis=1)
    print(f"Shape of X_train_t after concatenation: {X_train_t.shape}")

    X_test_t = np.expand_dims(X_test_t, axis=3)
    X_train_t = np.expand_dims(X_train_t, axis=3)

    Y_test_t = to_categorical(Y_test_t, nb_classes)
    Y_train_t = to_categorical(y_train_t, nb_classes)

    x_train_nex = np.swapaxes(X_train_t, 1, 2)
    x_test_nex = np.swapaxes(X_test_t, 1, 2)

    X_train_t = np.swapaxes(X_train_t, 1, 3)
    X_train_t = np.swapaxes(X_train_t, 2, 3)

    X_test_t = np.swapaxes(X_test_t, 1, 3)
    X_test_t = np.swapaxes(X_test_t, 2, 3)

    print(f"Shape of x_train_nex: {np.shape(x_train_nex)}")
    print(f"Shape of x_test_nex: {np.shape(x_test_nex)}")

    model_hybrid = EEGNeX_Transformer(samples, chans, extra_features + 1, nb_classes)

    opt_hybrid = Adam(learning_rate=lr)

    LR_callback = LearningRateScheduler(scheduler)

    model_hybrid.compile(loss='categorical_crossentropy', optimizer=opt_hybrid, metrics=['accuracy', tf.keras.metrics.AUC(curve='PR')])

    callbacks = [LR_callback]
    print(f'Epochs from 1 to 80')

    history_hybrid = model_hybrid.fit(x_train_nex, Y_train_t, batch_size=bs_t, epochs=epochs, verbose=0, class_weight=class_weight_dict)

    probs_hybrid = model_hybrid.predict(x_test_nex)
    preds_hybrid = probs_hybrid.argmax(axis=-1)
    acc_hybrid = np.mean(preds_hybrid == Y_test_t.argmax(axis=-1))
    print(f'Hybrid Model: {acc_hybrid} %')

    confx[sub, :, :] = confusion_matrix(Y_test_t.argmax(axis=-1), preds_hybrid)

    # Print confusion matrix for the current subject
    print(f'Confusion Matrix for Subject {sub}:\n{confx[sub, :, :]}')

    scores.append(acc_hybrid)

cfx_ = np.squeeze(np.mean(confx, axis=0))
print(cfx_)
sum = np.sum(cfx_, axis=1)
sum = np.repeat(sum[:, None], 2, axis=1)

cfx_ = 100 * (cfx_ / sum)

np.save('cfx_Hybrid', confx, allow_pickle=True)
np.save('scores_Hybrid', scores, allow_pickle=True)

disp = ConfusionMatrixDisplay(confusion_matrix=cfx_)
disp.plot()
plt.show()
plt.savefig('cfx_Hybrid.pdf')
fig, axs = plt.subplots()
print(np.mean(scores))
