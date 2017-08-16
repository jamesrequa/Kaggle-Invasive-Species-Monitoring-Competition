import numpy as np
import pandas as pd
import cv2
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, Input, Flatten, Dropout, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from batch_renorm import BatchRenormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam, SGD
from sklearn import metrics


def load_train():
    train_set = pd.read_csv('../input/train_labels.csv')
    train_label = np.array(train_set['invasive'].iloc[: ])
    path = "../input/train/"
    train_files = []
    for i in range(len(train_set)):
        train_files.append(path + str(int(train_set.iloc[i][0])) +'.jpg')
    train_set['name'] = train_files
    return train_files, train_set

def load_test():
    test_set = pd.read_csv('../input/sample_submission.csv')
    path = "../input/test/"
    test_files = []
    for i in range(len(test_set)):
        test_files.append(path + str(int(test_set.iloc[i][0])) +'.jpg')
    return test_files

def get_pred_set(path, test_files):
    pred_set = pd.read_csv(path)

    # Set prediction float values to integer value 0 or 1
    preds = pred_set['invasive'].values
    thres_index = preds > 0.5
    preds[thres_index] = 1
    thres_index = preds < 0.5
    preds[thres_index] = 0
    preds = preds.astype(int)
    pred_set['invasive'] = preds

    # Replace img names with img_paths
    pred_set['name'] = test_files

    return pred_set

def vgg19(img_dim):
    input_tensor = Input(shape=img_dim)
    base_model = VGG19(include_top=False,
                   weights='imagenet',
                   input_shape=img_dim)
    bn = BatchNormalization()(input_tensor)
    x = base_model(bn)
    x = Flatten()(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(input_tensor, output)
    return model

def vgg16(img_dim):
    input_tensor = Input(shape=img_dim)
    base_model = VGG16(include_top=False,
                   weights='imagenet',
                   input_shape=img_dim)
    bn = BatchNormalization()(input_tensor)
    x = base_model(bn)
    x = Flatten()(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(input_tensor, output)
    return model

def resnet50(img_dim):
    input_tensor = Input(shape=img_dim)
    base_model = ResNet50(include_top=False,
                   weights='imagenet',
                   input_shape=img_dim, pooling='avg')
    bn = BatchNormalization()(input_tensor)
    x = base_model(bn)
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(input_tensor, output)
    return model

def inceptionv3(img_dim):
    input_tensor = Input(shape=img_dim)
    base_model = InceptionV3(include_top=False,
                   weights='imagenet',
                   input_shape=img_dim)
    bn = BatchRenormalization()(input_tensor)
    x = base_model(bn)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(input_tensor, output)
    return model

def xception(img_dim):
    input_tensor = Input(shape=img_dim)
    base_model = Xception(include_top=False,
                   weights='imagenet',
                   input_shape=img_dim)
    bn = BatchNormalization()(input_tensor)
    x = base_model(bn)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(input_tensor, output)
    return model

def train_model(model, batch_size, epochs, img_size, x, y, test, n_fold, kf):
    roc_auc = metrics.roc_auc_score
    preds_train = np.zeros(len(x), dtype = np.float)
    preds_test = np.zeros(len(test), dtype = np.float)
    train_scores = []; valid_scores = []

    i = 1

    for train_index, test_index in kf.split(x):
        x_train = x.iloc[train_index]; x_valid = x.iloc[test_index]
        y_train = y[train_index]; y_valid = y[test_index]

        def augment(src, choice):
            if choice == 0:
                # Rotate 90
                src = np.rot90(src, 1)
            if choice == 1:
                # flip vertically
                src = np.flipud(src)
            if choice == 2:
                # Rotate 180
                src = np.rot90(src, 2)
            if choice == 3:
                # flip horizontally
                src = np.fliplr(src)
            if choice == 4:
                # Rotate 90 counter-clockwise
                src = np.rot90(src, 3)
            if choice == 5:
                # Rotate 180 and flip horizontally
                src = np.rot90(src, 2)
                src = np.fliplr(src)
            return src

        def train_generator():
            while True:
                for start in range(0, len(x_train), batch_size):
                    x_batch = []
                    y_batch = []
                    end = min(start + batch_size, len(x_train))
                    train_batch = x_train[start:end]
                    for filepath, tag in train_batch.values:
                        img = cv2.imread(filepath)
                        img = cv2.resize(img, img_size)
                        img = augment(img, np.random.randint(6))
                        x_batch.append(img)
                        y_batch.append(tag)
                    x_batch = np.array(x_batch, np.float32) / 255.
                    y_batch = np.array(y_batch, np.uint8)
                    yield x_batch, y_batch

        def valid_generator():
            while True:
                for start in range(0, len(x_valid), batch_size):
                    x_batch = []
                    y_batch = []
                    end = min(start + batch_size, len(x_valid))
                    valid_batch = x_valid[start:end]
                    for filepath, tag in valid_batch.values:
                        img = cv2.imread(filepath)
                        img = cv2.resize(img, img_size)
                        img = augment(img, np.random.randint(6))
                        x_batch.append(img)
                        y_batch.append(tag)
                    x_batch = np.array(x_batch, np.float32) / 255.
                    y_batch = np.array(y_batch, np.uint8)
                    yield x_batch, y_batch

        def test_generator():
            while True:
                for start in range(0, len(test), batch_size):
                    x_batch = []
                    end = min(start + batch_size, len(test))
                    test_batch = test[start:end]
                    for filepath in test_batch:
                        img = cv2.imread(filepath)
                        img = cv2.resize(img, img_size)
                        x_batch.append(img)
                    x_batch = np.array(x_batch, np.float32) / 255.
                    yield x_batch

        callbacks = [EarlyStopping(monitor='val_loss', patience=3, verbose=1, min_delta=1e-4),
             ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, cooldown=1, verbose=1, min_lr=1e-7),
             ModelCheckpoint(filepath='weights/inception.fold_' + str(i) + '.hdf5', verbose=1,
                             save_best_only=True, save_weights_only=True, mode='auto')]

        train_steps = len(x_train) / batch_size
        valid_steps = len(x_valid) / batch_size
        test_steps = len(test) / batch_size

        model = model

        model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics = ['accuracy'])

        model.fit_generator(train_generator(), train_steps, epochs=epochs, verbose=1, callbacks=callbacks,
                    validation_data=valid_generator(), validation_steps=valid_steps)

        model.load_weights(filepath='weights/inception.fold_' + str(i) + '.hdf5')

        print('Running validation predictions on fold {}'.format(i))
        preds_valid = model.predict_generator(generator=valid_generator(),
                                      steps=valid_steps, verbose=1)[:, 0]

        print('Running train predictions on fold {}'.format(i))
        preds_train = model.predict_generator(generator=train_generator(),
                                      steps=train_steps, verbose=1)[:, 0]

        valid_score = roc_auc(y_valid, preds_valid)
        train_score = roc_auc(y_train, preds_train)
        print('Val Score:{} for fold {}'.format(valid_score, i))
        print('Train Score: {} for fold {}'.format(train_score, i))

        valid_scores.append(valid_score)
        train_scores.append(train_score)
        print('Avg Train Score:{0:0.5f}, Val Score:{1:0.5f} after {2:0.5f} folds'.format(np.mean(train_scores), np.mean(valid_scores), i))

        print('Running test predictions with fold {}'.format(i))

        preds_test_fold = model.predict_generator(generator=test_generator(),
                                              steps=test_steps, verbose=1)[:, -1]

        preds_test += preds_test_fold

        print('\n\n')

        i += 1

        if i <= n_fold:
            print('Now beginning training for fold {}\n\n'.format(i))
        else:
            print('Finished training!')

    preds_test /= n_fold


    return preds_test
