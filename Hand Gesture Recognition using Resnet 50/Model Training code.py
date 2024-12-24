import os
import cv2
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from keras.layers import Dense, BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from keras.applications import ResNet50
from keras import Model
import joblib

main_folder = 'HG14-Hand Gesture'

def load_and_preprocess_data(folder):
    images = []
    labels = []

    for label in sorted(os.listdir(folder)):
        label_path = os.path.join(folder, label)

        if os.path.isdir(label_path):
            for filename in os.listdir(label_path):
                img_path = os.path.join(label_path, filename)
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
                img = cv2.resize(img, (128, 128))  # Resize images to a consistent size
                images.append(img)
                labels.append(label)

    return np.array(images), np.array(labels)


images, labels = load_and_preprocess_data(main_folder)

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

joblib.dump(label_encoder, 'label_encoder_resnet.joblib')

class_weights = compute_class_weight('balanced', classes=np.unique(encoded_labels), y=encoded_labels).tolist()
print("Class Weights:", class_weights)

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
predictions = Dense(len(label_encoder.classes_), activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Define k-fold cross-validation
kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Iterate over each fold
for fold, (train_indices, test_indices) in enumerate(kfold.split(images, encoded_labels)):
    X_train, X_test = images[train_indices], images[test_indices]
    y_train, y_test = encoded_labels[train_indices], encoded_labels[test_indices]

    model_checkpoint = ModelCheckpoint(f'best_model_fold_{fold}.h5', save_best_only=True, monitor='val_loss', mode='min')

    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-6)

    # Training
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        epochs=5,
        validation_data=(datagen.flow(X_test, y_test, batch_size=32)),
        class_weight=dict(enumerate(class_weights)),
        callbacks=[model_checkpoint, early_stopping, reduce_lr]
    )

    # Print training accuracy
    train_accuracy = history.history['accuracy'][-1]  # Get the last epoch's accuracy
    print(f'Fold {fold + 1} Training Accuracy: {train_accuracy * 100:.2f}%')

    # Evaluate on test data
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f'Fold {fold + 1} Testing Loss: {test_loss:.4f}')
    print(f'Fold {fold + 1} Testing Accuracy: {test_accuracy * 100:.2f}%')

# Save the model in the native Keras format
model.save('hand_gesture_model_resnet')

