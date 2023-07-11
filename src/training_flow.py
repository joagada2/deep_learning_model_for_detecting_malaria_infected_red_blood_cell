import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
import warnings
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
from prefect import Flow, Parameter, task
from prefect.schedules import IntervalSchedule
from sklearn.metrics import classification_report,confusion_matrix
import warnings

warnings.filterwarnings(action="ignore")
import numpy as np
import xgboost as xgb
import joblib
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, \
    roc_auc_score, precision_score, recall_score
from xgboost import XGBClassifier
import os
import mlflow
from dagshub import DAGsHubLogger

image_gen = ImageDataGenerator(rotation_range=20,  # rotate the image 20 degrees
                               width_shift_range=0.10,  # Shift the pic width by a max of 5%
                               height_shift_range=0.10,  # Shift the pic height by a max of 5%
                               # rescale=1/255, # Rescale the image by normalzing it.
                               shear_range=0.1,  # Shear means cutting away part of the image (max 10%)
                               zoom_range=0.1,  # Zoom in by 10% max
                               horizontal_flip=True,  # Allo horizontal flipping
                               fill_mode='nearest'  # Fill in missing pixels with the nearest filled value
                               )

train_image_gen = image_gen.flow_from_directory(train_path,
                                                target_size=image_shape[:2],
                                                color_mode='rgb',
                                                batch_size=batch_size,
                                                class_mode='binary')

test_image_gen = image_gen.flow_from_directory(test_path,
                                               target_size=image_shape[:2],
                                               color_mode='rgb',
                                               batch_size=batch_size,
                                               class_mode='binary', shuffle=False)

train_path = 'data\cell_images/train'
test_path = 'data\cell_images/test'

@task
def train_model():
    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=image_shape, activation='relu', ))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), input_shape=image_shape, activation='relu', ))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), input_shape=image_shape, activation='relu', ))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(128))
    model.add(Activation('relu'))

    # Dropouts help reduce overfitting by randomly turning neurons off during training.
    # Here we say randomly turn off 50% of neurons.
    model.add(Dropout(0.5))

    # Last layer, remember its binary so we use sigmoid
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.summary()

    early_stop = EarlyStopping(monitor='val_loss', patience=2)

    batch_size = 16


    warnings.filterwarnings('ignore')

    model.fit_generator(train_image_gen, epochs=20,
                        validation_data=test_image_gen,
                        callbacks=[early_stop])
    model.save('model/malaria_detector_model.h5')
    return model

@task
def evaluate(a):
    #link up to dagshub MLFlow environment
    os.environ['MLFLOW_TRACKING_URI'] = ''
    os.environ['MLFLOW_TRACKING_USERNAME'] = ''
    os.environ['MLFLOW_TRACKING_PASSWORD'] = ''
    with mlflow.start_run():
        # Load data and model

        a.evaluate_generator(test_image_gen)
        pred_probabilities = model.predict_generator(test_image_gen)

        # Get predictions
        predictions = pred_probabilities > 0.5

        # Get metrics
        f1 = f1_score(test_image_gen.classes,predictions)
        print(f"F1 Score of this model is {f1}.")

        # get metrics
        accuracy = balanced_accuracy_score(test_image_gen.classes,predictions)
        print(f"Accuracy Score of this model is {accuracy}.")

        area_under_roc = roc_auc_score(test_image_gen.classes,predictions)
        print(f"Area Under ROC is {area_under_roc}.")

        precision = precision_score(test_image_gen.classes,predictions)
        print(f"Precision of this model is {precision}.")

        recall = recall_score(test_image_gen.classes,predictions)
        print(f"Recall for this model is {recall}.")

        # helper class for logging model and metrics
        class BaseLogger:
            def __init__(self):
                self.logger = DAGsHubLogger()

            def log_metrics(self, metrics: dict):
                mlflow.log_metrics(metrics)
                self.logger.log_metrics(metrics)

            def log_params(self, params: dict):
                mlflow.log_params(params)
                self.logger.log_hyperparams(params)
        logger = BaseLogger()
        # function to log parameters to dagshub and mlflow
        def log_params(a: Sequential):
            logger.log_params({"model_class": type(c).__name__})
            model_params = a.get_params()

            for arg, value in model_params.items():
                logger.log_params({arg: value})

        # function to log metrics to dagshub and mlflow
        def log_metrics(**metrics: dict):
            logger.log_metrics(metrics)
        # log metrics to remote server (dagshub)
        log_params(a)
        log_metrics(f1_score=f1, accuracy_score=accuracy, area_Under_ROC=area_under_roc, precision=precision,
                recall=recall)
            # log metrics to local mlflow
            # mlflow.sklearn.log_model(model, "model")
            # mlflow.log_metric('f1_score', f1)
            # mlflow.log_metric('accuracy_score', accuracy)
            # mlflow.log_metric('area_under_roc', area_under_roc)
            # mlflow.log_metric('precision', precision)
            # mlflow.log_metric('recall', recall)

#adding schedule here automate the pipeline and make it run every 10 minutes
schedule = IntervalSchedule(interval=timedelta(minutes=10))

#create and run flow locally. To schedule to workflow to be automatically triggered every 4 hrs,
#add 'schedule' as Flow parameter (ie with Flow("loan-default-prediction", schedule)
with Flow("malaria_detection_model_training_flow", schedule) as flow:
    model = train_model()
    evaluate(model)

#flow.visualize()
flow.run()
#connect to prefect 1 cloud
flow.register(project_name='malaria-detection-model')
flow.run_agent()
