from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from datetime import datetime, timedelta
from prefect import Flow, Parameter, task
from prefect.schedules import IntervalSchedule
import warnings
warnings.filterwarnings(action="ignore")
from sklearn.metrics import balanced_accuracy_score, f1_score, \
    roc_auc_score, precision_score, recall_score
import os
import mlflow
from dagshub import DAGsHubLogger
from tensorflow.keras.models import load_model

image_shape = (130,130,3)

train_path = 'data\cell_images/train'
test_path = 'data\cell_images/test'

batch_size = 16

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

train_path = 'data\cell_images\train'
test_path = 'data\cell_images\test'

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

    warnings.filterwarnings('ignore')

    model.fit_generator(train_image_gen, epochs=20,
                        validation_data=test_image_gen,
                        callbacks=[early_stop])
    model.save('model/malaria_detector_model.h5')
    return model

@task
def evaluate(a):
    #link up to dagshub MLFlow environment
    os.environ['MLFLOW_TRACKING_URI'] = 'https://dagshub.com/joe88data/deep_learning_model_for_detecting_malaria_infected_red_blood_cell.mlflow'
    os.environ['MLFLOW_TRACKING_USERNAME'] = 'joe88data'
    os.environ['MLFLOW_TRACKING_PASSWORD'] = 'e94114ca328c75772401898d749decb6dbcbeb21'
    with mlflow.start_run():
        # Load data and model
        a  = load_model('model/malaria_detector_model.h5')
        a.evaluate_generator(test_image_gen)
        pred_probabilities = a.predict_generator(test_image_gen)

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

        logger = BaseLogger()

        # function to log metrics to dagshub and mlflow
        def log_metrics(**metrics: dict):
            logger.log_metrics(metrics)
        # log metrics to remote server (dagshub)
        #log_params(a)
        log_metrics(f1_score=f1, accuracy_score=accuracy, area_Under_ROC=area_under_roc, precision=precision,
                recall=recall)

#adding schedule here automate the pipeline and make it run every 10 minutes
schedule = IntervalSchedule(interval=timedelta(minutes=10))

#create and run flow locally. To schedule to workflow to be automatically triggered every 4 hrs,
#add 'schedule' as Flow parameter (ie with Flow("loan-default-prediction", schedule)
with Flow("malaria_detection_model_training_flow") as flow:
    model = train_model()
    evaluate(model)

#flow.visualize()
flow.run()
#connect to prefect 1 cloud
flow.register(project_name='malaria-detection-model')
flow.run_agent()
