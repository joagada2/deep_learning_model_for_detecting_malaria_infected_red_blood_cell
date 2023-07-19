import warnings
warnings.filterwarnings(action="ignore")
import tensorflow as tf
from tensorflow import keras
import whylogs
from whylogs.extras.image_metric import log_image
from whylogs.api.writer.whylabs import WhyLabsWriter
from PIL import Image
import os
import datetime
import pandas as pd

# Set WhyLabs environment variables
os.environ['WHYLABS_API_KEY'] = 'APIKEY'
os.environ["WHYLABS_DEFAULT_ORG_ID"] = 'ORGID'
os.environ["WHYLABS_DEFAULT_DATASET_ID"] = 'MODELID'

image_size = (130,130)

#log training data for ml monitoring
writer = WhyLabsWriter().option(reference_profile_name="training_ref_profile")

merged_profile = None
train_path = '..\data\cell_images/train'

for dir in os.listdir(train_path):
    for filename in os.listdir(f'{train_path}{dir}'):
        image_filepath = os.path.join(f'{train_path}{dir}', filename)
        PIL_image = Image.open(image_filepath)
        profile = log_image(PIL_image).profile()
        profile_view = profile.view()  # extract mergeable profile view

        # merge each profile while looping
        if merged_profile is None:
            merged_profile = profile_view
        else:
            merged_profile = merged_profile.merge(profile_view)

writer.write(file=merged_profile)

# log dataset to be used for monitoring and make predictions
preds = {'file_name':[], 'class_pred_output':[], 'class_score_output':[], 'day':[]}
preds_output = {'class_pred_output':[], 'class_score_output':[]}

days_dir = '..\data\cell_images/days'
writer = WhyLabsWriter()

for day in range(1,8):
    date = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=-7+day)
    print(f'Working with day #{day} | {date}')
    for filename in os.listdir(f'{days_dir}day{day}'):

        image_filepath = os.path.join(f'{days_dir}day{day}/', filename)
        image = Image.open(image_filepath)

        # Log raw images
        profile = log_image(image).profile()
        profile.set_dataset_timestamp(date)
        writer.write(profile)

        # Make Predictions with Model
        img = keras.preprocessing.image.load_img(
            image_filepath, target_size=image_size
        )
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create batch axis

        predictions = r50_model2.predict(img_array)
        score = float(predictions[0])

        if score < 0.5:
            print(f'Jetson: {100 * (1 - score):.2f} ')
            cls = 0
            proba = 100 * (1 - score)
        else:
            print(f'Pi: {100 * score:.2f} ')
            cls = 1
            proba = 100 * score

        # Create dict to use late for adding ground truth metrics
        preds['file_name'].append(filename)
        preds['class_pred_output'].append(cls)
        preds['class_score_output'].append(proba)
        preds['day'].append(day)
    print(preds)

# log predictions
df = pd.DataFrame(data=preds)
writer = WhyLabsWriter()
for day in range (1,8):
  date = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=-7+day)
  print(f'Logging day #{day} | {date}')

  day_df = df.loc[df['day'] == day]

  profile = whylogs.log(day_df[['class_pred_output', 'class_score_output']]).profile()
  profile.set_dataset_timestamp(date)
  writer.write(file=profile.view())

# monitor model performance

# add ground truth to DF based on file name
df['ground truth'] = None

# Loop through each row in the dataframe
for index, row in df.iterrows():
    file_name = row['file_name']

    # Check if 'file_name' contains "jet"
    if 'jet' in file_name:
        df.at[index, 'ground_truth'] = 0
    # Check if 'file_name' contains "pi"
    elif 'pi' in file_name:
        df.at[index, 'ground_truth'] = 1

writer = WhyLabsWriter()

for day in range(1, 8):
    date = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=-7 + day)
    print(f'Logging day #{day} | {date}')

    day_df = df.loc[df['day'] == day]

    results = whylogs.log_classification_metrics(
        day_df,
        target_column="ground_truth",
        prediction_column="class_pred_output",
        score_column="class_score_output"
    )
    profile = results.profile()
    profile.set_dataset_timestamp(date)

    writer.write(file=profile.view())