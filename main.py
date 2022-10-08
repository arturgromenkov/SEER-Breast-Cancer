import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.python.keras.callbacks import TensorBoard
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
import os
import shutil

#tensorboard --logdir="logs/"

def full_outer_join_except_inner_join(LesserDataframe,BiggerDataframe):
    lesser_indexes=LesserDataframe.index.to_list()
    bigger_indexes=BiggerDataframe.index.to_list()
    for i in lesser_indexes:
        bigger_indexes.remove(i)
    return bigger_indexes

def convert_categorical_to_numeric(PandasSeries,interpolate=False):

    if (interpolate):
        #assert PandasSeries.isnull().values.any()==True,"Given column posses no nan values"
        PandasSeries.interpolate(inplace=True)
        PandasSeries.replace(PandasSeries.unique(),[i for i in range(PandasSeries.unique().shape[0])],inplace=True)
    else:
        assert PandasSeries.isnull().values.any()==False,"Given column posses some nan values, use INTERPOLATE flag"
        PandasSeries.replace(PandasSeries.unique(), [i for i in range(PandasSeries.unique().shape[0])], inplace=True)
    return PandasSeries

# Folders preparations
shutil.rmtree("logs\\fit")
shutil.rmtree("models\\")
os.mkdir("models")
os.mkdir("logs\\fit")


with tf.device ("/GPU:0"):#dml

    dataset=pd.read_csv("res/SEER Breast Cancer Dataset .csv")
    #Dataset changing
    dataset.drop("Unnamed: 3",axis=1,inplace=True)
    for i in dataset:
        dataset[i]=convert_categorical_to_numeric(dataset[i])
    dataset.interpolate(inplace=True)
    #Visualisation
    # fig,ax=plt.subplots(figsize=(20,20))
    # sns.heatmap(dataset.corr(),annot=True)
    # plt.show()
    #split dataset into validation and training
    train_df=dataset.sample(axis=0,frac=0.7,random_state=1)
    valid_df=dataset.iloc[full_outer_join_except_inner_join(train_df,dataset)]
    #split training data into X and Y
    train_x=train_df.drop("Status",axis=1).to_numpy().astype(float)
    train_y=train_df["Status"].to_numpy().astype(float)
    #split validation data into X and Y
    valid_x=valid_df.drop("Status",axis=1).to_numpy().astype(float)
    valid_y=valid_df["Status"].to_numpy().astype(float)
    #Creating and fitting the model
    def create_model():
        return tf.keras.models.Sequential([
            tf.keras.layers.BatchNormalization(),#Show better results
            tf.keras.layers.Dense(69, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(20, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ])
    model = create_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # normal rate =0.001
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    time_stop = int(time.time())
    #tensorboard = TensorBoard(log_dir="logs\\fit\\{}".format(time_stop))
    tensorboard = TensorBoard(log_dir="logs\\fit\\{}".format("Batch_norm"))
    epochs = 10

    model.fit(
        x=train_x,
        y=train_y,
        epochs=epochs,
        validation_split=0.2,
        batch_size=32,
        callbacks=[tensorboard]
    )

    results=model.evaluate(valid_x,valid_y,batch_size=32)
    print("test loss, test acc:", results)
    model.save("models/{}".format(time_stop))
