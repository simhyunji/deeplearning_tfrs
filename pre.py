import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Dict, Text

def load_data():
    train1 = pd.read_csv("train.csv")
    train_data = train1.copy()
    train_data = train_data[["User-ID", "Book-Rating", "Book-Title"]]

    
    ratings = tf.data.Dataset.from_tensor_slices({
        "book_title": train_data["Book-Title"].values,
        "user_id": train_data["User-ID"].values,
        "book_rating": train_data["Book-Rating"].values
    })

    
    train_size = int(len(train_data) * 0.8)
    test_size = len(train_data) - train_size

    
    train = ratings.take(train_size)
    test = ratings.skip(train_size)

   
    unique_user_ids = np.unique(train_data["User-ID"].values)
    unique_book_titles = np.unique(train_data["Book-Title"].values)

    return train, test, unique_user_ids, unique_book_titles
