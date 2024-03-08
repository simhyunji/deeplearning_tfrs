import tensorflow as tf
import tensorflow_recommenders as tfrs
from typing import Dict, Text
import pre as pp

class RankingModel(tf.keras.Model):

    def __init__(self, unique_user_ids, unique_book_titles, embedding_dimension=64):
        super().__init__()

        
        self.user_embeddings = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=unique_user_ids, mask_token=None),
            tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
        ])

        
        self.book_embeddings = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=unique_book_titles, mask_token=None),
            tf.keras.layers.Embedding(len(unique_book_titles) + 1, embedding_dimension)
        ])

        
        self.ratings = tf.keras.Sequential([
            # Learn multiple dense layers.
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            # Make rating predictions in the final layer.
            tf.keras.layers.Dense(1)
        ])

    def call(self, inputs):
        user_id, book_title = inputs

        user_embedding = self.user_embeddings(user_id)
        book_embedding = self.book_embeddings(book_title)

        return self.ratings(tf.concat([user_embedding, book_embedding], axis=1))


class BookRankingModel(tfrs.models.Model):

    def __init__(self, unique_user_ids, unique_book_titles):
        super().__init__()
        self.ranking_model: tf.keras.Model = RankingModel(unique_user_ids, unique_book_titles)
        self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()]
        )

    def call(self, features: Dict[str, tf.Tensor]) -> tf.Tensor:
        return self.ranking_model(
            (features["user_id"], features["book_title"]))

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        labels = features.pop("book_rating")

        rating_predictions = self(features)

       
        return self.task(labels=labels, predictions=rating_predictions)
