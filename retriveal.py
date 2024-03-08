import tensorflow as tf
import tensorflow_recommenders as tfrs
from typing import Dict, Text

class RetrievalModel(tf.keras.Model):
    def __init__(self, unique_user_ids, unique_book_titles):
        super().__init__()
        embedding_dimension = 32

        self.user_model = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=unique_user_ids, mask_token=None),
            # We add an additional embedding to account for unknown tokens.
            tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
        ])

        self.book_model = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=unique_book_titles, mask_token=None),
            tf.keras.layers.Embedding(len(unique_book_titles) + 1, embedding_dimension)
        ])

class BookRetrievalModel(tfrs.Model):
    def __init__(self, unique_user_ids, unique_book_titles):
        super().__init__()

        self.retrieval_model = RetrievalModel(unique_user_ids, unique_book_titles)
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=tf.data.Dataset.from_tensor_slices(unique_book_titles).batch(128).map(self.retrieval_model.book_model)
            )
        )

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        
        user_embeddings = self.retrieval_model.user_model(features["user_id"])
  
        positive_book_embeddings = self.retrieval_model.book_model(features["book_title"])

       
        return self.task(user_embeddings, positive_book_embeddings)
