import pandas as pd
import numpy as np
import tensorflow as tf
from typing import Dict, Text
import model

def get_predicted_actual_ratings(ranking_model, cached_test):
    # 테스트 데이터셋에서 예측 평점 및 실제 평점 가져오기
    predicted_ratings = ranking_model.predict(cached_test).flatten()
    actual_ratings = np.concatenate([sample["book_rating"].numpy() for sample in cached_test])

    # 데이터프레임 생성
    result = pd.DataFrame({'predict_rating': predicted_ratings, 'actual_ranking': actual_ratings})

    return result

def get_top_5_recommendations(ranking_model, unique_user_ids, unique_book_titles, test_data, user_id):
    user_rand_index = np.where(unique_user_ids == user_id)[0][0]
    test_ratings = {}
    # 테스트 데이터셋에서 상위 5개 도서를 추천
    for sample in test_data.take(5):
        user_id_tensor = tf.convert_to_tensor([str(unique_user_ids[user_rand_index])])
        book_title_tensor = tf.convert_to_tensor([sample["book_title"].numpy().decode()])
        rating = model.RankingModel(unique_user_ids, unique_book_titles)((user_id_tensor, book_title_tensor))
        test_ratings[sample["book_title"].numpy()] = rating.numpy()[0][0]

    print("사용자 {}를 위한 상위 5개 권장 제품: ".format(unique_user_ids[user_rand_index]))
    for title in sorted(test_ratings, key=test_ratings.get, reverse=True):
        print(title)
