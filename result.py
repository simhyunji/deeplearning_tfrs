import pre as pp
import model
import tensorflow as tf
import recommend
import plt
import pandas as pd
import numpy as np

# 데이터 불러오기 및 전처리
train, test, unique_user_ids, unique_book_titles = pp.load_data()

# BookRankingModel 클래스 초기화
ranking_model = model.BookRankingModel(unique_user_ids, unique_book_titles)
ranking_model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.01))

cached_train = train.batch(4096).cache()
cached_test = test.batch(4096).cache()

# 모델 훈련
ranking_model.fit(cached_train , epochs=15)

# 모델 평가
ranking_model.evaluate(cached_test)

# 결과 확인
plt.plot_predicted_vs_actual_ratings(ranking_model, cached_test)
result = recommend.get_predicted_actual_ratings(ranking_model, cached_test)
print(result)
recommend.get_top_5_recommendations(ranking_model, unique_user_ids, unique_book_titles, test, 'USER_00000')

def calculate_ranking(ranking_model, cached_test):
    result2 = pd.DataFrame()
    predictions = ranking_model.predict(cached_test)
    result2['predicted_rating'] = predictions.flatten()
    real_ratings = np.concatenate([batch["book_rating"].numpy() for batch in cached_test])
    result2["predicted_rating_ranking"] = result2['predicted_rating'].rank(ascending=False, method="dense").astype(int)
    result2["real_rating_ranking"] = pd.Series(real_ratings).rank(ascending=False, method="first").astype(int)

    return result2
