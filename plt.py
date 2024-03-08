import matplotlib.pyplot as plt
import numpy as np

def plot_predicted_vs_actual_ratings(ranking_model, cached_test):
    
    predicted_ratings = ranking_model.predict(cached_test)
    actual_ratings = np.concatenate([sample["book_rating"].numpy() for sample in cached_test])

    
    sorted_indices = np.argsort(actual_ratings)
    predicted_ratings = predicted_ratings[sorted_indices]
    actual_ratings = actual_ratings[sorted_indices]

    
    plt.figure(figsize=(10, 6))
    plt.plot(predicted_ratings, 'ro-', label='Predicted Ratings')
    plt.plot(actual_ratings, 'bo-', label='Actual Ratings')
    plt.xlabel('Sample Index')
    plt.ylabel('Rating')
    plt.title('Comparison of Predicted and Actual Ratings')
    plt.legend()
    plt.grid(True)
    plt.show()
