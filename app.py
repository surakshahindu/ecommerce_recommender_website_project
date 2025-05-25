from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)

# --- 1. Simulated E-commerce Dataset ---
products_data = {
    'id': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
           111, 112, 113, 114, 115, 116, 117, 118, 119, 120], # 20 IDs
    'name': [
        'Luxury Leather Wallet', 'Smart Casual Shirt', 'High-Performance Blender',
        'Noise-Cancelling Headphones', 'Ergonomic Office Chair', 'Gaming Laptop Pro',
        'Organic Coffee Beans (1kg)', 'Wireless Charging Pad', 'Portable Bluetooth Speaker',
        'Vintage Collectible Watch', 'Ceramic Mug Set (4)', 'Professional Camera Drone',
        'Gourmet Chocolate Box', 'Fitness Tracker Watch', 'Yoga Mat Premium',
        'Stainless Steel Water Bottle', 'Handcrafted Soap Set', 'Childrens Educational Toy',
        'Designer Backpack', 'Aromatherapy Diffuser' # 20 Names
    ],
    'category': [
        'Accessories', 'Apparel', 'Home Appliances',
        'Electronics', 'Furniture', 'Electronics',
        'Food & Beverage', 'Electronics', 'Electronics',
        'Accessories', 'Home Goods', 'Electronics',
        'Food & Beverage', 'Wearable Tech', 'Sports & Outdoors',
        'Home Goods', 'Personal Care', 'Toys',
        'Bags', 'Home Goods' # 20 Categories
    ],
    # IMPORTANT: These paths are relative to your 'static' folder.
    # You MUST place your actual image files in the 'static//' folder within your project.
    # Ensure filenames exactly match these paths (including case and extension).
    'image_url': [
    "static/luxury_leather_wallet.jpg",
    "static/smart_casual_shirt.jpg",
    "static/high_performance_blender.jpg",
    "static/noise_cancelling_headphones.jpg",
    "static/ergonomic_office_chair.jpg",
    "static/gaming_laptop_pro.jpg",
    "static/organic_coffee_beans_1kg.jpg",
    "static/wireless_charging_pad.jpg",
    "static/portable_bluetooth_speaker.jpg",
    "static/vintage_collectible_watch.jpg",
    "static/ceramic_mug_set_4.jpg",
    "static/professional_camera_drone.jpg",
    "static/gourmet_chocolate_box.jpg",
    "static/fitness_tracker_watch.jpg",
    "static/yoga_mat_premium.jpg",
    "static/stainless_steel_water_bottle.jpg",
    "static/handcrafted_soap_set.jpg",
    "static/childrens_educational_toy.jpg",
    "static/designer_backpack.jpg",
    "static/aromatherapy_diffuser.jpg"
]

}
products_df = pd.DataFrame(products_data)

ratings_data = [
    {'userId': 1, 'productId': 104, 'rating': 5}, {'userId': 1, 'productId': 106, 'rating': 4},
    {'userId': 1, 'productId': 108, 'rating': 5}, {'userId': 1, 'productId': 109, 'rating': 3},
    {'userId': 1, 'productId': 101, 'rating': 4}, {'userId': 1, 'productId': 107, 'rating': 5},
    {'userId': 1, 'productId': 114, 'rating': 4},

    {'userId': 2, 'productId': 103, 'rating': 5}, {'userId': 2, 'productId': 111, 'rating': 4},
    {'userId': 2, 'productId': 116, 'rating': 5}, {'userId': 2, 'productId': 120, 'rating': 4},
    {'userId': 2, 'productId': 104, 'rating': 3}, {'userId': 2, 'productId': 109, 'rating': 4},

    {'userId': 3, 'productId': 102, 'rating': 5}, {'userId': 3, 'productId': 119, 'rating': 4},
    {'userId': 3, 'productId': 110, 'rating': 5}, {'userId': 3, 'productId': 101, 'rating': 3},
    {'userId': 3, 'productId': 118, 'rating': 4},

    {'userId': 4, 'productId': 106, 'rating': 5}, {'userId': 4, 'productId': 112, 'rating': 4},
    {'userId': 4, 'productId': 107, 'rating': 3}, {'userId': 4, 'productId': 113, 'rating': 4},
    {'userId': 4, 'productId': 115, 'rating': 5}, {'userId': 4, 'productId': 104, 'rating': 4},

    {'userId': 5, 'productId': 111, 'rating': 3}, {'userId': 5, 'productId': 116, 'rating': 4},
    {'userId': 5, 'productId': 117, 'rating': 5}, {'userId': 5, 'productId': 120, 'rating': 3},
    {'userId': 5, 'productId': 105, 'rating': 2}
]
ratings_df = pd.DataFrame(ratings_data)

# Get unique user IDs for the frontend dropdown
unique_user_ids = sorted(ratings_df['userId'].unique().tolist())

# --- 2. Data Preprocessing: Create User-Product Interaction Matrix ---
user_product_matrix = ratings_df.pivot_table(index='productId', columns='userId', values='rating')
user_product_matrix_filled = user_product_matrix.fillna(0)

# --- 3. Calculate Item-Item Similarity (Cosine Similarity) ---
item_similarity_matrix = cosine_similarity(user_product_matrix_filled)
item_similarity_df = pd.DataFrame(
    item_similarity_matrix,
    index=user_product_matrix.index,
    columns=user_product_matrix.index
)

# --- 4. Personalized Recommendation Function ---
def get_personalized_recommendations(user_id, num_recommendations=5):
    """
    Generates personalized product recommendations for a given user using
    Item-Based Collaborative Filtering.
    """
    user_rated_products = user_product_matrix.loc[:, user_id].dropna()
    unrated_product_ids = [pid for pid in products_df['id'].unique() if pid not in user_rated_products.index]

    predicted_ratings = {}

    for target_product_id in unrated_product_ids:
        numerator = 0
        denominator = 0

        similar_products_to_target = item_similarity_df.loc[target_product_id, user_rated_products.index]

        for rated_product_id, user_rating_for_rated_product in user_rated_products.items():
            similarity = similar_products_to_target.get(rated_product_id, 0)
            if similarity > 0:
                numerator += similarity * user_rating_for_rated_product
                denominator += similarity

        if denominator > 0:
            predicted_rating = numerator / denominator
            predicted_ratings[target_product_id] = predicted_rating

    if not predicted_ratings:
        return pd.DataFrame(columns=['productId', 'name', 'category', 'predicted_rating'])

    recommended_products_df = pd.DataFrame(
        list(predicted_ratings.items()),
        columns=['productId', 'predicted_rating']
    )

    # Merge with product details for display, including image_url
    recommended_products_df = recommended_products_df.merge(products_df[['id', 'name', 'category', 'image_url']],
                                                        left_on='productId', right_on='id')
    recommended_products_df = recommended_products_df.drop(columns=['id'])
    recommended_products_df = recommended_products_df.sort_values(by='predicted_rating', ascending=False)

    return recommended_products_df.head(num_recommendations)

# --- Flask Routes ---

@app.route('/')
def index():
    """Renders the main recommendation UI."""
    # Print the path Flask is looking for templates for debugging
    print(f"Flask is looking for templates in: {app.template_folder}")
    index_html_path = os.path.join(app.template_folder, 'index.html')
    print(f"Does index.html exist at {index_html_path}? {os.path.exists(index_html_path)}")

    return render_template('index.html', unique_user_ids=unique_user_ids)

@app.route('/get_recommendations', methods=['POST'])
def get_recommendations_endpoint():
    """
    API endpoint to receive user_id and return personalized recommendations.
    """
    user_id = int(request.form['user_id']) # Get user_id from the form submission

    # Get products user has already rated for display
    user_rated_products_list = []
    current_user_ratings = ratings_df[ratings_df['userId'] == user_id]
    for _, row in current_user_ratings.iterrows():
        product_info = products_df[products_df['id'] == row['productId']].iloc[0]
        user_rated_products_list.append({
            'name': product_info['name'],
            'rating': int(row['rating']), # Explicitly convert to Python int
            'category': product_info['category'], # Include category for rated products
            'image_url': product_info['image_url'] # Include image_url for rated products
        })

    # Get personalized recommendations
    recommendations_df = get_personalized_recommendations(user_id)

    # Ensure all numeric types in the DataFrame are standard Python types before converting to dict
    recommendations_df['productId'] = recommendations_df['productId'].astype(int)
    recommendations_df['predicted_rating'] = recommendations_df['predicted_rating'].astype(float)

    recommendations_list = recommendations_df.to_dict(orient='records')

    # Return both rated products and recommendations as JSON
    return jsonify({
        'user_rated': user_rated_products_list,
        'recommendations': recommendations_list
    })

if __name__ == '__main__':
    app.run(debug=True)