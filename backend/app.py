from flask import Flask, jsonify, request
from flask_cors import CORS
from pymongo import MongoClient
import os
from dotenv import load_dotenv
from bson.objectid import ObjectId  # Import ObjectId for ID handling

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests (React <-> Flask)

# Connect to MongoDB
client = MongoClient(os.getenv("MONGO_URI"))
db = client['clothing_store']  # Database name


# Define a route to fetch all clothing items

@app.route('/api/items', methods=['GET'])
def get_items():
    items_collection = db['items']  # Collection name
    items = list(items_collection.find({}))  # Retrieve all items
    for item in items:
        item['_id'] = str(item['_id'])  # Convert ObjectId to string
    return jsonify(items)


# Define a route to fetch a specific clothing item by ID, including its reviews

@app.route('/api/items/<item_id>', methods=['GET'])
def get_item(item_id):
    items_collection = db['items']
    item = items_collection.find_one({"_id": ObjectId(item_id)})  # Find item by ID
    if item:
        item['_id'] = str(item['_id'])  # Convert ObjectId to string
        # Fetch reviews associated with this item
        reviews_collection = db['reviews']
        reviews = list(reviews_collection.find({"item_id": item_id}))
        for review in reviews:
            review['_id'] = str(review['_id'])  # Convert ObjectId to string
        item['reviews'] = reviews  # Add reviews to the item
        return jsonify(item)  # Return the item details along with reviews
    else:
        return jsonify({"error": "Item not found"}), 404  # Handle item not found case


# Define a route to add a new clothing item

@app.route('/api/items', methods=['POST'])
def add_item():
    item_data = request.json  # Get the item data from the request body
    db.items.insert_one(item_data)  # Insert the item into the 'items' collection
    return jsonify({"message": "Item added successfully!"}), 201


# Define a route to add a new review

@app.route('/api/items/<item_id>/review', methods=['POST'])
def add_review(item_id):
    reviews_collection = db['reviews']
    review_data = request.json
    review_data['item_id'] = item_id  # Store the item ID with the review

    # Insert the review into the 'reviews' collection
    result = reviews_collection.insert_one(review_data)
    review_id = str(result.inserted_id)  # Get the ID of the new review

    # Update the item's reviews array in the 'items' collection
    items_collection = db['items']
    items_collection.update_one(
        {"_id": ObjectId(item_id)},
        {"$push": {"reviews": review_id}}  # Add the new review ID to the item's reviews array
    )

    return jsonify({"message": "Review added successfully!"}), 201


if __name__ == '__main__':
    app.run(debug=True)
