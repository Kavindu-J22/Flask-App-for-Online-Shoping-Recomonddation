import React, { useState } from 'react';
import axios from 'axios';
import './styles/ReviewForm.css';

const ReviewForm = ({ itemId }) => {
  const [title, setTitle] = useState('');
  const [description, setDescription] = useState('');
  const [rating, setRating] = useState(null);
  const [message, setMessage] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();

    const reviewData = {
      title,
      review_text: description, // Updated to match Flask API
    };

    try {
      // Send a POST request to the API to get the prediction
      const response = await axios.post('http://127.0.0.1:5001/predict', reviewData);

      // Set the rating based on the response from the Flask app
      const prediction = response.data.prediction === "Recommended" ? 1 : 0;
      setRating(prediction);

      // Now send the review with the title, description, and received rating to the database
      await axios.post(`/api/items/${itemId}/review`, {
        title,
        description,
        rating: prediction, // Sending the received rating
      });

      alert('Review added successfully!');
      setTitle('');
      setDescription('');
      setRating(null);
      setMessage('');
    } catch (err) {
      console.error(err);
      alert('Error adding review');
    }
  };

  return (
    <form onSubmit={handleSubmit} className="review-form">
      <h3>Add a Review 🔃</h3>
      <input
        type="text"
        value={title}
        onChange={(e) => setTitle(e.target.value)}
        placeholder="Review Title"
        required
      />
      <textarea
        value={description}
        onChange={(e) => setDescription(e.target.value)}
        placeholder="Write your review here..."
        required
      />
      {message && <p className="error-message">{message}</p>}
      <button type="submit">Submit Review</button>
    </form>
  );
};

export default ReviewForm;
