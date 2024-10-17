import React, { useState } from 'react';
import axios from 'axios';
import './styles/ReviewForm.css';

const ReviewForm = ({ itemId }) => {
  const [title, setTitle] = useState('');
  const [description, setDescription] = useState('');
  const [rating, setRating] = useState(null);
  const [message, setMessage] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    
    if (rating === null) {
      setMessage("Please select a recommendation option.");
      return;
    }

    const reviewData = {
      title,
      description,
      rating
    };

    // Send a POST request to the API to add the review for the specific item
    axios.post(`/api/items/${itemId}/review`, reviewData)
      .then(res => {
        alert('Review added successfully!');
        setTitle('');
        setDescription('');
        setRating(null);
        setMessage('');
      })
      .catch(err => {
        console.error(err);
        alert('Error adding review');
      });
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
      <div className="rating-container">
        <p className='ratethis'>Rate  this item ▶️ </p>

        <span
          className={`rating-icon ${rating === 1 ? 'selected' : ''}`}
          onClick={() => setRating(1)}
          role="button"
          aria-label="Recommend"
        >
          👍 |
        </span>
        <span
          className={`rating-icon ${rating === 0 ? 'selected' : ''}`}
          onClick={() => setRating(0)}
          role="button"
          aria-label="Not Recommend"
        >
          👎
        </span>
      </div>
      {message && <p className="error-message">{message}</p>}
      <button type="submit">Submit Review</button>
    </form>
  );
};

export default ReviewForm;
