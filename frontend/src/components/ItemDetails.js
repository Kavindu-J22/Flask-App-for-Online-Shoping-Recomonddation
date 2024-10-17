import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import axios from 'axios';
import ReviewForm from './ReviewForm';
import './styles/ItemDetails.css'; 
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faThumbsUp, faThumbsDown } from '@fortawesome/free-solid-svg-icons';


const ItemDetails = () => {
  const { id } = useParams();
  const [item, setItem] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    axios.get(`/api/items/${id}`)
      .then(res => {
        setItem(res.data);
        setLoading(false);
      })
      .catch(err => {
        console.error(err);
        setError("Could not fetch item details");
        setLoading(false);
      });
  }, [id]);

  if (loading) return <p className="loading-text">Loading item details...</p>;
  if (error) return <p className="error-text">{error}</p>;

  return (
    <div className="item-details">
      {item ? (
        <>
          <h2 className="item-title">{item.title}</h2>
          <img src={item.imageUrl} alt={item.title} className="item-image" />
          <p className="item-description">{item.description}</p>
          <p className="item-price">LKR {item.price.toFixed(2)}</p>
          
          <button className="add-to-cart-btn">
            <i className="fa fa-shopping-cart" aria-hidden="true"></i> Add to Cart
          </button>
          
          <h3>Reviews 🌟</h3>
          {item.reviews.length > 0 ? (
            <ul className="reviews-list">
            {item.reviews.map(review => (
              <li key={review._id} className="review-item">
                <p className="review-title"><strong>Title:</strong> {review.title}</p>
                <p className="review-description"><strong>Review:</strong> {review.description}</p>
                <p className="review-rating">
                  <strong>Rating:</strong> 
                  {review.rating === 1 ? (
                    <span className="rating-icon" aria-label="Recommended">
                      <FontAwesomeIcon icon={faThumbsUp} className="icon recommended" />
                    </span>
                  ) : (
                    <span className="rating-icon" aria-label="Not Recommended">
                      <FontAwesomeIcon icon={faThumbsDown} className="icon not-recommended" />
                    </span>
                  )}
                </p>
              </li>
            ))}
          </ul>
          ) : (
            <p>No reviews yet.</p>
          )}
          
          <ReviewForm itemId={id} />
        </>
      ) : (
        <p>Item not found.</p>
      )}
    </div>
  );
};

export default ItemDetails;
