import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import axios from 'axios';
import './styles/ItemList.css'; 

const ItemList = () => {
  const [items, setItems] = useState([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [filteredItems, setFilteredItems] = useState([]);

  useEffect(() => {
    axios.get('/api/items')
      .then(res => {
        setItems(res.data);
        setFilteredItems(res.data);  // Set initial filtered items to all items
      })
      .catch(err => console.error(err));
  }, []);

  // Handle search input change
  const handleSearchChange = (e) => {
    const keyword = e.target.value;
    setSearchTerm(keyword);
    
    // Filter items based on the search term
    const filtered = items.filter(item =>
      item.title.toLowerCase().includes(keyword.toLowerCase()) || // Match by title
      item.category.toLowerCase().includes(keyword.toLowerCase()) // Match by category
    );
    
    setFilteredItems(filtered);
  };

  return (
    <div className="item-list">
      <h2>Available Clothing Items</h2>
      
      <input
        type="text"
        placeholder="Search by title or category..."
        value={searchTerm}
        onChange={handleSearchChange}
        className="search-input"
      />
      
      <p className="item-count">
        {filteredItems.length} {filteredItems.length === 1 ? 'item' : 'items'} found.
      </p>
      
      <div className="card-container">
        {filteredItems.map(item => (
          <div className="card" key={item._id}>
            <img src={item.imageUrl} alt={item.title} className="card-image" />
            <div className="card-content">
              <h3 className="card-title">{item.title}</h3>
              <p className="card-category">{item.category}</p>
              <p className="card-description">{item.description}</p>
              <p className="card-price">LKR {item.price.toFixed(2)}</p>
              <Link to={`/item/${item._id}`} className="view-more-btn">View More</Link>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default ItemList;
