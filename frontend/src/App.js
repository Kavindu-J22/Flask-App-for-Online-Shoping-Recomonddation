import React from 'react';
import { Routes, Route } from 'react-router-dom';
import ItemList from './components/ItemList';
import ItemDetails from './components/ItemDetails';
import './App.css';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>Clothing Store</h1>
        <p>Your one-stop shop for fashionable clothing</p>
      </header>
      <main>
        <Routes>
          <Route path="/" element={<ItemList />} />
          <Route path="/item/:id" element={<ItemDetails />} />
        </Routes>
      </main>
      <footer className="App-footer">
        <p>&copy; {new Date().getFullYear()} Clothing Store. All rights reserved.</p>
      </footer>
    </div>
  );
}

export default App;
