import streamlit as st
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List, Dict
import time
import re
from fake_useragent import UserAgent
import logging

class ProductScraper:
    def __init__(self):
        self.setup_browser()
        
    def setup_browser(self):
        """Setup Selenium WebDriver with appropriate options"""
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # Run in headless mode
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        
        # Add random user agent
        ua = UserAgent()
        chrome_options.add_argument(f'user-agent={ua.random}')
        
        self.driver = webdriver.Chrome(options=chrome_options)
        
    def scrape_amazon_products(self, search_query: str, num_products: int = 10):
        """Scrape product data from Amazon"""
        products = []
        try:
            # Format search query for URL
            search_query = "+".join(search_query.split())
            url = f"https://www.amazon.com/s?k={search_query}"
            
            self.driver.get(url)
            time.sleep(2)  # Allow page to load
            
            # Wait for product elements to be present
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "[data-component-type='s-search-result']"))
            )
            
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            product_cards = soup.find_all("div", {"data-component-type": "s-search-result"})
            
            for card in product_cards[:num_products]:
                try:
                    # Extract product information
                    product = self.extract_product_info(card)
                    if product:
                        products.append(product)
                except Exception as e:
                    logging.error(f"Error extracting product info: {str(e)}")
                    continue
                    
        except Exception as e:
            logging.error(f"Error scraping Amazon: {str(e)}")
            
        return products
    
    def extract_product_info(self, card):
        """Extract product information from a product card"""
        try:
            # Extract name
            name_element = card.find("span", {"class": "a-text-normal"})
            name = name_element.text.strip() if name_element else "N/A"
            
            # Extract price
            price_element = card.find("span", {"class": "a-price-whole"})
            price = float(price_element.text.replace(",", "")) if price_element else 0.0
            
            # Extract rating
            rating_element = card.find("span", {"class": "a-icon-alt"})
            rating = float(rating_element.text.split()[0]) if rating_element else 0.0
            
            # Extract number of reviews
            reviews_element = card.find("span", {"class": "a-size-base"})
            num_reviews = int(re.sub(r'[^0-9]', '', reviews_element.text)) if reviews_element else 0
            
            # Extract image URL
            img_element = card.find("img", {"class": "s-image"})
            image_url = img_element.get("src") if img_element else None
            
            return {
                "name": name,
                "price": price,
                "rating": rating,
                "num_reviews": num_reviews,
                "image_url": image_url,
                "source": "Amazon",
                "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
        except Exception as e:
            logging.error(f"Error in extract_product_info: {str(e)}")
            return None

class ProductRAG:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.product_data = []
        self.index = None
        self.last_update = None
        self.update_interval = 300  # 5 minutes
        self.scraper = ProductScraper()
        
    def fetch_real_time_data(self, search_query: str):
        """Fetch real-time product data using web scraping"""
        products = self.scraper.scrape_amazon_products(search_query)
        return products
    
    def build_index(self):
        """Build FAISS index from current product data"""
        if not self.product_data:
            return
            
        texts = [
            f"{p['name']}" for p in self.product_data
        ]
        embeddings = self.model.encode(texts)
        
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Search products using FAISS"""
        # Fetch fresh data for the search query
        self.product_data = self.fetch_real_time_data(query)
        self.build_index()
        
        if not self.index:
            return []
            
        query_vector = self.model.encode([query])
        distances, indices = self.index.search(
            np.array(query_vector).astype('float32'), k
        )
        
        results = []
        for idx in indices[0]:
            if idx >= 0:
                results.append(self.product_data[idx])
        return results

def main():
    st.title("Real-time E-commerce Product Search")
    
    # Initialize RAG system
    @st.cache_resource
    def get_rag_system():
        return ProductRAG()
    
    rag = get_rag_system()
    
    # Add filters
    st.sidebar.header("Filters")
    price_range = st.sidebar.slider(
        "Price Range ($)", 
        min_value=0, 
        max_value=1000, 
        value=(0, 1000)
    )
    
    min_rating = st.sidebar.slider(
        "Minimum Rating", 
        min_value=0.0, 
        max_value=5.0, 
        value=0.0
    )
    
    # Search interface
    query = st.text_input(
        "Search for products:", 
        placeholder="Enter your search query..."
    )
    
    if query:
        with st.spinner("Searching products..."):
            results = rag.search(query)
            
            # Apply filters
            filtered_results = [
                p for p in results
                if price_range[0] <= p['price'] <= price_range[1]
                and p['rating'] >= min_rating
            ]
            
            if not filtered_results:
                st.warning("No products found matching your criteria.")
            
            for product in filtered_results:
                with st.expander(
                    f"{product['name']} - ${product['price']}", 
                    expanded=True
                ):
                    col1, col2 = st.columns([2, 3])
                    
                    with col1:
                        if product['image_url']:
                            st.image(
                                product['image_url'], 
                                width=150,
                                caption=product['name']
                            )
                    
                    with col2:
                        st.write("**Price:**", f"${product['price']}")
                        st.write(
                            "**Rating:**", 
                            "⭐" * int(product['rating']), 
                            f"({product['rating']})"
                        )
                        st.write("**Number of Reviews:**", product['num_reviews'])
                        st.write("**Source:**", product['source'])
                        st.write("**Last Updated:**", product['last_updated'])

    # Add refresh button
    if st.button("New Search"):
        st.experimental_rerun()

if __name__ == "__main__":
    main()