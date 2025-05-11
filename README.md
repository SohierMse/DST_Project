# DST_Project
a project i made with my team to apply (WebScrapping-Analysis and Streamlit)
## First WebScrapping:
## We made the scrapping on jumia
# Jumia Product Scraping

This project is a web scraping tool that extracts product details from various categories on Jumia (Egypt's leading online marketplace). The tool gathers information such as product names, prices, discounts, ratings, reviews, sellers, and shipping details. The data is then saved in both JSON and CSV formats for further analysis.

## Features

- Scrapes multiple product categories from Jumia, such as smartphones, laptops, and home appliances.
- Extracts key product details, including:
  - Product name
  - Price (current and original if discounted)
  - Discount percentage
  - Rating and review count
  - Brand
  - Seller information
  - Shipping details
  - Product URL
- Handles multiple pages within a category (up to 50 pages).
- Randomized delays between requests to avoid server overload and potential blocks.
- Saves data in JSON format for structured storage and CSV for easy analysis.

## Requirements

- Python 3.x
- `requests` library
- `BeautifulSoup` from `bs4` (for HTML parsing)
- `pandas` (for CSV export)
- `tqdm` (for progress bar during scraping)
- `time`, `random`, and `json` modules (built-in)

To install the required libraries, run:


```bash
pip install requests beautifulsoup4 pandas tqdm
```

---------------------------------------------------------------------------------------------------------------------------------------
How It Works
Session Configuration: The script creates a session with headers mimicking a browser request to avoid being blocked by Jumia's website.

Scraping Process: For each category, the script makes GET requests to retrieve product pages and extract product information using BeautifulSoup.

Data Extraction: The script extracts essential product data such as name, price, discount, ratings, and more. It stores each product's details in a dictionary.

Saving Data: The data is stored incrementally in JSON files and saved as a CSV for easy analysis.
-----------------------------------------------------------------------------------------------------------------------------------------
Limitations
* Website Blocking: Although delays are introduced to prevent overwhelming the server, websites may still block access if they detect suspicious activity.

* Dynamic Content: Some dynamic content might not be accessible via simple HTML scraping. The current setup works for static content on Jumia, but may not handle content loaded via JavaScript.

* Robustness: The scraper assumes the website's structure remains consistent. Any significant changes to Jumia’s page layout may break the script.
-------------------------------------------------------------------------------------------------------------------------------------------------------

# 🛒 Streamlit Shopping Cart App

This is a **Streamlit-based web application** that allows users to browse a product catalog, view product details, and manage a shopping cart — all within a user-friendly interface.

---

## 🚀 Features

- **Product Listing**: Displays available products with names, images, and prices.
- **View Details**: Navigate to a product-specific page for more information.
- **Add to Cart**: Add items to the shopping cart from both the product list and product detail pages.
- **Cart Management**:
  - Quantity updates (e.g., *2 if added twice)
  - Remove individual items
  - Total price calculation

---

## 📦 Requirements

- Python 3.8+
- `streamlit`
- `Pillow` (for image handling)

Install dependencies using:

```bash
pip install streamlit Pillow
```

---

## 🏁 How to Run

```bash
streamlit run app.py
```

Then open the provided local URL in your browser (e.g., http://localhost:8501).

---

## 📁 Project Structure

```
.
├── app.py             # Main Streamlit application
└── README.md          # This file
```

Make sure to include an `images` folder in the same directory with the appropriate image files for each product.

---

---

## 🧑‍💻 Author

Built with ❤️ BY:


Sohier Mohamed Salah Eldin


Sondos Yasser 


Mariam Salah 


using [Streamlit](https://streamlit.io/)  
