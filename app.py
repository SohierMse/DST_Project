# # **Data Science  Tools Project**

# #### Consists of :
# ##### 1.`Data Extraction` (Web Scraping)
# ##### 2.`Store the data` in a file
# ##### 3.`Data Cleaning `
# ##### 4.`Data PreProcessing `
# ##### 5.`Regular Expressions`
# ##### 6`.Data Analysis`
# ##### 7.`Data Visualization`
# ##### 8.`Data Storage`
# ##### BOUNS: `Streamlit`(to represent the results in interactive WebPage)

## we will make the scraping on: JUMIA

import pandas as pd
import numpy as np
import time
import random
import re
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import hashlib

dff=pd.read_csv('jumia_data.csv')

df=dff.copy()

df.drop( [ 'seller', 'shipping' , 'url' ], axis=1 , inplace=True)

# ### That is mean the same product exist more that once with different details

  

df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)


null_columns= ['original_price' ,'discount','rating','review_count']
null_percentage=df [null_columns ].isnull().mean()*100


# ### show high percentage of missing values .

# First, clean and convert the column safely
df['clean_price'] = df['price'].str.replace(r'[^\d.]', '', regex=True) #Removes any character that is not a digit (\d) or a dot (.)
df['clean_price'] = pd.to_numeric(df['clean_price'], errors='coerce') #Forces invalid conversions to NaN instead of raising an error.


# Show rows where conversion failed (i.e., became NaN)
nan_entries = df[df['clean_price'].isna()]



# Example: show the original values that caused NaN after conversion
df['clean_original_price'] = pd.to_numeric(
    df['original_price'].str.replace(r'[^\d.]', '', regex=True), 
    errors='coerce'
)
# the values which isn't nan before converting to numeric and becomes nan after the converting
conversion_failed = df[df['clean_original_price'].isna() & df['original_price'].notna()]

def to_float(val):
    try:
        return float(val.strip().replace('EGP', '').replace(',', '')) # removes (EGP) and comma and any spaces then convert to float
    except:
        return np.nan # if the input is nan

def is_range(val):
    return isinstance(val, str) and '-' in val # Returns True if it’s a string containing -, meaning it’s a range

def get_avg(val):
    try:
        numbers = [to_float(p) for p in val.split('-')] # If you have " 100 -  150", this:Splits it into ["100", "150"] 
                                                        #Converts both to floats using to_float(): → [100.0, 150.0]
        return sum(numbers) / len(numbers) #Takes the average → (100 + 150) / 2 = 125.0
    except:
        return np.nan

def clean_discount(discount):
    try:
        return float(str(discount).replace('%', '').strip()) # removes (%) and any spaces then convert to float
    except:
        return np.nan

#It checks each row’s price, original price, and discount 
def smart_price_clean(row):
    price = row['price']
    original = row['original_price']
    discount = clean_discount(row['discount'])

    if is_range(price) and not is_range(original):
        # original is a single value → estimate price
        original_val = to_float(original)
        if pd.notna(original_val) and pd.notna(discount):
            return original_val * (1 - discount / 100)
        else:
            return get_avg(price)

    elif not is_range(price) and is_range(original):
        # price is a single value → estimate original
        price_val = to_float(price)
        if pd.notna(price_val) and pd.notna(discount):
            return price_val  # keep price as is
        else:
            return np.nan

    elif is_range(price) and is_range(original):
        # take average of price range
        return get_avg(price)

    else:
        return to_float(price)

def smart_original_clean(row):
    price = row['price']
    original = row['original_price']
    discount = clean_discount(row['discount'])

    if not is_range(price) and is_range(original):
        # price is single → estimate original
        price_val = to_float(price)
        if pd.notna(price_val) and pd.notna(discount):
            return price_val / (1 - discount / 100)
        else:
            return get_avg(original)

    elif is_range(price) and not is_range(original):
        # original is single → keep as is
        return to_float(original)

    elif is_range(price) and is_range(original):
        # both are ranges → average
        return get_avg(original)

    else:
        return to_float(original)

def smart_discount_calc(row):
    price = row['clean_price']
    original = row['clean_original_price']
    if pd.notna(price) and pd.notna(original):
        return round(100 * (1 - price / original), 2)
    return np.nan

# Apply corrected logic
df['clean_price'] = df.apply(smart_price_clean, axis=1)
df['clean_original_price'] = df.apply(smart_original_clean, axis=1)
df['clean_discount'] = df.apply(smart_discount_calc, axis=1)


df['price'] = df['clean_price']
df['original_price'] = df['clean_original_price']
df['discount'] = df['clean_discount']
df.drop(columns=['clean_price', 'clean_original_price','clean_discount' ], inplace=True)

df[ 'review_count'].isna().sum()


# Step 1: Convert review_count to string 
df['clean_review_count'] = df['review_count'].astype(str)

# Step 2: Extract only the number inside the parentheses using regex
df['clean_review_count'] = df['clean_review_count'].str.extract(r'\((\d+)\)')

# Step 3: Convert to numeric (int), while keeping NaNs for any missing values
df['clean_review_count'] = pd.to_numeric(df['clean_review_count'], errors='coerce')

df[ 'clean_review_count'].isna().sum()

df [ 'review_count']=df[ 'clean_review_count' ]
df.drop(columns=['clean_review_count' ], inplace=True)

# Step 1: Convert to string 
df['rating'] = df['rating'].astype(str)

# Step 2: Use regex to extract the number before "out of 5"
df['rating'] = df['rating'].str.extract(r'(\d+\.?\d*)')

# Step 3: Convert to float 
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

df['brand'] = df['brand'].apply(
    lambda x: x.capitalize() 
    if isinstance(x, str) and len(x) > 3
    else x
)

# Get value counts, including NaN, and filter values that appear one time
counts = df['brand'].value_counts(dropna=False)

# Filter to show only values that appear once
filtered_counts = counts[counts ==1]

# Filter the dataframe for rows where the 'brand' value appears only once
filtered_rows = df[df['brand'].isin(filtered_counts.index)] #filtered_counts.index gives us just the brand names (not the counts)

# Display only 'name' and 'brand' columns
filtered_rows = filtered_rows[['name', 'brand', 'category']]

def has_arabic(text):
    return bool(re.search(r'[\u0600-\u06FF]', str(text)))

def has_digits(text):
    return bool(re.search(r'\d', str(text)))

# Step 1: Define the mask again
mask = df['brand'].apply(lambda x: has_arabic(x) or has_digits(x))

# Step 2: Filter suspicious entries
suspicious_df = df[mask & df['brand'].notna()]

# Step 3: Count suspicious brand occurrences
brand_counts = suspicious_df['brand'].value_counts()

# Step 4: Keep only those with more than 10 occurrences
suspicious_brands = brand_counts

 # Save the original brand column before replacement
df['original_brand'] = df['brand']

# Define suspicious brands 
unknown_brands = set(brand_counts.index)

# Perform the replacement
df['brand'] = df['brand'].apply(
    lambda x: 'Unknown' if x in unknown_brands else x
)

# Count actual replacements (where value changed to 'Unknown')
replaced_to_unknown = ((df['brand'] == 'Unknown') & (df['original_brand'] != 'Unknown')).sum()


df.drop(columns='original_brand', inplace=True)

def has_special_chars(text):
    return bool(re.search(r'[^\w\s]', str(text)))  # Excludes letters, digits, underscore

def is_short(text):
    return len(str(text).strip()) < 2

# Apply filter
mask = df['brand'].apply(lambda x:  has_special_chars(x) or is_short(x))
suspicious_brands = df[mask]['brand'].dropna().unique()

suspicious_df = df[mask & df['brand'].notna()]  # Filtered suspicious entries
# Count occurrences of each suspicious brand
brand_counts = suspicious_df['brand'].value_counts()

# Filter brands with more than 5 occurrences
frequent_suspicious = brand_counts[brand_counts > 15]

# Brands to keep
frequent_suspicious = set(brand_counts[brand_counts > 15].index)

# Update the brand column
df['brand'] = df['brand'].apply(
    lambda x: x if not has_special_chars(x) and not is_short(x) else (x if x in frequent_suspicious else 'Unknown')
)

replaced_to_unknown = ((df['brand'] == 'Unknown')).sum()
#replaced_to_unknown

# Find brand-category groups where ALL discount values are NaN
all_nan_groups = df.groupby(['brand', 'category'])['discount'].apply(lambda x: x.isna().all())

# Filter to only the problematic ones
problematic_groups = all_nan_groups[all_nan_groups].index.tolist()


# ## For those groups where all discount values are missing.

# Step 1: Save brands that have missing discounts
brands_with_missing_discount = df[df['discount'].isna()]['brand'].unique()

# Step 2: Fill missing discounts
def fill_discount(group):
    missing_mask = group['discount'].isna() #Creates a mask (True/False) to identify rows where discount is missing within that specific group.

    if missing_mask.any(): #Checks if there are any missing values in the group. If not, we don’t need to do anything.
        mode_discount = group['discount'].mode() #Calculates the mode of the discount column in this group.
        
        if not mode_discount.empty: # If there's a valid mode
            group.loc[missing_mask, 'discount'] = mode_discount.iloc[0]  # Fill missing discounts with first mode

        else:
            # If no non-null discount in the group, fill with 0
            group.loc[missing_mask, 'discount'] = 0
            group.loc[missing_mask, 'original_price'] = group.loc[missing_mask, 'price']
    
    return group

# Step 3: Apply by brand and category
df = df.groupby(['brand', 'category'], group_keys=False).apply(fill_discount).reset_index(drop=True)

# Step 4: Recalculate original_price if still missing and discount < 100
condition = df['original_price'].isna() & df['discount'].notna() & df['price'].notna()
safe_condition = condition & (df['discount'] < 100)

df.loc[safe_condition, 'original_price'] = (
    df.loc[safe_condition, 'price'] * 100 / (100 - df.loc[safe_condition, 'discount'])
)


# Recreate the DataFrame with value counts and percentages
category_counts = df['category'].value_counts()
category_percentages = (category_counts / len(df)) * 100
category_df = category_percentages.reset_index()
category_df.columns = ['Category', 'Percentage']

# Define category-specific thresholds based on the KDE plot
category_thresholds = {
    'cameras': {'low': 6.0, 'medium': 8.0},
    'computing-accessories': {'low': 5.5, 'medium': 7.5},
    'mobile-accessories': {'low': 4.0, 'medium': 6.0},
    'women-watches': {'low': 5.0, 'medium': 7.0},
    'kitchen-appliances': {'low': 5.0, 'medium': 7.0},
    'televisions': {'low': 7.0, 'medium': 9.0},
    'laptops': {'low': 7.5, 'medium': 9.5},
    'smartphones': {'low': 7.0, 'medium': 9.0}
}

def categorize_price(row):
    category = row['category']  # assuming you have a 'category' column
    price = row['price']
    log_price = np.log(price)
    
    thresholds = category_thresholds.get(category, {'low': 5.5, 'medium': 7.5})
    ##########
    if log_price < thresholds['low']:
        return 'low'
    elif log_price < thresholds['medium']:
        return 'medium'
    else:
        return 'high'

# Apply the categorization
df['price_category'] = df.apply(categorize_price, axis=1)

def get_price_ranges_actual_egp(df, category_col='category', price_col='price'):
    """
    Calculate low/medium/high price ranges in actual EGP (no log transformation)
    
    Returns:
        - DataFrame with price ranges
        - Dictionary of thresholds for categorization
    """
    results = []
    threshold_dict = {}
    
    for category in df[category_col].unique():
        # Get prices for this category
        prices = df[df[category_col] == category][price_col]
        prices = prices[prices > 0]  # Remove invalid prices
        
        # Calculate direct percentiles in EGP
        low_max = prices.quantile(0.33)
        medium_max = prices.quantile(0.66)
        
        # Store results
        results.append({
            'category': category,
            'low_range': f"< {low_max:,.2f} EGP",
            'medium_range': f"{low_max:,.2f} - {medium_max:,.2f} EGP",
            'high_range': f">= {medium_max:,.2f} EGP",
            'low_threshold': low_max,
            'medium_threshold': medium_max
        })
        
        threshold_dict[category] = {
            'low_max': low_max,
            'medium_max': medium_max
        }
    
    return pd.DataFrame(results), threshold_dict

# Usage
price_ranges, thresholds = get_price_ranges_actual_egp(df)

# Recreate the DataFrame with value counts and percentages
category_counts = df['price_category'].value_counts()
category_percentages = (category_counts / len(df)) * 100
price_category_df = category_percentages.reset_index()
price_category_df.columns = ['Price Category', 'Percentage']

# Step 1: Log transform the features
df['log_price'] = np.log1p(df['price'])
df['log_original_price'] = np.log1p(df['original_price'])
df['log_review_count'] = np.log1p(df['review_count'])
df['log_discount'] = np.log1p(df['discount'])

price_by_category = df.groupby("category")["log_price"].mean()

top_brands = df['brand'].value_counts().nlargest(20).index
df['brand'] = df['brand'].apply(lambda x: x if x in top_brands else 'Other')
filtered_df = df[~df['brand'].str.lower().isin(['other', 'unknown'])]
brand_price_sum = filtered_df.groupby('brand')['price'].sum().sort_values(ascending=False)

# Reset index for Plotly
brand_price_df = brand_price_sum.reset_index()
brand_price_df.columns = ['Brand', 'Total Sales']

# # **Discount Analysis**

# This calculates how much discount is being given on each product 
df['price_ratio'] = df['price'] / df['original_price']

# Calculate the threshold (Q3) from review_count
threshold = df['price_ratio'].quantile(0.25)
######################3

# Filter the high and low discounts based on the threshold
high_discount = df[df['price_ratio'] <= threshold]['discount']
low_discount = df[df['price_ratio'] > threshold]['discount']

# ## That is means less than 34% is low discount and above is high.

df['price_ratio_category'] = df['price_ratio'].apply(lambda x: 'low discount' if x > threshold else 'high discount')

# Recreate the DataFrame with value counts and percentages
category_counts = df['price_ratio_category'].value_counts()
category_percentages = (category_counts / len(df)) * 100
price_ratio_category_df = category_percentages.reset_index()
price_ratio_category_df.columns = ['Price ratio Category', 'Percentage']



# Calculate the threshold (Q3) from review_count
threshold = df['review_count'].quantile(0.75)
##########################

df['review_category'] = pd.Series(np.where(
    df['review_count'] >= threshold, 'high', 'low'
), index=df.index)

# Replace missing values explicitly
df.loc[df['review_count'].isna(), 'review_category'] = np.nan

# Filter out missing values first
valid_reviews = df.dropna(subset=['review_count'])

# Get counts and percentages
counts = valid_reviews['review_category'].value_counts()
percentages = valid_reviews['review_category'].value_counts(normalize=True).mul(100).round(1) #normalize converts counts to proportions (0.75 for 75%)



# Show NaN count for transparency
nan_count = df['review_count'].isna().sum()

# ### most have low review but there are extreme high reviews

#  Engagement score
df['engagement_score'] = df['rating'] * np.log1p(df['review_count'])

# to convert the range into 0 and 1.
scaler = MinMaxScaler()
df[['norm_engagement']] = scaler.fit_transform(df[['engagement_score']])

df['best_seller_score'] = df['norm_engagement']
# choosing only top 10% of the high scores
threshold = df['best_seller_score'].quantile(0.90)
####
df['is_best_seller'] = df['best_seller_score'] >= threshold

best_sellers = df[df['is_best_seller']].sort_values(by='best_seller_score', ascending=False)


# #  Top 10 `best sellers` in the store

# Filter best sellers in smartphones category
best_sellers_smartphones = df[(df['is_best_seller']) & (df['category'] == 'smartphones')].sort_values(by='best_seller_score', ascending=False)

# ## `Best sellers` in `SmartPhones`
#show full row length
#show full row length
pd.set_option('display.max_colwidth', None)

recommender_df = df[['category','brand','name','price','rating','discount','original_price']]
recommender_df['product_id'] = range(len(df))  # Unique ID

# Create a combined text feature for similarity analysis
recommender_df['combined_features'] = recommender_df['brand'] + ' ' + recommender_df['category'] + ' ' + recommender_df['name']

# Initialize TF-IDF Vectorizer
tfidf = TfidfVectorizer(ngram_range=(1,2), stop_words='english')

# Fit and transform the combined features
tfidf_matrix = tfidf.fit_transform(recommender_df['combined_features'])

#Compute cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def process_user_input(user_input, recommender_df):
    """
    Process user input to find the most relevant product matches
    Handles shorthand names, brands, or categories
    """
    # Normalize input
    user_input = user_input.lower().strip()

    # Initialize results
    matches = []

    # Check if input matches any product's full name (partial match)
    name_matches = recommender_df[recommender_df['name'].str.lower().str.contains(user_input, regex=False)]
    if not name_matches.empty:
        matches.extend(name_matches['product_id'].tolist())

    # Check if input matches any brand
    brand_matches = recommender_df[recommender_df['brand'].str.lower() == user_input]
    if not brand_matches.empty:
        matches.extend(brand_matches['product_id'].tolist())

    # Check if input matches any category
    category_matches = recommender_df[recommender_df['category'].str.lower() == user_input]
    if not category_matches.empty:
        matches.extend(category_matches['product_id'].tolist())

    # If no matches found, try fuzzy matching on product names
    if not matches:
        for idx, row in recommender_df.iterrows():
            # Extract model number/identifier if exists (e.g., "A06" from "Samsung Galaxy A06")
            model_num = re.search(r'(\b[a-z]\d+\b)', row['name'].lower())
            if model_num and user_input in model_num.group(1):
                matches.append(row['product_id'])

    return list(set(matches))  # Remove duplicates

#!pip install fuzzywuzzy
from fuzzywuzzy import fuzz, process

def get_enhanced_recommendations(user_input, recommender_df, cosine_sim, top_n=5):
    """
    Enhanced recommender with fuzzy matching for brands/categories
    - Handles typos and partial matches
    """
    # Helper function for fuzzy matching
    def get_best_match(input_str, options, threshold=70):
        match = process.extractOne(input_str.lower(),
                                 [str(x).lower() for x in options],
                                 scorer=fuzz.token_set_ratio,
                                 score_cutoff=threshold)
        return match[0] if match else None

    # 1. Find matching products (exact + fuzzy)
    user_input_lower = user_input.lower().strip()
    matched_product_ids = process_user_input(user_input, recommender_df)

    # If no exact matches, try fuzzy matching on brand/category
    if not matched_product_ids:
        best_brand = get_best_match(user_input, recommender_df['brand'].unique())
        best_category = get_best_match(user_input, recommender_df['category'].unique())

        # Determine which fuzzy match is better
        if best_brand and best_category:
            brand_score = fuzz.token_set_ratio(user_input_lower, best_brand.lower())
            category_score = fuzz.token_set_ratio(user_input_lower, best_category.lower())
            best_match = best_brand if brand_score > category_score else best_category
        else:
            best_match = best_brand or best_category

        if best_match:
            if best_match.lower() in [b.lower() for b in recommender_df['brand'].unique()]:
                matched_product_ids = recommender_df[recommender_df['brand'].str.lower() == best_match.lower()]['product_id'].tolist()
            else:
                matched_product_ids = recommender_df[recommender_df['category'].str.lower() == best_match.lower()]['product_id'].tolist()
        else:
            return "No products found matching your input. Please try a different search term."

    # 2. Determine search context (brand/category/product)
    indices = [recommender_df[recommender_df['product_id'] == pid].index[0] for pid in matched_product_ids]

    # Check if input matches a brand or category (with fuzzy matching)
    best_brand = get_best_match(user_input, recommender_df['brand'].unique())
    best_category = get_best_match(user_input, recommender_df['category'].unique())

    # Determine subset of products to consider
    if best_brand and (not best_category or
                      fuzz.token_set_ratio(user_input_lower, best_brand.lower()) >=
                      fuzz.token_set_ratio(user_input_lower, best_category.lower())):
        subset_df = recommender_df[recommender_df['brand'].str.lower() == best_brand.lower()]
    elif best_category:
        subset_df = recommender_df[recommender_df['category'].str.lower() == best_category.lower()]
    else:
        subset_df = recommender_df

    # 3. Generate recommendations
    recommendation_scores = {}
    subset_indices = subset_df.index.tolist()

    for idx in indices:
        sim_scores = list(enumerate(cosine_sim[idx]))
        for subset_idx in subset_indices:
            score = sim_scores[subset_idx][1]
            if subset_idx not in recommendation_scores or score > recommendation_scores[subset_idx]:
                recommendation_scores[subset_idx] = score

    # 4. Filter and return results
    recommended_products = []
    seen_products = set()

    for i, _ in sorted(recommendation_scores.items(), key=lambda x: x[1], reverse=True):
        product = recommender_df.iloc[i]
        if product['product_id'] not in seen_products:
            recommended_products.append(product)
            seen_products.add(product['product_id'])
            if len(recommended_products) >= top_n:
                break

    return pd.DataFrame(recommended_products) if recommended_products else "No relevant recommendations found."
# Ensure these state variables exist

if 'selected_product' not in st.session_state:
    st.session_state.selected_product = None

# Define default page
if 'Page' not in st.session_state:
    st.session_state.Page = 'Home'

# Function to navigate between pages
def go_to(Page_name):
    st.session_state.Page = Page_name
    st.rerun()



# Custom styles for buttons

st.markdown("""
<style>
.stApp {
    background: linear-gradient(-45deg, #89CFF0, #D8BFD8, #89CFF0, #D8BFD8);
    background-size: 400% 400%;
    animation: gradientBG 3s ease infinite; 
    overflow: hidden;
}
@keyframes gradientBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}
.stButton>button {
    height: 60px;
    width: 200px;
    font-size: 20px;
    border-radius: 10px;
}
.purple-button button {
    background-color: #D8BFD8 !important;
    color: black !important;
}
.blue-button button {
    background-color: #ADD8E6 !important;
    color: black !important;
}
</style>
""", unsafe_allow_html=True)
#### search page
if 'search_input' not in st.session_state:
    st.session_state.search_input = ""

if 'selected_product' not in st.session_state:
    st.session_state.selected_product = None

###   Cart page
if "cart" not in st.session_state:
    st.session_state["cart"] = []  
# Home Page
if st.session_state.Page == 'Home':
    st.header("Home Page")
    st.write("Welcome to our WEB !🥰🥰🥰🥰🥰")
    st.write("Are you a Customer or Business user ?")

  #  st.markdown('</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        with st.container():
            st.markdown('<div class="purple-button">', unsafe_allow_html=True)
            if st.button("Business user"):
                go_to('Business user')
            st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        with st.container():
            st.markdown('<div class="blue-button">', unsafe_allow_html=True)
            if st.button("Customer"):
                go_to('Customer')
            st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.Page == "Features Distributions":
    # Step 2: Create Distribution Plots
    fig = make_subplots(rows=4, cols=2, 
                        subplot_titles=(
                            "Price Histogram (Log)", 
                            "Price Violin (Log)",
                            "Original Price Histogram (Log)", 
                            "Original Price Violin (Log)",
                            "Review Count Histogram (Log)", 
                            "Review Count Violin (Log)",
                            "Discount Histogram (Log)",
                            "Discount Violin (Log)"

                        ))
    # Price
    fig.add_trace(go.Histogram(x=df['log_price'],
            name='log_price', marker_color='orange'), row=1, col=1)
    fig.add_trace(go.Violin(y=df['log_price'],
        name='log_price', box_visible=True, line_color='orange'), row=1, col=2)
    # Original Price
    fig.add_trace(go.Histogram(x=df['log_original_price'],
            name='log_original_price', marker_color='green'), row=2, col=1)
    fig.add_trace(go.Violin(y=df['log_original_price'],
        name='log_original_price', box_visible=True, line_color='green'),
            row=2, col=2)

    # Review Count
    fig.add_trace(go.Histogram(x=df['log_review_count'],
            name='log_review_count', marker_color='purple'), row=3, col=1)
    fig.add_trace(go.Violin(y=df['log_review_count'],
        name='log_review_count', box_visible=True, line_color='purple'), row=3, col=2)

    #Discount
    fig.add_trace(go.Histogram(x=df['log_discount'],
            name='log_discount', marker_color='purple'), row=4, col=1)
    fig.add_trace(go.Violin(y=df['log_discount'], 
    name='log_discount', box_visible=True, line_color='purple'), row=4, col=2)

    # Update layout
    fig.update_layout(height=900, width=1000, 
    title_text="Improved Feature Distributions (Log Scale)", template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("From this graphs:")
    st.markdown("""
    ### 📊 Why We Transformed the Numbers Visually

    Some product data — like **prices** and **number of reviews** — has a few very large values that make everything else look small.

    To fix this, we applied a transformation that **evens out the scale**.  
    This helps us:
    - 📈 **See hidden patterns** more clearly  
    - 🔍 **Compare products fairly**, even when some prices or reviews are extremely high

    Think of it like **zooming in** on the important details.
    """)

    if st.button("Back "):
        go_to('DashBoard')
elif st.session_state.Page == "Price per Category":
    grouped_data = []
    labels = []

    for category in df['category'].unique():
        category_data = df[df['category'] == category]['log_price'].dropna().tolist()
        if category_data:  # Ensure there's data in the list
            grouped_data.append(category_data)
            labels.append(f"{category}")

    # Create KDE plot
    kde_fig = ff.create_distplot(
        grouped_data, 
        group_labels=labels, 
        colors=px.colors.qualitative.Pastel,
        show_hist=False,
        show_rug=False
    )

    kde_fig.update_layout(title="KDE Plot of Log-Price by Category", template='plotly_dark')
    st.plotly_chart(kde_fig, use_container_width=True)
    st.markdown("""
    ### 🔍 Price Insights by Product Category

    Here's what we learned about product pricing:

    - **Smartphones** 💡 are generally the most expensive, with prices peaking at the top.
    - **Laptops** 💻 come next — still high in price, just a bit below smartphones.
    - **Televisions** 📺 have two common price levels — showing both mid-range and high-end options.
    - **Cameras** 📷 show a wide variety in pricing, with many in the mid-to-high range.
    - **Computing accessories** 💾 are more affordable, with most prices in the lower range.
    - **Mobile accessories** 📱 are usually the **least expensive** items.
    - **Kitchen appliances** 🍳 are also low-priced, similar to computing accessories.
    - **Women’s watches** ⌚ fall in the **mid-range** for price.

    This helps us understand which categories tend to have premium vs. budget-friendly products.
    """)

    if st.button("Back "):
        go_to('DashBoard')
elif st.session_state.Page =="Most_Products have low_price":
    # KDE for price
    ff_fig = ff.create_distplot([df['price_ratio']],
            group_labels=['price_ratio'], colors=['#FFA07A'], show_hist=True)
    ff_fig.update_layout(title="KDE Plot of Log-Price", template='plotly_dark')
    st.plotly_chart(ff_fig, use_container_width=True)
    st.markdown("""
    The **KDE (Kernel Density Estimate)** shows that most products have a **high ratio (low discount)**, 
    while only a few have a **low ratio (high discount)**.
    """)

    # Create the pie chart in the same style
    fig = px.pie(
        price_ratio_category_df,
        names='Price ratio Category',
        values='Percentage',
        title=' Product Discount',
        color_discrete_sequence=px.colors.sequential.Magenta_r,
        template='plotly_dark'
    )
    # Customize trace appearance
    fig.update_traces(
        textinfo='percent+label',
        pull=[0.05] * len(price_category_df)  # add a small pull for all slices
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    💡 **Only a small portion of products have big discounts** —  
    just **25%** of all items.  
    Most products have smaller discounts or no discount at all.
    """)


    if st.button("Back "):
        go_to('DashBoard')
elif st.session_state.Page == "The most frequent Brand":
        # Step 1: Get top 50 brands
    top_brands = df['brand'].value_counts().nlargest(50).index

    # Step 2: Replace all brands not in top 50 with 'Other'
    df['brand'] = df['brand'].apply(lambda x: x if x in top_brands else 'Other')

    # Step 3: Count and sort values
    brand_counts = df['brand'].value_counts().sort_values(ascending=False)

    # Step 4: Remove 'Other' and 'unknown' from the counts
    brand_counts = brand_counts[~brand_counts.index.str.lower().isin(['other', 'unknown'])]

    # --- Seaborn Bar Plot ---
    plt.figure(figsize=(15, 10))
    sns.barplot(x=brand_counts.values, y=brand_counts.index, palette="cubehelix")
    plt.title("Top 50 Most Commonly Sold Brands ", fontsize=18)
    plt.xlabel("Number of Products", fontsize=14)
    plt.ylabel("Brand", fontsize=14)
    plt.tight_layout()
    st.pyplot(plt.gcf())
    st.markdown("""
    From this, we identified the **most popular brands** in the store.
    """)

    if st.button("Back"):
        go_to('DashBoard')
elif st.session_state.Page=="Percentage of the Prducts price categoties":
    # Create the pie chart in the same style
    fig3 = px.pie(
        price_category_df,
        names='Price Category',
        values='Percentage',
        title=' Product Distribution by Price Category',
        color_discrete_sequence=px.colors.sequential.Magenta_r,
        template='plotly_dark'
    )

    # Customize trace appearance
    fig3.update_traces(
        textinfo='percent+label',
        pull=[0.05] * len(price_category_df)  # add a small pull for all slices
    )
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown("""
    From this, it's clear that:
    - **Low price**: 12.2%
    - **Medium price**: 51%
    - **High price**: 36.96%
    """)
    if st.button("Back "):
            go_to('DashBoard')
elif st.session_state.Page=="Total sales per price category":
    # 3. Group by price_ratio_category and sum sales
    price_by_ratio = df.groupby('price_ratio_category')['price'].sum().reset_index()

    fig = px.bar(
        price_by_ratio, 
        x='price_ratio_category', 
        y='price', 
        title='Total Sales by Price Ratio Category',
        labels={'price': 'Total Sales (EGP)', 'price_ratio_category': 'Price Ratio Category'},
        color='price_ratio_category',
        color_discrete_sequence=px.colors.sequential.RdBu
    )
    fig.update_layout(template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    That’s because only a few products have high discounts, 
    so the sales appear to be low.
    """)

    if st.button("Back "):
            go_to('DashBoard')
elif st.session_state.Page=="The percentage of the product categories in the shop ":
    # Create the pie chart using Plotly
    fig1 = px.pie(
        category_df,
        names='Category',
        values='Percentage',
        title='Product Distribution by Category',
        color_discrete_sequence=px.colors.sequential.Magenta_r,
        template='plotly_dark'
    )
    # Customize trace appearance
    fig1.update_traces(
        textinfo='percent+label',
        pull=[0.05] * len(category_df)  # slight pull for all slices
    )
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown("""
    The pie chart clearly shows that **computing accessories** and **mobile accessories** have the largest share. 
    However, **kitchen appliances** (0.59%) is almost invisible.
    """)
    if st.button("Back "):
            go_to('DashBoard')
elif st.session_state.Page=="The total sales per producr category":
    fig = px.bar(price_by_category, 
                            x=price_by_category.index, 
                            y=price_by_category.values, 
                            labels={'x': 'category', 'y': 'sales'},
                            title="Total Sales by category")

    fig.update_layout(
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white'),  
        title_font=dict(color='white'),
        xaxis_title='category',
        yaxis_title='Sales',
    )
    st.plotly_chart(fig, use_container_width=True)
    if st.button("Back "):
            go_to('DashBoard')
elif st.session_state.Page=="Review the ratting count":
    fig = px.violin(
                        df.dropna(subset=['review_category']),  # Exclude NaN categories from plot
                        x="review_category",
                        y=np.log1p(df["review_count"]),
                        box=True,
                        points=False,
                        color="review_category",
                        color_discrete_sequence=["#2E8B57", "#4169E1"],
                        title=f"Review Count Distribution (Threshold={threshold:.0f} reviews)" if not np.isnan(threshold) else "Review Count Distribution",
                        template="plotly_dark",
                        category_orders={"review_category": ["low", "high"]}
                    )

    # Customize layout
    fig.update_layout(
        xaxis_title="Review Category (NaN values excluded)",
        yaxis_title="Log(Review Count + 1)",
        showlegend=False
    )

    # Only add threshold line if calculation succeeded
    if not np.isnan(threshold):
        fig.add_hline(
            y=np.log1p(threshold),
            line_dash="dot",
            line_color="red",
            annotation_text=f"Threshold: {threshold:.0f} reviews",
            annotation_position="top right"
        )
    st.plotly_chart(fig, use_container_width=True)
                            
    st.markdown("""
    Most products have **low reviews**, but there are a few with **extremely high reviews**.
    """)

    if st.button("Back "):
        go_to('DashBoard')
elif st.session_state.Page=="The most frequent ratting":
    fig2 = px.histogram(
                            df,
                            x='rating',
                            nbins=5,
                            title="Distribution of rating",
                            template='plotly_dark'
                        )

    fig2.update_layout(
        xaxis_title='rating',
        yaxis_title='Count',
        height=400,
        bargap=0.1  
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("""
    The distribution is **left-skewed**, meaning higher ratings (4 and 5) are more common.
    """)

    if st.button("Back "):
            go_to('DashBoard')
elif st.session_state.Page=="The total sales for the top 20 brands":
    # --- Plotly Vertical Bar Chart ---
    fig = px.bar(
        brand_price_df,
        x='Brand',
        y='Total Sales',
        text='Total Sales',
        title='Total Sales for Top 20 Brands',
        color='Total Sales',
        color_continuous_scale='magma',
        template='plotly_white'
    )

    # Format hover and text
    fig.update_traces(
        texttemplate='%{text:,.0f}',
        textposition='outside'
    )
    fig.update_layout(
        xaxis_title='Brand',
        yaxis_title='Total Sales',
        xaxis_tickangle=-45,
        uniformtext_minsize=8,
        uniformtext_mode='hide',
        height=700
    )
    st.plotly_chart(fig, use_container_width=True)
    if st.button("Back "):
        go_to('DashBoard')
elif st.session_state.Page=="Tob ten products seller":
    # Step 1: Get the top 10 best sellers sorted by engagement
    top_10 = df[df['is_best_seller']].sort_values(by='engagement_score', ascending=False).head(10)

    # Step 2: Create bar chart
    fig = px.bar(
        top_10,
        x='name',
        y='engagement_score',
        color='brand',
        title='Top 10 Best Seller Products by Engagement Score',
        labels={'name': 'Product Name', 'engagement_score': 'Engagement Score'},
        text='engagement_score'
    )

    # Step 3: Improve layout
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(xaxis_tickangle=-45, yaxis=dict(title='Engagement Score'))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(best_sellers[['name', 'brand', 'category', 'engagement_score', 'best_seller_score']].head(10))
    if st.button("Back "):
            go_to('DashBoard')
elif st.session_state.Page =="DashBoard":
    
    st.title("Dashboard Page")

    cols = st.columns([2, 2])
    with cols[0]:
        if st.button("Features Distributions",use_container_width=True):
            go_to("Features Distributions")
        if st.button("Price per Category",use_container_width=True):
            go_to("Price per Category")
        if st.button("Most_Products have low_price",use_container_width=True):
            go_to("Most_Products have low_price")
        if st.button("The percentage of the product categories in the shop ",use_container_width=True):
                go_to("The percentage of the product categories in the shop ")
        
        if st.button("The total sales per producr category",use_container_width=True):
            go_to("The total sales per producr category")
    
        if st.button("Review the ratting count",use_container_width=True):
            go_to("Review the ratting count")
    with cols[1]:
            if st.button("The most frequent Brand",use_container_width=True):
                go_to("The most frequent Brand")
            if st.button("Percentage of the Prducts price categoties",use_container_width=True):
                go_to("Percentage of the Prducts price categoties")
            if st.button("Total sales per price category",use_container_width=True):
                go_to("Total sales per price category")
            if st.button("The most frequent ratting",use_container_width=True):
                go_to("The most frequent ratting")
            if st.button("The total sales for the top 20 brands",use_container_width=True):
                go_to("The total sales for the top 20 brands")
            if st.button("Tob ten products seller",use_container_width=True):
                go_to("Tob ten products seller")
    if st.button("Back to Home"):
        go_to('Home')
# Cast Page
elif st.session_state.Page == 'Business user':

    st.title("Business user Page")

    x=23010158
    y=23010132
    z=23011530

    id=st.text_input(" What is your id ?")

    if id:  
        if int(id) == x or int(id) == y or int(id) == z:
            st.success("Welcome, Business user")
            if st.button("Go to Dashboard"):
                go_to('DashBoard')
            
        else:
            st.write("You are not of our cast !")
    else:
        st.write("Please enter your id.")  

    if st.button("Back to Home"):
        go_to('Home')
#### search page
elif st.session_state.Page == 'SearchResults':
    st.title("🔍 Search Results")
    # 🔙 Back button
    if st.button("🔙 Back to Customer"):
        st.session_state.search_input = ""
        st.session_state.Page = 'Customer'
        st.rerun()
    def navigate_to_product(product_dict):  # No underscore
        """Public navigation handler"""
        st.session_state.selected_product = product_dict
        st.session_state.Page = 'ProductDetails'
    # Get search input
    search_input = st.session_state.get('search_input', '').strip()

    # 🚫 Show warning if input is empty
    if search_input:
        # Normalize input and perform partial match
        search_term = search_input.lower()

        def normalize(text):
            return str(text).lower().strip()

        search_results = df[df['name'].apply(lambda x: search_term in normalize(x))]

        if 'clicked_products' not in st.session_state:
            st.session_state.clicked_products = {}
        # ✅ If matching products found
        # Perform the search (more robust method)
        try:
            # Convert to lowercase and check for substring match
            search_results = df[df['name'].str.lower().str.contains(search_input.lower(), na=False)]
            st.write(f"Found {len(search_results)} matches")  # Debug line
        except Exception as e:
            st.error(f"Search error: {str(e)}")
            st.stop()
        if not search_results.empty:
            st.subheader(f"Showing results for: `{search_input}`")

            for i, row in search_results.iterrows():
                with st.container():
                    st.markdown(f"""
                        <div style="background-color:#222831; padding:15px; border-radius:12px; margin-bottom:12px;
                            border: 1px solid #393E46; box-shadow: 2px 2px 8px rgba(0,0,0,0.2); color: #EEEEEE;">
                            <h4 style="margin-bottom:8px;">🧾 {row['name']}</h4>
                            <p><strong>💵 Price:</strong> {row['price']} EGP</p>
                            <p><strong>🏷️ Brand:</strong> {row['brand']} | <strong>📁 Category:</strong> {row['category']}</p>
                        </div>
                    """, unsafe_allow_html=True)
                    unique_str = f"{row['name']}{row['price']}{row['brand']}".encode()
                    btn_key = hashlib.md5(unique_str).hexdigest()
                    st.button(
                        "View Details",
                        on_click=navigate_to_product,  # Our callback
                        args=(row.to_dict(),),          # Pass product data
                        key=f"product_{i}"      # Unique key
                    )

               
        elif not search_input:
            st.warning("❗ Please enter a product name to search.")
            if st.button("🔙 Back to Customer"):
                st.session_state.Page = 'Customer'
                st.rerun()
            st.stop()
        else:
            # ❌ No results found
            st.markdown(f"""
                <div style="background-color:#ffe6e6; padding:12px; border-radius:10px; margin-top:20px;
                    border: 1px solid #ff4d4d; color: #990000; font-weight: 500; font-size: 15px;">
                    ❌ No results found for "<span style='text-decoration: underline;'>{search_input}</span>".
                    Please try searching for something else!
                </div>
            """, unsafe_allow_html=True)

    

# Customer Page
elif st.session_state.Page == 'Customer':
    col1, col2 = st.columns([4,2])
    with col1:
        st.title("Customer Page")
    with col2:
        st.write("")
        if st.button("🛍️ View Cart"):
                go_to('Cart')
    if 'search_input' not in st.session_state:
        st.session_state.search_input = ""

    def trigger_search():
        if st.session_state.search_input.strip():
            st.session_state.Page = 'SearchResults'
    st.write("You can Entre name of a Product/Category/Brand")
    st.text_input("What do you want to buy?", key="search_input", on_change=trigger_search)

 
    st.markdown("## 🔥 Top 10 Best Sellers")
    top_10 = df[df['is_best_seller']].sort_values(by='engagement_score', ascending=False).head(10)

    for i, row in top_10.iterrows():
        with st.container():
            st.markdown(f"""
                <div style="background-color:#222831; padding:15px; border-radius:12px; margin-bottom:12px;
                    border: 1px solid #393E46; box-shadow: 2px 2px 8px rgba(0,0,0,0.2); color: #EEEEEE;">
                    <h4 style="margin-bottom:8px;">🧾 {row['name']}</h4>
                    <p style="margin:4px;"><strong>💵 Price:</strong> {row['price']} EGP</p>
                </div>
            """, unsafe_allow_html=True)

            if st.button("🔍 View more details", key=f"top_{i}"):
                st.session_state.selected_product = row.to_dict()
                go_to('ProductDetails')

    if st.button("Back to Home"):
        go_to('Home')


elif st.session_state.Page == 'ProductDetails':
    product = st.session_state.get('selected_product', None)
    st.title(product['name'])
    if product:
        col1, col2 = st.columns([4,2])
        with col1:
            
            st.markdown(f"""
                <div style="background-color:#f0f2f6; padding:20px; border-radius:10px;
                            color:#000000; box-shadow: 2px 2px 10px rgba(0,0,0,0.2);">
                    <p><strong>💵 Price:</strong> {product['price']} EGP</p>
                    <p><strong>💰 Original Price:</strong> {product['original_price']} EGP </p>
                    <p><strong>🎯 Discount:</strong> {product['discount']}%</p>
                    <p><strong>⭐ Rating:</strong> {product['rating']} of 5</p>
                </div>
            """, unsafe_allow_html=True)

            recommended_df = get_enhanced_recommendations(product['name'], recommender_df, cosine_sim, top_n=4)

            if isinstance(recommended_df, pd.DataFrame) and not recommended_df.empty:
                st.markdown("### 🧠 You Might Also Like:")
                for i, rec in recommended_df.iterrows():
                    with st.container():
                        st.markdown(f"""
                            <div style="background-color:#e6e6e6; padding:10px; border-radius:8px; margin:10px 0;
                                        border: 1px solid #ccc; color: #000000;">
                                <p><strong>{rec['name']}</strong></p>
                                <p>Brand: {rec['brand']} | Category: {rec['category']}</p>
                                <p>💵 Price: {rec['price']} EGP</p>
                            </div>
                        """, unsafe_allow_html=True)
                    if st.button(f"View More Details ", key=f"view_details_{i}"):
                        st.session_state.selected_product = rec.to_dict()
                        st.session_state.Page = 'Recommendition Details'
                        st.rerun()
            else:
                st.info("ℹ️ No similar products found.")

        with  col2:
            if st.button("🛒 Add to Cart"):
                st.session_state.cart.append(product)
                st.success("Product added to cart!")

            if st.button("🛍️ View Cart"):
                go_to('Cart')

        
            if st.button("🔙 Back to Search"):
                go_to('Customer')

    else:
        st.error("❌ No product selected.")
        if st.button("🔙 Back"):
            go_to('Customer')
### Cart Page
elif st.session_state.Page == 'Cart':
    st.title("🛒 Your Cart")

    if st.session_state.cart:
        # to calculate the number of the same item
        cart_items = {}
        for item in st.session_state.cart:
            key = item['name']
            if key in cart_items:
                cart_items[key]["quantity"] += 1
            else:
                cart_items[key] = {"product": item, "quantity": 1}

        for i, (name, item_info) in enumerate(cart_items.items()):
            product = item_info["product"]
            quantity = item_info["quantity"]

            col1, col2 = st.columns([5, 1])
            with col1:
                # to make the backround dark
                st.markdown(f"""
                    <div style="background-color:#222831; padding:12px; border-radius:10px;
                        border: 1px solid #393E46; margin-bottom: 10px; color: #FFFFFF;">
                        <h4>{product['name']}</h4>
                        <p><strong>Price:</strong> {product['price']} EGP</p>
                        <p><strong>Quantity:</strong> {quantity}</p>
                    </div>
                """, unsafe_allow_html=True)

            with col2:
                # to decrease the number of the quantities
                if st.button("➖", key=f"remove_one_{i}"):
                    if quantity > 1:
                        # to decrese
                        st.session_state.cart.remove(product)
                        st.rerun()  
                    elif quantity == 1:
                        # delete the whole item
                        st.session_state.cart = [item for item in st.session_state.cart if item['name'] != name]
                        st.rerun()  

                # to increase the number of quantities
                if st.button("➕", key=f"add_one_{i}"):
                    st.session_state.cart.append(product)
                    st.rerun()  
                    # ====== Order Summary ======
        st.markdown("---")
        st.subheader("🧾 Order Summary")

        total_price = 0
        for name, item_info in cart_items.items():
            product = item_info["product"]
            quantity = item_info["quantity"]
            item_total = product['price'] * quantity
            total_price += item_total

            st.markdown(f"""
                <div style="background-color:#393E46; padding:10px; border-radius:8px; color:#EEEEEE; margin-bottom:5px;">
                    <strong>{name}</strong><br>
                    Quantity: {quantity}<br>
                    Price per item: {product['price']} EGP<br>
                    <strong>Total: {item_total} EGP</strong>
                </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
            <div style="background-color:#00ADB5; padding:15px; border-radius:10px; color:#FFFFFF; font-size:18px; text-align:center;">
                <strong>Total: {total_price} EGP</strong>
            </div>
        """, unsafe_allow_html=True)
        if st.button("✅ Place Order"):
            st.session_state.cart = []  # تفريغ السلة
            st.success("✅ Order placed successfully!")

    else:
        st.info("Your cart is empty.")

    if st.button("Back to Home"):
        go_to('Home')
elif st.session_state.Page =="Recommendition Details":
    product = st.session_state.get('selected_product', None)

    if product:
        
        st.title(product['name'])
        st.markdown(f"""
            <div style="background-color:#f0f2f6; padding:20px; border-radius:10px;
                        color:#000000; box-shadow: 2px 2px 10px rgba(0,0,0,0.2);">
                <p><strong>💵 Price:</strong> {product['price']} EGP</p>
                <p><strong>💰 Original Price:</strong> {product['original_price']} EGP </p>
                <p><strong>🎯 Discount:</strong> {product['discount']}%</p>
                <p><strong>⭐ Rating:</strong> {product['rating']} of 5</p>
            </div>
        """, unsafe_allow_html=True)

        if st.button("🛒 Add to Cart"):
            st.session_state.cart.append(product)
            st.success("Product added to cart!")

        if st.button("🛍️ View Cart"):
            go_to('Cart')

        if st.button("🔙 Back to Search"):
            go_to('Customer')

    else:
        st.error("❌ No product selected.")
        if st.button("🔙 Back"):
            go_to('Customer')
