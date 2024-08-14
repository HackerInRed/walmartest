### README

# Walmart Product Placement and Inventory Optimization

## Overview

This Streamlit application is designed to optimize product placement and inventory management in Walmart stores by leveraging customer behavior data and machine learning models. The platform helps store managers make data-driven decisions to enhance sales, minimize costs, and improve customer satisfaction.

## Features

- **Data Generation:** Simulates product-related data for analysis.
- **Data Preprocessing:** Transforms and preprocesses the data to prepare it for model training and analysis.
- **Customer Segmentation:** Clusters customers based on purchasing behavior.
- **Model Training:** Trains machine learning models to predict high sales potential and high stock levels.
- **AI-Generated Insights:** Provides a summary of key insights derived from the data.
- **Interactive Visualization:** Allows users to visualize product placement and sales data.
- **Sales and Inventory Prediction:** Predicts the probability of high sales and stock levels based on input parameters.

## Usage

1. **Generate Data:**
   - The app simulates data for products, including categories, prices, stock levels, sales, and customer ratings.

2. **Preprocess Data:**
   - The data is prepared for analysis by creating features and converting categorical variables.

3. **Customer Segmentation:**
   - The app clusters customers into segments based on their purchasing behavior using K-Means clustering.

4. **Train Models:**
   - The app trains Random Forest models to predict the likelihood of high sales and high stock levels.

5. **AI-Generated Insights:**
   - The app generates insights from the data, highlighting top-selling categories, key factors influencing sales, and inventory recommendations.

6. **Interactive Visualizations:**
   - Users can visualize product placement and sales data, filter by category and price, and explore customer segments.

7. **Predict Sales and Inventory:**
   - Input product details (price, customer rating, days since restock) to predict the likelihood of high sales and stock levels.

## Code Explanation

### 1. **Data Generation**

```python
@st.cache_data
def generate_data(n_samples=1000):
    np.random.seed(42)
    data = pd.DataFrame({
        'product_id': range(1, n_samples + 1),
        'category': np.random.choice(['Electronics', 'Clothing', 'Groceries', 'Home & Garden'], n_samples),
        'price': np.random.uniform(10, 1000, n_samples),
        'stock_level': np.random.randint(0, 100, n_samples),
        'sales_last_month': np.random.randint(0, 50, n_samples),
        'customer_rating': np.random.uniform(1, 5, n_samples),
        'shelf_position': np.random.choice(['Top', 'Middle', 'Bottom'], n_samples),
        'days_since_last_restock': np.random.randint(0, 30, n_samples)
    })
    return data
```

- Generates a synthetic dataset mimicking real-world retail data.

### 2. **Data Preprocessing**

```python
def preprocess_data(data):
    data['high_sales'] = (data['sales_last_month'] > data['sales_last_month'].median()).astype(int)
    data['high_stock'] = (data['stock_level'] > data['stock_level'].median()).astype(int)
    data = pd.get_dummies(data, columns=['category', 'shelf_position'])
    return data
```

- Creates new features for high sales and stock levels and applies one-hot encoding to categorical variables.

### 3. **Customer Segmentation**

```python
def customer_segmentation(data):
    features = ['price', 'sales_last_month', 'customer_rating']
    X = data[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=3, random_state=42)
    data['segment'] = kmeans.fit_predict(X_scaled)
    return data
```

- Segments customers into clusters using K-Means based on purchasing behavior.

### 4. **Model Training**

```python
def train_models(data):
    features = [col for col in data.columns if col not in ['product_id', 'high_sales', 'high_stock', 'sales_last_month', 'stock_level', 'segment']]
    
    X = data[features]
    y_sales = data['high_sales']
    y_stock = data['high_stock']
    
    sales_model = RandomForestClassifier(n_estimators=100, random_state=42)
    sales_model.fit(X, y_sales)
    
    stock_model = RandomForestClassifier(n_estimators=100, random_state=42)
    stock_model.fit(X, y_stock)
    
    return sales_model, stock_model
```

- Trains Random Forest models to predict high sales and stock levels based on the preprocessed data.

### 5. **AI-Generated Insights**

```python
def get_llm_summary(data, sales_model, stock_model):
    category_sales = data.groupby('category')['sales_last_month'].mean().to_dict()
    top_selling_category = max(category_sales, key=category_sales.get)
    avg_stock_level = data['stock_level'].mean()
    
    features = [col for col in data.columns if col not in ['product_id', 'high_sales', 'high_stock', 'sales_last_month', 'stock_level', 'segment']]
    sales_importance = dict(zip(features, sales_model.feature_importances_))
    stock_importance = dict(zip(features, stock_model.feature_importances_))
    
    prompt = f"""
    Analyze the following retail data and provide a concise summary of insights:

    1. Top selling category: {top_selling_category}
    2. Average stock level: {avg_stock_level:.2f}
    3. Category sales averages: {category_sales}
    4. Top factors influencing sales: {dict(sorted(sales_importance.items(), key=lambda x: x[1], reverse=True)[:3])}
    5. Top factors influencing stock levels: {dict(sorted(stock_importance.items(), key=lambda x: x[1], reverse=True)[:3])}

    Provide a summary of key insights and recommendations based on this data.
    """

    try:
        response = requests.post(LLM_API_ENDPOINT, json={"prompt": prompt})
        if response.status_code == 200:
            return response.json()['generated_text']
        else:
            return "LLM API failed. Provide default insights here."
    except Exception as e:
        return "LLM API connection error. Provide default insights here."
```

- Generates insights based on model predictions and provides a summary of key data-driven insights.

### 6. **Visualization and Prediction**

- **Interactive Visualizations:** Uses Plotly to create heatmaps, scatter plots, and histograms to visualize sales data by category, shelf position, and customer segment.
- **Sales and Inventory Prediction:** Takes user inputs and predicts the likelihood of high sales and stock levels using trained models.

## Issues Solved

- **Efficient Shelf Space Usage:** Optimizes product placement to align with customer shopping behavior.
- **Inventory Balancing:** Predicts optimal inventory levels to prevent stockouts or overstocking.
- **Adaptation to Customer Preferences:** Uses real-time data to adapt to changing customer preferences and enhance sales.

## Expected Impact

- **Increased Sales:** Strategic product placement leads to higher sales.
- **Reduced Costs:** Optimized inventory levels reduce holding costs.
- **Improved Customer Satisfaction:** Better product availability enhances the shopping experience.
- **Operational Efficiency:** Provides store managers with actionable insights for better decision-making.