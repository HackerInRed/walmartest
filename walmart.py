import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
import plotly.express as px
import requests
import json

LLM_API_ENDPOINT = "http://localhost:4000/api/llm" 

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

def preprocess_data(data):
    data['high_sales'] = (data['sales_last_month'] > data['sales_last_month'].median()).astype(int)
    data['high_stock'] = (data['stock_level'] > data['stock_level'].median()).astype(int)
    data = pd.get_dummies(data, columns=['category', 'shelf_position'])
    return data

def customer_segmentation(data):
    features = ['price', 'sales_last_month', 'customer_rating']
    X = data[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=3, random_state=42)
    data['segment'] = kmeans.fit_predict(X_scaled)
    return data

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

def generate_fake_llm_summary(data, sales_model, stock_model):
    category_sales = data.groupby('category')['sales_last_month'].mean().to_dict()
    top_selling_category = max(category_sales, key=category_sales.get)
    avg_stock_level = data['stock_level'].mean()
    
    features = [col for col in data.columns if col not in ['product_id', 'high_sales', 'high_stock', 'sales_last_month', 'stock_level', 'segment']]
    sales_importance = dict(zip(features, sales_model.feature_importances_))
    stock_importance = dict(zip(features, stock_model.feature_importances_))
    
    summary = f"""
    **Top selling category:** {top_selling_category}
    **Average stock level:** {avg_stock_level:.2f}
    **Category sales averages:** {category_sales}
    **Top factors influencing sales:** {dict(sorted(sales_importance.items(), key=lambda x: x[1], reverse=True)[:3])}
    **Top factors influencing stock levels:** {dict(sorted(stock_importance.items(), key=lambda x: x[1], reverse=True)[:3])}
    """
    return summary

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
            return generate_fake_llm_summary(data, sales_model, stock_model)
    except Exception as e:
        return generate_fake_llm_summary(data, sales_model, stock_model) # this generates the fake LLM summary

def main():
    st.title("Walmart Product Placement and Inventory Optimization")

    data = generate_data()
    data = customer_segmentation(data)
    processed_data = preprocess_data(data)
    sales_model, stock_model = train_models(processed_data)

    summary = get_llm_summary(data, sales_model, stock_model)
    st.header("AI-Generated Insights Summary")
    st.write(summary)

    st.sidebar.header("Filters")
    category = st.sidebar.multiselect("Select Category", data['category'].unique(), default=data['category'].unique())
    price_range = st.sidebar.slider("Price Range", float(data['price'].min()), float(data['price'].max()), (float(data['price'].min()), float(data['price'].max())))

    filtered_data = data[data['category'].isin(category) & (data['price'] >= price_range[0]) & (data['price'] <= price_range[1])]

    st.header("Product Placement Analysis")

    placement_data = filtered_data.groupby(['category', 'shelf_position'])['sales_last_month'].mean().unstack()
    fig_heatmap = px.imshow(placement_data, text_auto=True, aspect="auto", title="Average Sales by Category and Shelf Position")
    fig_heatmap.update_layout(height=500)
    st.plotly_chart(fig_heatmap)

    fig_scatter = px.scatter(filtered_data, x='price', y='sales_last_month', color='segment', 
                             hover_data=['product_id', 'category', 'customer_rating'],
                             title="Customer Segmentation",
                             labels={'segment': 'Customer Segment'})
    fig_scatter.update_layout(height=500)
    st.plotly_chart(fig_scatter)

    st.header("Inventory and Sales Prediction")

    col1, col2, col3, col4 = st.columns(4)
    price = col1.number_input("Price", min_value=10.0, max_value=1000.0, value=100.0)
    rating = col2.number_input("Customer Rating", min_value=1.0, max_value=5.0, value=4.0)
    days_since_restock = col3.number_input("Days Since Last Restock", min_value=0, max_value=30, value=7)
    category = col4.selectbox("Category", data['category'].unique())

    if st.button("Predict"):
        input_data = pd.DataFrame({
            'price': [price],
            'customer_rating': [rating],
            'days_since_restock': [days_since_restock],
            'category_Clothing': [1 if category == 'Clothing' else 0],
            'category_Electronics': [1 if category == 'Electronics' else 0],
            'category_Groceries': [1 if category == 'Groceries' else 0],
            'category_Home & Garden': [1 if category == 'Home & Garden' else 0],
            'shelf_position_Bottom': [0],
            'shelf_position_Middle': [0],
            'shelf_position_Top': [0]
        })

        sales_pred = sales_model.predict_proba(input_data)[0][1]
        stock_pred = stock_model.predict_proba(input_data)[0][1]

        st.write(f"Probability of High Sales: {sales_pred:.2f}")
        st.write(f"Probability of High Stock Level: {stock_pred:.2f}")

    fig_hist = px.histogram(filtered_data, x='stock_level', color='category', 
                            title="Stock Level Distribution by Category",
                            marginal="box")
    fig_hist.update_layout(height=500)
    st.plotly_chart(fig_hist)

    fig_sales_price = px.scatter(filtered_data, x='price', y='sales_last_month', color='category',
                                 size='stock_level', hover_data=['product_id', 'customer_rating'],
                                 title="Sales vs. Price by Category")
    fig_sales_price.update_layout(height=500)
    st.plotly_chart(fig_sales_price)

if __name__ == "__main__":
    main()