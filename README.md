# AI-Driven Customer Analytics

This project provides a Streamlit-based customer analytics platform for exploring customer data, filtering by name/salary/age, making product recommendations, predicting monthly transactions, performing segmentation, income analysis, and churn detection.

## Files

- `app.py`: Streamlit app.
- `data_generator.py`: Generates synthetic datasets for customers, products, and transactions.
- `assets/style.css`: CSS styling injected into the Streamlit app.
- `data/customers.csv`: Generated customer dataset.
- `data/products.csv`: Generated product dataset.
- `data/transactions.csv`: Generated transaction dataset.

## Run

1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Generate datasets:
```bash
python data_generator.py
```
3. Launch the app:
```bash
streamlit run app.py
```
