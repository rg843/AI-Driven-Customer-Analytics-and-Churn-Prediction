import os
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import warnings

# Compatibility wrapper for Streamlit cache decorator across versions
try:
    cache_data = st.cache_data
except Exception:
    def cache_data(func=None, **kwargs):
        if func is None:
            return lambda f: f
        return func

# Safe rerun shim for Streamlit versions without `experimental_rerun`
def safe_rerun():
    try:
        st.experimental_rerun()
    except Exception:
        try:
            st.experimental_set_query_params(_rerun='1')
        except Exception:
            pass
        st.info('Please refresh the page to see the latest updates.')

DATA_DIR = 'data'
CUSTOMER_PATH = os.path.join(DATA_DIR, 'customers.csv')
PRODUCT_PATH = os.path.join(DATA_DIR, 'products.csv')
TRANSACTION_PATH = os.path.join(DATA_DIR, 'transactions.csv')

st.set_page_config(page_title='AI-Driven Customer Analytics', layout='wide', page_icon='assets/logo.svg')

# Inject shared CSS (unchanged shared file)
with open('assets/style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Scoped local overrides (do not modify shared stylesheet)
st.markdown("<script>document.body.setAttribute('data-adc-theme','true')</script>", unsafe_allow_html=True)
with open('assets/local_override.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# App header (logo + title) to differentiate the UI
header_html = '''
<div class="app-header">
    <img src="assets/logo.svg" width="140"/>
    <div>
        <div class="app-title">ADC CX Studio</div>
        <div class="app-sub">Fresh customer insights • actionable recommendations</div>
    </div>
</div>
'''
st.markdown(header_html, unsafe_allow_html=True)

# Reduce noisy warnings in the UI/logs
warnings.filterwarnings('ignore')

st.title('AI-Driven Customer Analytics')
st.write('Explore customer segments, predict transactions, recommend products, and analyze churn using ML-driven insights.')

@cache_data
def load_data():
    customers = pd.read_csv(CUSTOMER_PATH)
    products = pd.read_csv(PRODUCT_PATH)
    transactions = pd.read_csv(TRANSACTION_PATH)
    merged = customers.merge(transactions, on='customer_id', how='left')
    return customers, products, transactions, merged

@cache_data
def train_models(data):
    features = data[['age', 'salary', 'monthly_expenditures']]
    labels = data['monthly_transactions']
    churn = data['churn']

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score

    X_train, X_test, y_train, y_test, y_churn_train, y_churn_test = train_test_split(
        features, labels, churn, test_size=0.25, random_state=42, stratify=churn
    )

    reg = RandomForestRegressor(n_estimators=100, random_state=42)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    reg_r2 = r2_score(y_test, y_pred)
    reg_mae = mean_absolute_error(y_test, y_pred)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_churn_train)
    churn_pred = clf.predict(X_test)
    churn_acc = accuracy_score(y_churn_test, churn_pred)

    return reg, clf, {
        'reg_r2': reg_r2,
        'reg_mae': reg_mae,
        'churn_accuracy': churn_acc,
    }

@cache_data
def build_segmentation(data, clusters=4):
    scaler = StandardScaler()
    X = scaler.fit_transform(data[['age', 'salary', 'monthly_expenditures']])
    kmeans = KMeans(n_clusters=clusters, random_state=42, n_init=10)
    data['segment'] = kmeans.fit_predict(X)
    return data, kmeans

@cache_data
def load_assets():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    return

load_assets()
customers, products, transactions, merged = load_data()
reg_model, churn_model, model_scores = train_models(merged)
segmented, kmeans = build_segmentation(merged)

if 'customer_df' not in st.session_state:
    st.session_state.customer_df = customers.copy()
if 'product_df' not in st.session_state:
    st.session_state.product_df = products.copy()
if 'transaction_df' not in st.session_state:
    st.session_state.transaction_df = transactions.copy()

# Sidebar filters and controls (grouped)
with st.sidebar:
    st.header('Controls')
    with st.expander('Filters', expanded=True):
        search_mode = st.radio('Search customer by', ['All', 'Name', 'Age', 'Salary'])
        search_name = ''
        exact_age = None
        exact_salary = None
        if search_mode == 'Name':
            search_name = st.text_input('Customer name')
        elif search_mode == 'Age':
            exact_age = st.number_input('Exact age', min_value=18, max_value=65, value=30, step=1)
        elif search_mode == 'Salary':
            exact_salary = st.number_input('Exact salary', min_value=22000, max_value=140000, value=60000, step=1000)

        age_min, age_max = st.slider('Age range', 18, 65, (18, 65), step=1)
        salary_min, salary_max = st.slider('Salary range', 22000, 140000, (22000, 140000), step=1000)
        selected_segment = st.selectbox('Segment filter', ['All'] + sorted(segmented['segment'].unique().tolist()))

    with st.expander('Actions', expanded=False):
        if st.button('Save datasets'):
            st.session_state.customer_df.to_csv(CUSTOMER_PATH, index=False)
            st.session_state.product_df.to_csv(PRODUCT_PATH, index=False)
            st.session_state.transaction_df.to_csv(TRANSACTION_PATH, index=False)
            st.success('Saved datasets to CSV files.')

        if st.button('Reset view'):
            # Reset session datasets to original loaded copies
            st.session_state.customer_df = customers.copy()
            st.session_state.product_df = products.copy()
            st.session_state.transaction_df = transactions.copy()
            st.experimental_rerun()

# Filtered dataset
filtered = st.session_state.customer_df.copy()
if search_mode == 'Name' and search_name:
    filtered = filtered[filtered['name'].str.contains(search_name, case=False, na=False)]
elif search_mode == 'Age' and exact_age is not None:
    filtered = filtered[filtered['age'] == exact_age]
elif search_mode == 'Salary' and exact_salary is not None:
    filtered = filtered[filtered['salary'] == exact_salary]

filtered = filtered[(filtered['age'] >= age_min) & (filtered['age'] <= age_max)]
filtered = filtered[(filtered['salary'] >= salary_min) & (filtered['salary'] <= salary_max)]
if selected_segment != 'All':
    filtered = filtered.merge(segmented[['customer_id', 'segment']], on='customer_id', how='left')
    filtered = filtered[filtered['segment'] == selected_segment]

# Top-level metrics (rearranged to be above the tabs)
with st.container():
    mcol1, mcol2, mcol3 = st.columns([1,1,1])
    mcol1.metric('Total customers', len(filtered))
    mcol2.metric('Average salary', f"₹{int(filtered['salary'].mean()):,}" if not filtered.empty else 'N/A')
    mcol3.metric('Average expenditure', f"₹{int(filtered['monthly_expenditures'].mean()):,}" if not filtered.empty else 'N/A')

# Place a filtered CSV download into the sidebar for easy access
try:
    csv = filtered.reset_index(drop=True).to_csv(index=False).encode('utf-8')
    st.sidebar.download_button('Download filtered CSV', data=csv, file_name='filtered_customers.csv', mime='text/csv')
except Exception:
    pass

# Layout tabs
tabs = st.tabs(['Explorer', 'Profile & CRUD', 'Add Customer', 'Analytics', 'Prediction', 'Income Analysis', 'Product Catalog'])

with tabs[0]:
    st.subheader('Customer Data Explorer')
    col1, col2 = st.columns([2, 1])
    with col1:
        st.write('Filtered customers based on the current search and ranges.')
        st.dataframe(filtered.reset_index(drop=True))
    with col2:
        st.metric('Total customers', len(filtered))
        st.metric('Average salary', f"₹{int(filtered['salary'].mean()):,}" if not filtered.empty else 'N/A')
        st.metric('Average expenditure', f"₹{int(filtered['monthly_expenditures'].mean()):,}" if not filtered.empty else 'N/A')

with tabs[1]:
    st.subheader('Customer Profile & CRUD')
    selection_mode = st.radio('Select details by', ['Customer ID', 'Age', 'Salary'])
    selected_customer_id = None
    selected_age = None
    selected_salary = None
    details = pd.DataFrame()

    if selection_mode == 'Customer ID':
        customer_options = ['None'] + filtered['customer_id'].astype(str).tolist()
        selected_customer_id = st.selectbox('Select customer ID', customer_options)
        if selected_customer_id != 'None':
            selected_customer_id = int(selected_customer_id)
            details = st.session_state.customer_df[st.session_state.customer_df['customer_id'] == selected_customer_id]
    elif selection_mode == 'Age':
        age_options = sorted(filtered['age'].unique().tolist())
        if age_options:
            selected_age = st.selectbox('Select age', age_options)
            details = filtered[filtered['age'] == selected_age]
    else:
        salary_options = sorted(filtered['salary'].unique().tolist())
        if salary_options:
            selected_salary = st.selectbox('Select salary', salary_options)
            details = filtered[filtered['salary'] == selected_salary]

    if details.empty:
        st.info('Choose a selection method and value to display customer details.')
    else:
        st.write('### Selected customer details')
        st.dataframe(details.reset_index(drop=True))
        if selection_mode == 'Customer ID' and len(details) == 1:
            row = details.iloc[0]
            st.write(f'#### Profile: {row.name}')
            st.write(f'- Age: {row.age}')
            st.write(f'- Salary: ₹{row.salary:,}')
            st.write(f'- Monthly expenditures: ₹{row.monthly_expenditures:,}')
            st.write('#### Product recommendations')
            recomm = st.session_state.product_df.copy()
            recomm = recomm[recomm['recommended_min_salary'] <= row.salary]
            recomm['affordability'] = recomm['price'] / max(1, row.salary / 12)
            recomm = recomm.sort_values(['affordability', 'price']).head(8)
            st.table(recomm[['product_name', 'category', 'price', 'recommended_min_salary']])

            st.write('#### Prediction & churn')
            pred_trans = reg_model.predict(np.array([[row.age, row.salary, row.monthly_expenditures]]))[0]
            pred_churn = churn_model.predict(np.array([[row.age, row.salary, row.monthly_expenditures]]))[0]
            st.metric('Predicted monthly transactions', f'{pred_trans:.1f}')
            st.metric('Churn risk', 'High' if pred_churn else 'Low')

            st.write('---')
            with st.expander('Edit customer details'):
                new_name = st.text_input('Name', row.name)
                new_age = st.number_input('Age', min_value=18, max_value=90, value=int(row.age))
                new_salary = st.number_input('Salary', min_value=20000, max_value=200000, value=int(row.salary), step=1000)
                new_expenditures = st.number_input('Monthly expenditures', min_value=500, max_value=200000, value=int(row.monthly_expenditures), step=100)
                if st.button('Update customer'):
                    idx = st.session_state.customer_df.index[st.session_state.customer_df['customer_id'] == selected_customer_id][0]
                    st.session_state.customer_df.at[idx, 'name'] = new_name
                    st.session_state.customer_df.at[idx, 'age'] = new_age
                    st.session_state.customer_df.at[idx, 'salary'] = new_salary
                    st.session_state.customer_df.at[idx, 'monthly_expenditures'] = new_expenditures
                    st.success('Customer updated successfully.')
                    safe_rerun()

            if st.button('Delete customer'):
                st.session_state.customer_df = st.session_state.customer_df[st.session_state.customer_df['customer_id'] != selected_customer_id]
                st.session_state.transaction_df = st.session_state.transaction_df[st.session_state.transaction_df['customer_id'] != selected_customer_id]
                st.success('Customer deleted successfully.')
                safe_rerun()
        else:
            st.write('#### Selected group summary')
            st.metric('Customers found', len(details))
            st.metric('Average salary', f"₹{int(details['salary'].mean()):,}")
            st.metric('Average expenditures', f"₹{int(details['monthly_expenditures'].mean()):,}")
            st.write('Use the filtered table above to inspect matching customers by age or salary.')

with tabs[2]:
    st.subheader('Add a new customer')
    with st.form('add_customer_form'):
        new_name = st.text_input('Customer name')
        new_age = st.number_input('Age', min_value=18, max_value=90, value=30)
        new_salary = st.number_input('Salary', min_value=22000, max_value=200000, value=60000, step=1000)
        new_expenditures = st.number_input('Monthly expenditures', min_value=500, max_value=200000, value=3500, step=100)
        submitted = st.form_submit_button('Create customer')
        if submitted:
            new_id = int(st.session_state.customer_df['customer_id'].max() + 1)
            st.session_state.customer_df = pd.concat([
                st.session_state.customer_df,
                pd.DataFrame([{
                    'customer_id': new_id,
                    'name': new_name,
                    'age': new_age,
                    'salary': new_salary,
                    'monthly_expenditures': new_expenditures
                }])
            ], ignore_index=True)
            st.session_state.transaction_df = pd.concat([
                st.session_state.transaction_df,
                pd.DataFrame([{
                    'customer_id': new_id,
                    'monthly_transactions': int(np.clip(np.random.poisson(10) + new_expenditures // 1500, 1, 40)),
                    'churn': int((new_expenditures > new_salary * 0.9) or (new_expenditures / new_salary > 0.75)),
                    'active_months': int(np.clip(np.random.normal(10, 2), 3, 12))
                }])
            ], ignore_index=True)
            st.success('New customer added.')
            safe_rerun()

with tabs[3]:
    st.subheader('Analytics & Visualizations')
    col1, col2 = st.columns(2)
    with col1:
        st.write('### Salary vs Expenditure')
        fig1 = px.scatter(st.session_state.customer_df, x='salary', y='monthly_expenditures', color='age', hover_name='name', trendline='ols', labels={'salary':'Salary', 'monthly_expenditures':'Monthly Expenditure'})
        st.plotly_chart(fig1, width='stretch')
        st.write('### Customer segments')
        fig2 = px.scatter(segmented, x='salary', y='monthly_expenditures', color='segment', size='age', hover_name='name', labels={'segment':'Segment'})
        st.plotly_chart(fig2, width='stretch')
    with col2:
        st.write('### Salary distribution')
        fig3 = px.histogram(st.session_state.customer_df, x='salary', nbins=25, title='Salary distribution', labels={'salary':'Salary'})
        st.plotly_chart(fig3, width='stretch')
        st.write('### Churn distribution')
        churn_table = merged.groupby('churn').size().reset_index(name='count')
        fig4 = px.pie(churn_table, names='churn', values='count', title='Churn vs Active Customers')
        st.plotly_chart(fig4, width='stretch')

with tabs[4]:
    st.subheader('Prediction Analysis & Segmentation Summary')
    col1, col2 = st.columns(2)
    with col1:
        st.write('#### Transaction prediction model')
        prediction_set = merged[merged['customer_id'].isin(filtered['customer_id'])] if not filtered.empty else merged.copy()
        if not prediction_set.empty:
            prediction_set = prediction_set.copy()
            prediction_set['predicted_transactions'] = reg_model.predict(prediction_set[['age', 'salary', 'monthly_expenditures']])
            prediction_set['prediction_error'] = (prediction_set['predicted_transactions'] - prediction_set['monthly_transactions']).abs()
            st.metric('Model R²', f'{model_scores["reg_r2"]:.2f}')
            st.metric('Model MAE', f'{model_scores["reg_mae"]:.1f}')
            st.write('#### Sample prediction quality for selected customers')
            st.dataframe(prediction_set[['customer_id', 'age', 'salary', 'monthly_expenditures', 'monthly_transactions', 'predicted_transactions', 'prediction_error']].head(10))
        else:
            st.info('No customers are selected for prediction. Adjust filters or choose another search mode.')

    with col2:
        st.write('#### Churn prediction summary')
        st.metric('Churn model accuracy', f'{model_scores["churn_accuracy"]:.1%}')
        summary = merged.groupby('churn').agg({'customer_id':'count', 'monthly_expenditures':'mean', 'salary':'mean'}).reset_index()
        summary.columns = ['churn', 'count', 'avg_expenditure', 'avg_salary']
        st.dataframe(summary)

with tabs[5]:
    st.subheader('Income Analysis')
    analysis_set = merged[merged['customer_id'].isin(filtered['customer_id'])] if not filtered.empty else merged.copy()
    label = 'selected filters' if not filtered.empty else 'full dataset'
    st.write(f'#### Income metrics for the {label}')
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric('Average salary', f'₹{int(analysis_set.salary.mean()):,}')
    with col2:
        st.metric('Median expenditure', f'₹{int(analysis_set.monthly_expenditures.median()):,}')
    with col3:
        st.metric('Average transactions', f'{analysis_set.monthly_transactions.mean():.1f}')
    with col4:
        st.metric('Avg expenditure ratio', f'{(analysis_set.monthly_expenditures / analysis_set.salary).mean() * 100:.1f}%')

    st.write('### Salary band spend analysis')
    bands = analysis_set.copy()
    bands['salary_band'] = pd.cut(bands['salary'], bins=[20000, 40000, 60000, 80000, 100000, 140000], labels=['20k-40k','40k-60k','60k-80k','80k-100k','100k+'])
    band_summary = bands.groupby('salary_band').agg(avg_salary=('salary','mean'), avg_expenditure=('monthly_expenditures','mean'), count=('customer_id','count')).reset_index()
    fig5 = px.bar(band_summary, x='salary_band', y=['avg_salary', 'avg_expenditure'], barmode='group', title='Average salary and expenditure by salary band')
    st.plotly_chart(fig5, width='stretch')

with tabs[6]:
    st.subheader('Product recommendation dataset')
    st.dataframe(st.session_state.product_df)
    st.write('Use the filters above to locate customers, then view product recommendations and predicted transaction behavior for each profile.')

# Footer
footer_html = '''
<div class="app-footer">
    <div>© 2026 AI-Driven CX Studio — <a href="mailto:hello@example.com">Contact</a> • Built with Streamlit</div>
</div>
'''
st.markdown(footer_html, unsafe_allow_html=True)
