import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import numpy as np
import seaborn as sns

st.set_page_config(page_title="Superstore Visualization", page_icon=":chart_with_upwards_trend:",layout="wide")
st.title(" :chart_with_upwards_trend:  SuperStore Visualization")
st.markdown('<style>div.block-container{padding-top:1rem;}</style>',unsafe_allow_html=True)

uploaded_file=st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    df=pd.read_csv(uploaded_file)
    with st.expander("Data Preview"):
        st.dataframe(df)
else: 
    df = pd.read_csv('./Superstore.csv', encoding = "utf-8")

# Replace null values in the 'Postal Code' column with random values
df['Postal Code'].fillna(np.random.randint(10000, 99999), inplace=True)

# Check for null values in the DataFrame
null_values = df.isnull().sum()

# Display null values row-wise
null_values_row_wise = df[df.isnull().any(axis=1)]
st.subheader("Rows with Null Values:")
st.write(null_values_row_wise)

col1, col2 = st.columns((2))
df["Order Date"] = pd.to_datetime(df["Order Date"])

# grouped barplots
fig_grouped_bar = px.bar(df, x='Category', y='Sales', color='Region', barmode='group',
                          title='Grouped Barplot: Sales by Category and Region')
st.plotly_chart(fig_grouped_bar, use_container_width=True)

st.sidebar.header("Choose your filter: ")
# Create for Region
region = st.sidebar.multiselect("Pick your Region", df["Region"].unique())
if not region:
    df2 = df.copy()
else:
    df2 = df[df["Region"].isin(region)]

# Create for State
state = st.sidebar.multiselect("Pick the State", df2["State"].unique())
if not state:
    df3 = df2.copy()
else:
    df3 = df2[df2["State"].isin(state)]

# Create for City
city = st.sidebar.multiselect("Pick the City",df3["City"].unique())

# Filter the data based on Region, State and City
if not region and not state and not city:
    filtered_df = df
elif not state and not city:
    filtered_df = df[df["Region"].isin(region)]
elif not region and not city:
    filtered_df = df[df["State"].isin(state)]
elif state and city:
    filtered_df = df3[df["State"].isin(state) & df3["City"].isin(city)]
elif region and city:
    filtered_df = df3[df["Region"].isin(region) & df3["City"].isin(city)]
elif region and state:
    filtered_df = df3[df["Region"].isin(region) & df3["State"].isin(state)]
elif city:
    filtered_df = df3[df3["City"].isin(city)]
else:
    filtered_df = df3[df3["Region"].isin(region) & df3["State"].isin(state) & df3["City"].isin(city)]

category_df = filtered_df.groupby(by = ["Category"], as_index = False)["Sales"].sum()

# Custom colors for the bars
custom_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

with col1:
    st.subheader("Category wise Sales")
    fig = px.bar(category_df, x = "Category", y = "Sales", text = ['${:,.2f}'.format(x) for x in category_df["Sales"]],
                 template = "seaborn",
                 color=category_df["Category"],  # Set custom colors
        color_discrete_map={category: color for category, color in zip(category_df["Category"], custom_colors)}
        )
    st.plotly_chart(fig,use_container_width=True, height = 200)

with col2:
    st.subheader("Region wise Sales")
    fig = px.pie(filtered_df, values = "Sales", names = "Region", hole = 0.5)
    fig.update_traces(text = filtered_df["Region"], textposition = "outside")
    st.plotly_chart(fig,use_container_width=True)

cl1, cl2 = st.columns((2))
with cl1:
    with st.expander("Category_ViewData"):
        st.write(category_df.style.background_gradient(cmap="Oranges"))
        csv = category_df.to_csv(index = False).encode('utf-8')

with cl2:
    with st.expander("Region_ViewData"):
        region = filtered_df.groupby(by = "Region", as_index = False)[["Sales", "Quantity","Profit","Shipping Cost","Discount"]].sum()
        st.write(region.style.background_gradient(cmap="Oranges"))
        csv = region.to_csv(index = False).encode('utf-8')
        
        
filtered_df["month_year"] = filtered_df["Order Date"].dt.to_period("M")
st.subheader('TimeLine Analysis')

linechart = pd.DataFrame(filtered_df.groupby(filtered_df["month_year"].dt.strftime("%Y : %b"))[["Sales", "Profit","Shipping Cost"]].sum()).reset_index()
fig2 = px.line(linechart, x="month_year", y=["Sales", "Profit","Shipping Cost"], labels={"value": "Amount"}, height=500, width=1000, template="gridon")
st.plotly_chart(fig2, use_container_width=True)


with st.expander("View Data of TimeSeries:"):
    st.write(linechart.T.style.background_gradient(cmap="Blues"))
    csv = linechart.to_csv(index=False).encode("utf-8")
    st.download_button('Download Data', data = csv, file_name = "TimeSeries.csv", mime ='text/csv')


chart1, chart2 = st.columns((2))
with chart1:
    st.subheader('Segment wise Sales')
    fig = px.pie(filtered_df, values = "Sales", names = "Segment", template = "plotly_dark")
    fig.update_traces(text = filtered_df["Segment"], textposition = "inside")
    st.plotly_chart(fig,use_container_width=True)


import plotly.figure_factory as ff
st.subheader(":point_right: Month wise Sub-Category Sales Summary")
with st.expander("Summary_Table"):
    df_sample = df[0:5][["Region","State","City","Category","Sales","Profit","Quantity","Shipping Cost","Order Priority"]]
    fig = ff.create_table(df_sample, colorscale ="aggrnyl")
    st.plotly_chart(fig, use_container_width=True)

    filtered_df.iloc[2, 1] = np.nan  # Simulating a NaN value

    st.markdown("Month wise Sub-Category Table")
    filtered_df["month"] = filtered_df["Order Date"].dt.month_name()
    sub_category_Year = pd.pivot_table(data=filtered_df, values="Sales", index=["Sub-Category"], columns="month")

# Fill NaN values with 0 before applying the background gradient
    styled_df = sub_category_Year.fillna(0).style.background_gradient(cmap='rainbow_r')

# Display the styled DataFrame
    st.write(styled_df)

# Create a scatter plot
data1 = px.scatter(filtered_df, x = "Sales", y = "Profit", size = "Quantity")
data1['layout'].update(title="Relationship between Sales and Profits using Scatter Plot.",
                       titlefont = dict(size=20),xaxis = dict(title="Sales",titlefont=dict(size=19)),
                       yaxis = dict(title = "Profit", titlefont = dict(size=19)))
st.plotly_chart(data1,use_container_width=True)

# Create a scatter plot
data2 = px.scatter(filtered_df, x = "Sales", y = "Quantity", size = "Quantity")
data2['layout'].update(title="Relationship between Sales and quantitys using Scatter Plot.",
                       titlefont = dict(size=20),xaxis = dict(title="Sales",titlefont=dict(size=19)),
                       yaxis = dict(title = "quantity", titlefont = dict(size=19)))
st.plotly_chart(data2,use_container_width=True)

# Heatmap Analysis
st.subheader(":fire: Heatmap Analysis")
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
df_numeric = df[numeric_columns]
corr = df_numeric.corr(method='pearson')
fig, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(corr, annot=True, fmt='.2f', cbar=True, linewidth=0.5, ax=ax)
ax.set_title("Heatmap Analysis")
st.pyplot(fig)


# Top 10 products based on sales
top_sales_products = df.groupby('Product Name')['Sales'].sum().nlargest(10).reset_index()
st.subheader("Top 10 Products Based on Sales")
top_sales_products_sorted = top_sales_products.sort_values(by='Sales')
fig, ax = plt.subplots()
sns.barplot(x='Sales', y='Product Name', data=top_sales_products_sorted, ax=ax)
ax.set_xlabel('Sales')
ax.set_ylabel('Product Name')
fig_plotly = px.bar(top_sales_products_sorted, x='Sales', y='Product Name', orientation='h', title="Top 10 Products Based on Sales (Ascending Order)")
st.plotly_chart(fig_plotly)




# Create Box and Whisker Plot 
with col1:
    st.subheader("Box and Whisker Plot: Market vs. Sales")
    fig = px.box(df, x='Market', y='Sales', color='Market', title="Market vs. Sales",
             labels={"Market": "Market", "Sales": "Sales"})
    fig.update_layout(
    xaxis=dict(title=dict(text="Market", font=dict(size=14))),
    yaxis=dict(title=dict(text="Sales", font=dict(size=14))),
    font=dict(size=10)
    )
    st.plotly_chart(fig)


# Create Bubble Plot
st.subheader(":chart_with_upwards_trend: Bubble Plot Analysis")
bubble_data = df[['Sales', 'Profit', 'Quantity', 'Discount','Product Name']]
fig_bubble = px.scatter(
    bubble_data,
    x='Sales',
    y='Profit',
    size='Quantity',
    color=bubble_data['Product Name'],
    hover_name=bubble_data['Product Name'],
    title='Bubble Plot: Sales vs Profit (Bubble size = Quantity, Color = Product Name)',
    labels={'Sales': 'Sales', 'Profit': 'Profit'},
    template='plotly',
    width=800,
    height=600
)
fig_bubble.update_traces(marker=dict(line=dict(width=1, color='white')))
st.plotly_chart(fig_bubble, use_container_width=True)


# density plot
fig_density = px.density_contour(df, x='Sales', y='Profit',
                                 title='Density Plot: Sales vs. Profit')
st.plotly_chart(fig_density, use_container_width=True)
fig = px.sunburst(df, path=['Region', 'Country', 'State', 'City'], values='Sales',
                  title='Sunburst Chart: Sales by Region, Country, State, and City')
fig.update_layout(margin=dict(l=0, r=0, b=0, t=40))
st.plotly_chart(fig, use_container_width=True)

#Map Plot
with col2:
    fig = px.choropleth(df, 
                    locations='Country',  # Column containing country names
                    locationmode='country names',  # Set the location mode
                    color='Sales',  # Column containing the color values
                    hover_name='Country',  # Column containing hover information
                    title='Map: Sales by Country',
                    color_continuous_scale='Viridis'  # Set the color scale
                    )
    st.plotly_chart(fig, use_container_width=True)


# Violin Plot: Sales by Category
with col1:
    st.subheader('Violin Plot: Sales by Category')
    plt.figure(figsize=(10, 6))
    ax = sns.violinplot(x='Category', y='Sales', data=df, palette='viridis')
    quartiles = df.groupby('Category')['Sales'].quantile([0.25, 0.5, 0.75]).unstack()
    for category in df['Category'].unique():
        q25, q50, q75 = quartiles.loc[category]
    ax.text(category, q25, f'Q1\n{q25:.2f}', ha='center', va='center', fontdict={'size': 8})
    ax.text(category, q50, f'Q2\n{q50:.2f}', ha='center', va='center', fontdict={'size': 8})
    ax.text(category, q75, f'Q3\n{q75:.2f}', ha='center', va='center', fontdict={'size': 8})
    plt.title('Violin Plot: Sales by Category')
    plt.xticks(rotation=45)  
    quartiles = df.groupby('Category')['Sales'].quantile([0.25, 0.5, 0.75]).unstack()
    fig = px.violin(df, x='Category', y='Sales', color='Category', title='Violin Plot: Sales by Category')
    for category in df['Category'].unique():
        q25, q50, q75 = quartiles.loc[category]
    fig.add_shape(type='line', x0=category, x1=category, y0=q25, y1=q75, line=dict(color='black', width=2))
    st.plotly_chart(fig)


top_customers = df.groupby('Customer Name')['Sales'].sum().sort_values(ascending=False).head(10)

# Bar Plot: Top 10 Customers by Sales
with col2:
    st.subheader('Top 10 Customers by Sales')
    plt.figure(figsize=(12, 6))
    sns.barplot(x=top_customers.index, y=top_customers.values, palette='viridis')
    plt.title('Top 10 Customers by Sales')
    plt.xticks(rotation=45) 
    fig_plotly = px.bar(x=top_customers.index, y=top_customers.values, color=top_customers.index, title='Top 10 Customers by Sales')
    st.plotly_chart(fig_plotly)


# Display the top market
st.subheader("Top Market")
top_market = df.groupby('Market')['Sales'].sum().idxmax()
st.write(top_market)

# Display the top profit-making product
st.subheader("Top Profit-Making Product")
top_profit_product = df.groupby('Product Name')['Sales'].sum().idxmax()
st.write(top_profit_product)

with st.expander("View Data"):
    st.write(filtered_df.iloc[:500,1:20:2].style.background_gradient(cmap="Oranges"))

# Download orginal DataSet
csv = df.to_csv(index = False).encode('utf-8')
st.download_button('Download Data', data = csv, file_name = "Data.csv",mime = "text/csv")
