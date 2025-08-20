import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.cluster import KMeans

# Page config
st.set_page_config(page_title="Customer Segmentation", layout="wide")

# Sidebar branding
st.sidebar.title("🔧 Segmentation Controls")
st.sidebar.markdown("Customize clustering and explore customer behavior.")
n_clusters = st.sidebar.slider("Number of clusters", 2, 10, 5)
show_elbow = st.sidebar.checkbox("Show Elbow Method")
selected_features = st.sidebar.multiselect(
    "Features for clustering",
    ['Annual Income (k$)', 'Spending Score (1-100)', 'Age'],
    default=['Annual Income (k$)', 'Spending Score (1-100)']
)

# File upload
uploaded_file = st.sidebar.file_uploader("Upload your dataset", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("Mall_Customers.csv")

# Title
st.title("🛍️ Customer Segmentation Dashboard")
st.markdown("Use K-Means clustering to uncover customer segments based on income, spending, and age.")

# Preview data
st.subheader("📄 Raw Data Preview")
st.dataframe(df.head())

# Feature selection
x = df[selected_features]

# Elbow Method
if show_elbow:
    wcss = []
    for k in range(1, 11):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(x)
        wcss.append(km.inertia_)
    fig, ax = plt.subplots()
    ax.plot(range(1, 11), wcss, marker='o', linestyle='--')
    ax.set_title("Elbow Method")
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("WCSS")
    st.pyplot(fig)

# Clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(x)

# Dynamic segment labeling
centers = pd.DataFrame(kmeans.cluster_centers_, columns=selected_features)
sorted_centers = centers.sort_values(by='Annual Income (k$)', ascending=True).reset_index(drop=True)
segment_names = ["Low Income", "Mid Income", "High Income", "Very High Income", "Luxury Spenders"]
sorted_centers['Segment'] = segment_names[:n_clusters]
label_map = dict(zip(sorted_centers.index, sorted_centers['Segment']))
df['Segment'] = df['cluster'].map(label_map)

# Cluster visualization
st.subheader("📊 Cluster Visualization")
fig2 = px.scatter(
    df,
    x='Annual Income (k$)',
    y='Spending Score (1-100)',
    color='cluster',
    hover_data=['CustomerID', 'Age', 'Segment'],
    title='Customer Clusters'
)
st.plotly_chart(fig2, use_container_width=True)

# Segment summary
st.subheader("📌 Segment Summary")
st.dataframe(df[['CustomerID', 'Annual Income (k$)', 'Spending Score (1-100)', 'cluster', 'Segment']].head())

# Cluster statistics
st.subheader("📈 Cluster Statistics")
st.dataframe(df.groupby('cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].describe())

# Cluster profile cards
st.subheader("🧠 Cluster Profiles")
cols = st.columns(n_clusters)
for i, cluster_id in enumerate(sorted(df['cluster'].unique())):
    cluster_data = df[df['cluster'] == cluster_id]
    segment_name = label_map.get(cluster_id, 'Segment')
    with cols[i]:
        st.markdown(f"### 🧍 Cluster {cluster_id}")
        st.write(f"**Segment**: {segment_name}")
        st.write(f"**Avg Income**: {cluster_data['Annual Income (k$)'].mean():.2f}k$")
        st.write(f"**Avg Spending**: {cluster_data['Spending Score (1-100)'].mean():.2f}")
        st.write(f"**Avg Age**: {cluster_data['Age'].mean():.1f}")

# Download segmented data
st.subheader("📥 Download Segmented Data")
csv = df.to_csv(index=False).encode('utf-8')
st.download_button("Download CSV", csv, "segmented_customers.csv", "text/csv")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by N. Mariyam — [LinkedIn](https://www.linkedin.com/in/yourprofile) | [GitHub](https://github.com/yourrepo)")
