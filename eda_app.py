import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import io
import missingno as msno
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
from sklearn.feature_selection import SelectKBest, f_regression, chi2
from wordcloud import WordCloud
from sklearn.cluster import KMeans

# Set page config at the very beginning
st.set_page_config(layout="wide")

def load_css():
    st.markdown("""
    <style>
    .reportview-container {
        background: #f0f2f6
    }
    .sidebar .sidebar-content {
        background: #ffffff
    }
    .Widget>label {
        color: #31333F;
        font-family: sans-serif;
    }
    .stTextInput>div>div>input {
        color: #31333F;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    st.title('Advanced EDA on Uploaded Dataset')
    st.subheader('Exploratory Data Analysis using Streamlit and Plotly')

    # File uploader allows user to add their own CSV
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        # Identify numeric and categorical columns
        numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()

        # Sidebar options
        analysis_options = [
            'Show Dataset', 'Show Data Info', 'Show Summary Statistics',
            'Show Missing Values', 'Show Distribution Plot', 'Show Box Plot',
            'Show Violin Plot', 'Correlation Heatmap', 'Show Scatter Plot',
            'Show Pair Plot', 'Show Correlation Analysis', 'Show Feature Scaling',
            'Show Feature Selection', 'Show Outlier Detection', 'Show Word Cloud',
            'Show Parallel Coordinates Plot', 'Show 3D Scatter Plot',
            'Show 3D K-Means Clustering'
        ]
        selected_analyses = st.sidebar.multiselect("Select Analyses", analysis_options)

        # Main content
        for analysis in selected_analyses:
            if analysis == 'Show Dataset':
                st.subheader("Dataset Preview")
                st.write(data.head())

            elif analysis == 'Show Data Info':
                st.subheader("Data Info")
                buffer = io.StringIO()
                data.info(buf=buffer)
                s = buffer.getvalue()
                st.text(s)

            elif analysis == 'Show Summary Statistics':
                st.subheader("Summary Statistics")
                st.write(data.describe(include='all'))

            elif analysis == 'Show Missing Values':
                st.subheader("Missing Values Matrix")
                fig, ax = plt.subplots(figsize=(10, 6))
                msno.matrix(data, ax=ax)
                st.pyplot(fig)

            elif analysis == 'Show Distribution Plot':
                st.subheader("Distribution Plot")
                column_to_plot = st.selectbox("Select Column for Histogram", data.columns, key='hist')
                if data[column_to_plot].dtype in ['int64', 'float64']:
                    fig = px.histogram(data, x=column_to_plot, nbins=20)
                else:
                    fig = px.bar(data[column_to_plot].value_counts().reset_index(), x='index', y=column_to_plot)
                st.plotly_chart(fig, use_container_width=True)

            elif analysis == 'Show Box Plot':
                st.subheader("Box Plot")
                column_to_plot = st.selectbox("Select Column for Box Plot", numeric_columns, key='box')
                fig = px.box(data, y=column_to_plot)
                st.plotly_chart(fig, use_container_width=True)

            elif analysis == 'Show Violin Plot':
                st.subheader("Violin Plot")
                column_to_plot = st.selectbox("Select Column for Violin Plot", numeric_columns, key='violin')
                fig = px.violin(data, y=column_to_plot)
                st.plotly_chart(fig, use_container_width=True)

            elif analysis == 'Correlation Heatmap':
                st.subheader("Correlation Heatmap")
                corr = data[numeric_columns].corr()
                fig = px.imshow(corr, text_auto=".2f")
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)

            elif analysis == 'Show Scatter Plot':
                st.subheader("Scatter Plot")
                x_axis = st.selectbox("Select X Axis", numeric_columns, index=0, key='scatter_x')
                y_axis = st.selectbox("Select Y Axis", numeric_columns, index=1, key='scatter_y')
                color_by = st.selectbox("Color By (Optional)", ["None"] + categorical_columns, index=0, key='scatter_color')
                
                if color_by == "None":
                    fig = px.scatter(data, x=x_axis, y=y_axis)
                else:
                    fig = px.scatter(data, x=x_axis, y=y_axis, color=color_by)
                
                st.plotly_chart(fig, use_container_width=True)

            elif analysis == 'Show Pair Plot':
                st.subheader("Pair Plot")
                selected_columns = st.multiselect("Select Columns for Pair Plot", numeric_columns)
                if len(selected_columns) > 1:
                    fig = px.scatter_matrix(data[selected_columns])
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Please select at least two columns for the pair plot.")

            elif analysis == 'Show Correlation Analysis':
                st.subheader("Correlation Analysis")
                var = st.selectbox("Select Variable", numeric_columns, index=0, key='corr_var')
                corr_method = st.radio("Correlation Method", ("Pearson", "Spearman"))
                
                if corr_method == "Pearson":
                    corr_matrix = data[numeric_columns].corr(method='pearson')
                else:
                    corr_matrix = data[numeric_columns].corr(method='spearman')
                
                corr_with_var = corr_matrix[var].sort_values(ascending=False)
                st.write(corr_with_var)

            elif analysis == 'Show Feature Scaling':
                st.subheader("Feature Scaling")
                selected_columns = st.multiselect("Select Columns for Scaling", numeric_columns)
                
                scaling_options = {
                    "StandardScaler": StandardScaler(),
                    "MinMaxScaler": MinMaxScaler(),
                    "RobustScaler": RobustScaler(),
                    "Normalizer": Normalizer()
                }
                
                selected_scaler = st.selectbox("Select Scaling Type", list(scaling_options.keys()))
                
                if len(selected_columns) > 0:
                    scaler = scaling_options[selected_scaler]
                    scaled_data = scaler.fit_transform(data[selected_columns])
                    scaled_df = pd.DataFrame(scaled_data, columns=[f"{col}_scaled" for col in selected_columns])
                    before_scaling_df = data[selected_columns]
                    comparison_df = pd.concat([before_scaling_df, scaled_df], axis=1)
                    st.write(comparison_df)
                else:
                    st.warning("Please select at least one column for feature scaling.")

            elif analysis == 'Show Feature Selection':
                st.subheader("Feature Selection")
                target_column = st.selectbox("Select Target Column", data.columns)
                k = st.slider("Select Number of Top Features", min_value=1, max_value=len(data.columns)-1, value=5)

                X = data.drop(columns=[target_column])
                y = data[target_column]

                if y.dtype in ['int64', 'float64']:
                    selector = SelectKBest(score_func=f_regression, k=k)
                else:
                    selector = SelectKBest(score_func=chi2, k=k)
                    X = X.select_dtypes(include=['int64', 'float64'])

                selector.fit(X, y)

                feature_scores = pd.DataFrame({'Feature': X.columns, 'Score': selector.scores_})
                feature_scores = feature_scores.sort_values(by='Score', ascending=False)

                st.write("Feature Importance Scores:")
                st.write(feature_scores)

                fig = px.bar(feature_scores, x='Feature', y='Score', title='Feature Importance Scores')
                st.plotly_chart(fig, use_container_width=True)

            elif analysis == 'Show Outlier Detection':
                st.subheader("Outlier Detection")
                selected_column = st.selectbox("Select Column for Outlier Detection", numeric_columns)

                Q1 = data[selected_column].quantile(0.25)
                Q3 = data[selected_column].quantile(0.75)
                IQR = Q3 - Q1

                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outliers = data[(data[selected_column] < lower_bound) | (data[selected_column] > upper_bound)]

                if outliers.empty:
                    st.write("No outliers found in the selected column.")
                else:
                    st.write("Outliers:")
                    st.write(outliers)

                fig = go.Figure()
                fig.add_trace(go.Box(y=data[selected_column], name=selected_column))
                fig.update_layout(title=f"Box Plot for {selected_column}")
                st.plotly_chart(fig, use_container_width=True)

            elif analysis == 'Show Word Cloud':
                st.subheader("Word Cloud")
                selected_column = st.selectbox("Select Text Column for Word Cloud", categorical_columns)
                text = ' '.join(data[selected_column].astype(str))
                wordcloud = WordCloud(background_color='white').generate(text)
                plt.figure(figsize=(10, 8))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                st.pyplot(plt)

            elif analysis == 'Show Parallel Coordinates Plot':
                st.subheader("Parallel Coordinates Plot")
                selected_columns = st.multiselect("Select Columns for Parallel Coordinates Plot", numeric_columns)
                if len(selected_columns) > 1:
                    fig = px.parallel_coordinates(data, dimensions=selected_columns)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Please select at least two columns for the parallel coordinates plot.")

            elif analysis == 'Show 3D Scatter Plot':
                st.subheader("3D Scatter Plot")
                x_column = st.selectbox("Select X-axis Column", numeric_columns, key='3d_scatter_x')
                y_column = st.selectbox("Select Y-axis Column", numeric_columns, key='3d_scatter_y')
                z_column = st.selectbox("Select Z-axis Column", numeric_columns, key='3d_scatter_z')
                color_column = st.selectbox("Select Color Column", ["None"] + categorical_columns, key='3d_scatter_color')
                
                if color_column == "None":
                    fig = px.scatter_3d(data, x=x_column, y=y_column, z=z_column)
                else:
                    fig = px.scatter_3d(data, x=x_column, y=y_column, z=z_column, color=color_column)
                
                st.plotly_chart(fig, use_container_width=True)

            elif analysis == 'Show 3D K-Means Clustering':
                st.subheader("3D K-Means Clustering")
                x_column = st.selectbox("Select X-axis Column", numeric_columns, key='3d_kmeans_x')
                y_column = st.selectbox("Select Y-axis Column", numeric_columns, key='3d_kmeans_y')
                z_column = st.selectbox("Select Z-axis Column", numeric_columns, key='3d_kmeans_z')
                n_clusters = st.number_input("Select Number of Clusters", min_value=2, max_value=10, value=3, step=1)
                
                X = data[[x_column, y_column, z_column]]
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                kmeans.fit(X)
                labels = kmeans.labels_
                
                fig = px.scatter_3d(data, x=x_column, y=y_column, z=z_column, color=labels.astype(str))
                st.plotly_chart(fig, use_container_width=True)

        # Recommendations and Insights
        st.subheader("Recommendations and Insights")
        st.write("Based on the analysis, here are some recommendations and insights:")
        
        # Missing Values
        missing_percentages = data.isnull().mean() * 100
        columns_with_missing = missing_percentages[missing_percentages > 0].index.tolist()
        if columns_with_missing:
            st.write("1. Missing Values:")
            for col in columns_with_missing:
                st.write(f"   - {col}: {missing_percentages[col]:.2f}% missing. Consider imputation or removal.")
        else:
            st.write("1. No missing values found in the dataset.")
        
        # Correlation
        if len(numeric_columns) > 1:
            corr_matrix = data[numeric_columns].corr()
            high_corr = (corr_matrix.abs() > 0.8) & (corr_matrix != 1.0)
            high_corr_pairs = [(corr_matrix.index[i], corr_matrix.columns[j]) 
                               for i in range(len(corr_matrix.index)) 
                               for j in range(i+1, len(corr_matrix.columns)) 
                               if high_corr.iloc[i,j]]
            if high_corr_pairs:
                st.write("2. High Correlation:")
                for pair in high_corr_pairs:
                    st.write(f"   - {pair[0]} and {pair[1]} have a high correlation. Consider feature selection or dimensionality reduction.")
            else:
                st.write("2. No highly correlated features found.")
        else:
            st.write("2. Not enough numeric columns to perform correlation analysis.")
        
        # Skewness
        if len(numeric_columns) > 0:
            skewed_features = data.select_dtypes(include=['float64', 'int64']).apply(lambda x: x.skew()).abs().sort_values(ascending=False)
            highly_skewed = skewed_features[skewed_features > 1]
            # Continuing from the previous code...

        # Skewness
        skewed_features = data.select_dtypes(include=['float64', 'int64']).apply(lambda x: x.skew()).abs().sort_values(ascending=False)
        highly_skewed = skewed_features[skewed_features > 1]
        if not highly_skewed.empty:
            st.write("3. Highly Skewed Features:")
            for feature, skew in highly_skewed.items():
                st.write(f"   - {feature}: Skewness = {skew:.2f}. Consider applying a transformation (e.g., log, square root) to normalize the distribution.")
        else:
            st.write("3. No highly skewed features found.")

        # Outliers
        st.write("4. Outliers:")
        for col in data.select_dtypes(include=['float64', 'int64']).columns:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
            if not outliers.empty:
                st.write(f"   - {col}: {len(outliers)} outliers detected. Consider investigating these data points or applying outlier treatment techniques.")

        # Class Imbalance (for classification tasks)
        categorical_columns = data.select_dtypes(include=['object', 'category']).columns
        if len(categorical_columns) > 0:
            st.write("5. Potential Class Imbalance:")
            for col in categorical_columns:
                value_counts = data[col].value_counts(normalize=True)
                if value_counts.max() > 0.75:  # If any class represents more than 75% of the data
                    st.write(f"   - {col}: Potential class imbalance detected. The majority class represents {value_counts.max()*100:.2f}% of the data.")
                    st.write("     Consider using techniques like oversampling, undersampling, or SMOTE for balanced learning.")

        # Feature Importance
        st.write("6. Feature Importance:")
        st.write("   To identify the most important features for your target variable, use the 'Show Feature Selection' analysis option.")

        # Data Quality
        st.write("7. Data Quality:")
        for col in data.columns:
            unique_count = data[col].nunique()
            total_count = len(data)
            if unique_count == total_count:
                st.write(f"   - {col}: All values are unique. This might be an ID column or a timestamp.")
            elif unique_count == 1:
                st.write(f"   - {col}: Contains only one unique value. Consider dropping this column as it doesn't provide any information.")

        # Recommendations for further analysis
        st.write("8. Further Analysis Recommendations:")
        st.write("   - Conduct hypothesis tests to validate any assumptions about the data.")
        st.write("   - Perform dimensionality reduction techniques like PCA if dealing with high-dimensional data.")
        st.write("   - Consider creating new features through feature engineering to capture more complex patterns.")
        st.write("   - If dealing with time-series data, analyze trends and seasonality patterns.")

        # UI Improvements
        st.sidebar.markdown("---")
        st.sidebar.subheader("About")
        st.sidebar.info("This is a generic EDA app created using Streamlit and Plotly. "
                        "Upload your CSV file and select the analyses you want to perform.")

        # Add a footer
        st.markdown("---")
        st.markdown("Created with ❤️ using Streamlit and Plotly")

def load_css():
    st.markdown("""
    <style>
    .reportview-container {
        background: #f0f2f6
    }
    .sidebar .sidebar-content {
        background: #ffffff
    }
    .Widget>label {
        color: #31333F;
        font-family: sans-serif;
    }
    .stTextInput>div>div>input {
        color: #31333F;
    }
    </style>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    load_css()
    main()
