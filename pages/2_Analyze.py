import streamlit as st
import pandas as pd
import altair as alt

# Initialize session state for storing the uploaded data
if 'df' not in st.session_state:
    st.session_state.df = None

st.set_page_config(
        page_title="Python Code Similarity Detection and Clustering Tool",
        page_icon="logo/logo.png",  # Set your logo image as the page icon
    )

# Title of the Streamlit app
st.title('Clustered Code Similarity Analysis')

# Adding an introductory section with markdown
st.markdown("""
This application allows you to analyze clustered code similarity using a CSV file. 
Upload your data to get started and explore various interactive visualizations and filters.
""")

# Uploading the CSV file
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(uploaded_file)

    # Store the dataframe in session state
    st.session_state.df = df

if st.session_state.df is not None:
    df = st.session_state.df

    # Display the dataframe with an expander to save space
    with st.expander("View Uploaded Data"):
        st.write(df)

    # Display the detected columns
    st.write("Detected Columns:", df.columns.tolist())

    # Normalize column names for consistency
    df.columns = df.columns.str.strip().str.lower()

    # Define required columns in lowercase
    required_columns = ['code1', 'code2', 'text_similarity', 'structural_similarity', 'weighted_similarity', 'cluster']

    # Check if all required columns are present in the dataframe
    if all(column in df.columns for column in required_columns):
        # Rename columns back to original names for consistency in visualizations
        df.columns = ['Code1', 'Code2', 'Text_Similarity', 'Structural_Similarity', 'Weighted_Similarity', 'Cluster']
        
        # Summary statistics with an expander
        with st.expander("Summary Statistics"):
            st.write(df.describe())

        # Filter options in the sidebar
        st.sidebar.header('Filter Options')
        selected_cluster = st.sidebar.multiselect('Select cluster(s) to visualize', options=df['Cluster'].unique(), default=df['Cluster'].unique())
        text_similarity_range = st.sidebar.slider('Text Similarity Range', 0.0, 1.0, (0.0, 1.0))
        structural_similarity_range = st.sidebar.slider('Structural Similarity Range', 0.0, 1.0, (0.0, 1.0))
        weighted_similarity_range = st.sidebar.slider('Weighted Similarity Range', 0.0, 1.0, (0.0, 1.0))

        # Filter the dataframe based on user selection
        filtered_df = df[
            (df['Cluster'].isin(selected_cluster)) &
            (df['Text_Similarity'].between(*text_similarity_range)) &
            (df['Structural_Similarity'].between(*structural_similarity_range)) &
            (df['Weighted_Similarity'].between(*weighted_similarity_range))
        ]

        # Display filtered dataframe
        st.write("Filtered Data", filtered_df)

        # Enhanced visualizations with custom themes and layering
        st.subheader('Text Similarity vs Structural Similarity')
        scatter_plot = alt.Chart(filtered_df).mark_circle(size=60).encode(
            x=alt.X('Text_Similarity', title='Text Similarity'),
            y=alt.Y('Structural_Similarity', title='Structural Similarity'),
            color=alt.Color('Weighted_Similarity', scale=alt.Scale(scheme='viridis')),
            tooltip=['Code1', 'Code2', 'Text_Similarity', 'Structural_Similarity', 'Weighted_Similarity']
        ).interactive().properties(
            width=800,
            height=400
        )

        st.altair_chart(scatter_plot, use_container_width=True)

        st.subheader('Number of Code Pairs in Each Cluster')
        cluster_count = filtered_df['Cluster'].value_counts().reset_index()
        cluster_count.columns = ['Cluster', 'Count']
        bar_chart = alt.Chart(cluster_count).mark_bar().encode(
            x=alt.X('Cluster:N', title='Cluster'),
            y=alt.Y('Count:Q', title='Number of Code Pairs'),
            color='Cluster:N',
            tooltip=['Cluster', 'Count']
        ).properties(
            width=800,
            height=400
        )

        st.altair_chart(bar_chart, use_container_width=True)

        st.subheader('Histograms of Similarity Metrics')
        for column in ['Text_Similarity', 'Structural_Similarity', 'Weighted_Similarity']:
            hist_chart = alt.Chart(filtered_df).mark_bar().encode(
                alt.X(column, bin=alt.Bin(maxbins=30), title=column.replace('_', ' ').title()),
                y=alt.Y('count()', title='Frequency'),
                tooltip=[column, 'count()']
            ).properties(
                width=300,
                height=300,
                title=f'Distribution of {column.replace("_", " ").title()}'
            )
            st.altair_chart(hist_chart, use_container_width=True)

        st.subheader('Pair Plot of Similarity Metrics')
        pair_plot = alt.Chart(filtered_df).mark_circle(size=60).encode(
            x=alt.X(alt.repeat("column"), type='quantitative', title=None),
            y=alt.Y(alt.repeat("row"), type='quantitative', title=None),
            color=alt.Color('Cluster:N', legend=alt.Legend(title="Cluster")),
            tooltip=['Code1', 'Code2', 'Text_Similarity', 'Structural_Similarity', 'Weighted_Similarity']
        ).properties(
            width=250,
            height=250
        ).repeat(
            row=['Text_Similarity', 'Structural_Similarity', 'Weighted_Similarity'],
            column=['Text_Similarity', 'Structural_Similarity', 'Weighted_Similarity']
        ).interactive()

        st.altair_chart(pair_plot, use_container_width=True)

        # Download button for filtered data in the sidebar
        st.sidebar.download_button(
            label="Download Filtered Data as CSV",
            data=filtered_df.to_csv(index=False),
            file_name='filtered_data.csv',
            mime='text/csv'
        )

    else:
        st.error('The uploaded file does not contain the required columns.')
else:
    st.info('Please upload a CSV file to analyze.')
