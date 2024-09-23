import streamlit as st
import pandas as pd
import altair as alt

# Initialize session state for storing the uploaded data
if 'df' not in st.session_state:
    st.session_state.df = None

st.set_page_config(
    page_title="Analyze",
    page_icon="logo/logo.png",  # Set your logo image as the page icon
)

# Title of the Streamlit app
st.markdown("<h3 style='text-align: center; margin: 20px;'>Clustered Code Similarity Analysis</h3>", unsafe_allow_html=True)

# Adding an introductory section with markdown
st.markdown("""
This application allows you to analyze clustered code similarity using a CSV file. 
Upload your data to get started and explore various interactive visualizations and filters.
""")

# Function to apply color based on similarity score
def apply_color(val):
    """
    Color code cells based on the Weighted Similarity percentage.
    """
    if val >= 75:
        color = 'background-color: #A04747'#red
    elif 50 <= val < 75:
        color = 'background-color: #D8A25E'#orange
    elif 25 <= val < 50:
        color = 'background-color: #EEDF7A'#yellow
    elif 1 <= val < 25:
        color = 'background-color: #557C56'#green
    else:
        color = 'background-color: #6A9AB0'#blue
    return color

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
        st.write("Full DataFrame")
        # Normalize column names for consistency
        df.columns = df.columns.str.strip().str.lower()

        # Format columns to two decimal places and apply color formatting to Weighted_Similarity_%
        df[['text_similarity_%', 'structural_similarity_%', 'weighted_similarity_%']] = df[['text_similarity_%', 'structural_similarity_%', 'weighted_similarity_%']].round(2)

        styled_df = df.style.format({
            'text_similarity_%': '{:.2f}%',
            'structural_similarity_%': '{:.2f}%',
            'weighted_similarity_%': '{:.2f}%'
        }).applymap(apply_color, subset=['weighted_similarity_%'])
        
        st.dataframe(styled_df)

    # Display the detected columns
    st.write("Detected Columns:", df.columns.tolist())

    # Define required columns in lowercase
    required_columns = ['code1', 'code2', 'text_similarity_%', 'structural_similarity_%', 'weighted_similarity_%', 'cluster']

    # Check if all required columns are present in the dataframe
    if all(column in df.columns for column in required_columns):
        # Rename columns back to original names for consistency in visualizations
        df.columns = ['Code1', 'Code2', 'Text_Similarity_%', 'Structural_Similarity_%', 'Weighted_Similarity_%', 'Cluster']
        
        # Summary statistics with an expander
        with st.expander("Summary Statistics"):
            st.write(df.describe())

        # Filter options in the sidebar
        st.sidebar.header('Filter Options')
        selected_cluster = st.sidebar.multiselect('Select cluster(s) to visualize', options=df['Cluster'].unique(), default=df['Cluster'].unique())
        text_similarity_range = st.sidebar.slider('Text Similarity Range (%)', 0.0, 100.0, (0.0, 100.0))
        structural_similarity_range = st.sidebar.slider('Structural Similarity Range (%)', 0.0, 100.0, (0.0, 100.0))
        weighted_similarity_range = st.sidebar.slider('Weighted Similarity Range (%)', 0.0, 100.0, (0.0, 100.0))

        # Filter the dataframe based on user selection
        filtered_df = df[
            (df['Cluster'].isin(selected_cluster)) &
            (df['Text_Similarity_%'].between(*text_similarity_range)) &
            (df['Structural_Similarity_%'].between(*structural_similarity_range)) &
            (df['Weighted_Similarity_%'].between(*weighted_similarity_range))
        ]

        # Display filtered dataframe with formatted similarity columns
        st.write("Filtered Data")
        # Expander for Color Labels Explanation with styled text
        with st.expander("Color Labels Explanation"):
            st.markdown("""
            <p style='color: #6A9AB0; padding: 5px; border-radius: 5px;'>
            <strong>Blue</strong>: 0% similarity score or not similar.</p>
            
            <p style='color: #557C56; padding: 5px; border-radius: 5px;'>
            <strong>Green</strong>: 1% - 24% very low similarity score.</p>
            
            <p style='color: #EEDF7A; padding: 5px; border-radius: 5px;'>
            <strong>Yellow</strong>: 25% - 49% low similarity score.</p>
            
            <p style='color: #D8A25E; padding: 5px; border-radius: 5px;'>
            <strong>Orange</strong>: 50% - 74% mid similarity score.</p>
            
            <p style='color: #A04747; padding: 5px; border-radius: 5px;'>
            <strong>Red</strong>: 75% - 100% high similarity score.</p>
            """, unsafe_allow_html=True)


        filtered_df[['Text_Similarity_%', 'Structural_Similarity_%', 'Weighted_Similarity_%']] = filtered_df[['Text_Similarity_%', 'Structural_Similarity_%', 'Weighted_Similarity_%']].round(2)

        styled_filtered_df = filtered_df.style.format({
            'Text_Similarity_%': '{:.2f}%',
            'Structural_Similarity_%': '{:.2f}%',
            'Weighted_Similarity_%': '{:.2f}%'
        }).applymap(apply_color, subset=['Weighted_Similarity_%'])

        st.dataframe(styled_filtered_df)

        # Enhanced visualizations with custom themes and layering
        st.subheader('Text Similarity vs Structural Similarity')

        # Define the custom color scale based on Weighted_Similarity_% thresholds
        color_scale = alt.Scale(
            domain=[0, 1, 25, 50, 75, 100],
            range=['#6A9AB0', '#557C56', '#EEDF7A', '#D8A25E', '#A04747']
        )

        scatter_plot = alt.Chart(filtered_df).mark_circle(size=60).encode(
            x=alt.X('Text_Similarity_%', title='Text Similarity (%)'),
            y=alt.Y('Structural_Similarity_%', title='Structural Similarity (%)'),
            color=alt.Color('Weighted_Similarity_%', scale=color_scale, legend=alt.Legend(title="Weighted Similarity (%)")),
            tooltip=['Code1', 'Code2', 'Text_Similarity_%', 'Structural_Similarity_%', 'Weighted_Similarity_%']
        ).interactive().properties(
            width=800,
            height=400
        )

        st.altair_chart(scatter_plot, use_container_width=True)

        # Calculate and display overall similarity for each code
        st.write("""
        Each code's Average Similarity Scores
        """)

        # Calculate the overall/average similarity metrics for each code
        overall_similarity = df.groupby('Code1').agg(
            average_text_similarity=('Text_Similarity_%', 'mean'),
            average_structural_similarity=('Structural_Similarity_%', 'mean'),
            average_weighted_similarity=('Weighted_Similarity_%', 'mean')
        ).reset_index()

        # Round the results for better display
        overall_similarity[['average_weighted_similarity', 'average_text_similarity', 'average_structural_similarity']] = overall_similarity[
            ['average_weighted_similarity', 'average_text_similarity', 'average_structural_similarity']].round(2)

        # Sort the DataFrame from highest to lowest average_weighted_similarity
        overall_similarity = overall_similarity.sort_values(by='average_weighted_similarity', ascending=False)

        # Sidebar filter for overall similarity
        weighted_similarity_range = st.sidebar.slider('Average Weighted Similarity Range (%)', 0.0, 100.0, (0.0, 100.0))

        # Filter the overall similarity DataFrame based on user selection
        filtered_overall_similarity = overall_similarity[(overall_similarity['average_weighted_similarity'].between(*weighted_similarity_range)) ]

        # Apply the existing apply_color function to color-code the similarity scores
        styled_filtered_overall_similarity = filtered_overall_similarity.style.applymap(
            apply_color, 
            subset=['average_weighted_similarity']
        )

        # Add percentage formatting after applying the color
        styled_filtered_overall_similarity = styled_filtered_overall_similarity.format({
            'average_weighted_similarity': '{:.2f}%',
            'average_text_similarity': '{:.2f}%',
            'average_structural_similarity': '{:.2f}%'
        })

        # Display the styled filtered dataframe
        st.dataframe(styled_filtered_overall_similarity)

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
        for column in ['Text_Similarity_%', 'Structural_Similarity_%', 'Weighted_Similarity_%']:
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
            tooltip=['Code1', 'Code2', 'Text_Similarity_%', 'Structural_Similarity_%', 'Weighted_Similarity_%']
        ).properties(
            width=250,
            height=250
        ).repeat(
            row=['Text_Similarity_%', 'Structural_Similarity_%', 'Weighted_Similarity_%'],
            column=['Text_Similarity_%', 'Structural_Similarity_%', 'Weighted_Similarity_%']
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
