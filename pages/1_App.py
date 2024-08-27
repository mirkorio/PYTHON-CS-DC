import streamlit as st
import pandas as pd
import altair as alt
from backend.code_similarity_detection import extract_files, compare_files, sanitize_title
from backend.code_clustering import CodeClusterer, find_elbow_point
import os
import multiprocessing


def main():
    st.set_page_config(
        page_title="Python Code Similarity Detection and Clustering Tool",
        page_icon="logo/logo.png",  # Set your logo image as the page icon
    )
    st.markdown("<h3 style='text-align: center; margin: 20px;'>Code Similarity Detection and Clustering</h2>", unsafe_allow_html=True)

    # Adding an introductory section with markdown
    st.markdown("""
    This application enables you to detect and analyze code similarity by uploading Python files. 
    Get started by uploading your files to explore clustered similarity results, visualizations, and detailed comparisons.
    """)

    # Text input for activity title
    activity_title = st.text_input("Enter a title for the code activity")

    # Initialize session state variables
    if 'similarity_df' not in st.session_state:
        st.session_state.similarity_df = pd.DataFrame()

    if 'elbow_scores' not in st.session_state:
        st.session_state.elbow_scores = []

    if 'best_num_clusters' not in st.session_state:
        st.session_state.best_num_clusters = 2

    if 'clustered_data' not in st.session_state:
        st.session_state.clustered_data = pd.DataFrame()

    if 'silhouette_avg' not in st.session_state:
        st.session_state.silhouette_avg = None

    if 'davies_bouldin' not in st.session_state:
        st.session_state.davies_bouldin = None

    if 'extracted_files_content' not in st.session_state:
        st.session_state.extracted_files_content = {}

    if 'selected_pair' not in st.session_state:
        st.session_state.selected_pair = None

    if 'clustering_performed' not in st.session_state:
        st.session_state.clustering_performed = False

    # File Uploader
    uploaded_files = st.file_uploader("Upload Python files", type=['py'], accept_multiple_files=True)

    if uploaded_files:
        st.write(f"Number of uploaded files: {len(uploaded_files)}")
        if st.button("Process Files"):
            with st.spinner("Processing files..."):
                extracted_files, extracted_files_content = extract_files(uploaded_files)
                st.session_state.extracted_files_content = extracted_files_content
                file_pairs = [(extracted_files[i], extracted_files[j]) for i in range(len(extracted_files)) for j in range(i + 1, len(extracted_files))]

                pool = multiprocessing.Pool()
                results = pool.starmap(compare_files, [(pair, st.session_state.extracted_files_content) for pair in file_pairs])

                results = [result for result in results if all(result)]
                pool.close()
                pool.join()

                try:
                    similarity_df = pd.DataFrame(results, columns=['Code1', 'Code2', 'Text_Similarity', 'Structural_Similarity', 'Weighted_Similarity'])
                    st.session_state.similarity_df = similarity_df

                    st.success("Processing complete!")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
    else:
        st.info('Please upload Python files.')

    # Allow users to tweak parameters
    st.sidebar.header("Clustering Parameters")
    num_clusters = st.sidebar.slider("Number of Clusters", min_value=2, max_value=10, value=st.session_state.best_num_clusters)

    # Show similarity results
    if 'similarity_df' in st.session_state and not st.session_state.similarity_df.empty:
        st.header("Similarity Results")
        st.dataframe(st.session_state.similarity_df)

        # Clustering
        if st.button("Perform Clustering"):
            with st.spinner("Performing clustering..."):
                # Calculate the elbow method automatically
                clusterer = CodeClusterer(num_clusters=num_clusters)
                clusterer.load_data(st.session_state.similarity_df)
                clusterer.calculate_elbow(max_clusters=10)
                st.session_state.elbow_scores = clusterer.elbow_scores
                st.session_state.best_num_clusters = find_elbow_point(clusterer.elbow_scores)

                clusterer = CodeClusterer(num_clusters=num_clusters)
                clusterer.load_data(st.session_state.similarity_df)

                try:
                    features = clusterer.cluster_codes()
                    st.session_state.clustered_data = clusterer.get_clustered_data()
                    st.session_state.silhouette_avg = clusterer.silhouette_avg
                    st.session_state.davies_bouldin = clusterer.davies_bouldin
                    st.session_state.clustering_performed = True

                    st.success("Clustering complete!")
                except ValueError as e:
                    st.error(f"Error clustering data: {str(e)}")

        # Display Elbow Chart and Best Number of Clusters
        if st.session_state.clustering_performed and 'elbow_scores' in st.session_state and st.session_state.elbow_scores:
            st.header("Elbow Method")
            elbow_chart = alt.Chart(pd.DataFrame({
                'Clusters': list(range(2, len(st.session_state.elbow_scores) + 2)),
                'Inertia': st.session_state.elbow_scores
            })).mark_line().encode(
                x='Clusters:O',
                y='Inertia:Q'
            ).interactive()
            st.altair_chart(elbow_chart, use_container_width=True)
            st.write(f"Best Number of Clusters: {st.session_state.best_num_clusters}")

        # Display Clustering Visualization (Scatter Plot)
        if st.session_state.clustering_performed and 'clustered_data' in st.session_state and not st.session_state.clustered_data.empty:
            st.header("Scatter Plot")
            cluster_chart = alt.Chart(st.session_state.clustered_data).mark_circle(size=60).encode(
                x='Text_Similarity',
                y='Structural_Similarity',
                color='Cluster:N',
                tooltip=['Code1', 'Code2', 'Text_Similarity', 'Structural_Similarity', 'Weighted_Similarity']
            ).interactive()
            st.altair_chart(cluster_chart, use_container_width=True)

            # Display Silhouette Plot and Scores
            if 'clustered_data' in st.session_state and 'features' in locals():
                st.header("Silhouette Plot")
                silhouette_data = clusterer.get_silhouette_data(features)
                silhouette_chart = alt.Chart(silhouette_data).mark_bar().encode(
                    x='Silhouette Value',
                    y='Cluster:N',
                    color='Cluster:N',
                    tooltip=['Silhouette Value', 'Cluster']
                ).interactive()
                st.altair_chart(silhouette_chart, use_container_width=True)
                st.write(f"Silhouette Score: {st.session_state.silhouette_avg:.4f}")
                st.write(f"Davies-Bouldin Index: {st.session_state.davies_bouldin:.4f}")

            # Display Clustered codes from highest to lowest weighted similarity
            st.header("Clustered Codes")
            clustered_data_sorted = st.session_state.clustered_data.sort_values(by='Weighted_Similarity', ascending=False)
            st.dataframe(clustered_data_sorted)

            # Side-by-Side Code Comparison
            st.header("Side-by-Side Code Comparison")
            code_pairs = st.session_state.similarity_df[['Code1', 'Code2']].apply(tuple, axis=1).tolist()
            selected_pair = st.selectbox("Select a pair of files to compare", options=code_pairs)
            st.session_state.selected_pair = selected_pair

            if selected_pair:
                code1, code2 = selected_pair
                code1_path = [key for key in st.session_state.extracted_files_content.keys() if os.path.basename(key) == code1]
                code2_path = [key for key in st.session_state.extracted_files_content.keys() if os.path.basename(key) == code2]

                if code1_path and code2_path:
                    code1_content = st.session_state.extracted_files_content.get(code1_path[0], "Content not found.")
                    code2_content = st.session_state.extracted_files_content.get(code2_path[0], "Content not found.")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.code(code1_content, language='python')
                        st.write(f"Text Similarity: {st.session_state.similarity_df[(st.session_state.similarity_df['Code1'] == code1) & (st.session_state.similarity_df['Code2'] == code2)]['Text_Similarity'].values[0]:.4f}")
                        st.write(f"Structural Similarity: {st.session_state.similarity_df[(st.session_state.similarity_df['Code1'] == code1) & (st.session_state.similarity_df['Code2'] == code2)]['Structural_Similarity'].values[0]:.4f}")
                        st.write(f"Weighted Similarity: {st.session_state.similarity_df[(st.session_state.similarity_df['Code1'] == code1) & (st.session_state.similarity_df['Code2'] == code2)]['Weighted_Similarity'].values[0]:.4f}")

                    with col2:
                        st.code(code2_content, language='python')
                        st.write(f"Text Similarity: {st.session_state.similarity_df[(st.session_state.similarity_df['Code1'] == code1) & (st.session_state.similarity_df['Code2'] == code2)]['Text_Similarity'].values[0]:.4f}")
                        st.write(f"Structural Similarity: {st.session_state.similarity_df[(st.session_state.similarity_df['Code1'] == code1) & (st.session_state.similarity_df['Code2'] == code2)]['Structural_Similarity'].values[0]:.4f}")
                        st.write(f"Weighted Similarity: {st.session_state.similarity_df[(st.session_state.similarity_df['Code1'] == code1) & (st.session_state.similarity_df['Code2'] == code2)]['Weighted_Similarity'].values[0]:.4f}")
           
            # Download buttons
            if st.session_state.similarity_df is not None and not st.session_state.similarity_df.empty:
                st.header("Download Results")
                # Download button for clustered codes
                if not st.session_state.clustered_data.empty:
                    df = st.session_state.clustered_data[['Code1', 'Code2', 'Text_Similarity', 'Structural_Similarity', 'Weighted_Similarity', 'Cluster']]
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download Clustered Codes",
                        data=csv,
                        file_name=f"{activity_title}_clustered_codes.csv",
                        mime="text/csv"
                    )
                    
if __name__ == "__main__":
    main()