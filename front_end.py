import streamlit as st
from main import get_all_stats
from PIL import Image
import os
import pandas as pd

i = 1
def save_image(uploaded_file, filename=None):
    # Ensure the folder exists
    path = "./user_images"
    if not filename:
        file_path = os.path.join(path, uploaded_file.name)
    else:
        file_path = os.path.join(path, filename)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def display_results(results):
    # Automatically extract the single entry from results
    photo_path, data = next(iter(results.items()))
    st.divider()

    st.subheader('Vanilla NIMA', help='Technical and aesthetic ratings out of 10 based on NIMA only.')
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Technical Quality", f"{data['nima_technical_result']:.2f}")
    with col2:
        st.metric("Aesthetic Quality", f"{data['nima_aesthetic_result']:.2f}")

    st.subheader('NIMA Augmented with CLIP', help='Aesthetic rating out of 10 based on NIMA with CLIP adjusted training data.')
    st.metric("Aesthetic Quality", f"{data['nima_clip_aesthetic_result']:.2f}")

    st.subheader('CLIP Aesthetic Results', help='Aesthetic rating defined by CLIP')
    cols = st.columns(3)
    cols[0].metric("Great Picture", f"{data['clip_aesthetic_result']['great picture']:.2f}%")
    cols[1].metric("Average Picture", f"{data['clip_aesthetic_result']['average picture']:.2f}%")
    cols[2].metric("Subpar Picture", f"{data['clip_aesthetic_result']['subpar picture']:.2f}%")

    st.subheader('CLIP Classifications', help="Photo content as defined by CLIP with hierarchical categorization")
    # Create DataFrame to display classifications
    classification_data = []
    for level, categories in data['clip_classification'].items():
        classification_data.append({
            'Level': level.capitalize(),
            'Classifications': ", ".join(sorted(categories))
        })
    df_classifications = pd.DataFrame(classification_data)
    st.table(df_classifications)



def main():
    global i
    st.header('Image Analysis and Classification')
    st.markdown('_with [NIMA](https://idealo.github.io/image-quality-assessment/) and [CLIP](https://openai.com/research/clip)_')
    options = ['Sample Photos','Upload Photo']
    try:
        image_source = st.radio('Image source:', options, horizontal=True, key = i)
        i += 1

        user_images_path = "./user_images"
        sample_images_path = "./my_test_images"
        
        if image_source == 'Upload Photo':
            uploaded_file = st.file_uploader("Upload a JPEG image:", type=['jpg'], key=i)
            i+=1
            if uploaded_file is not None:
                filename = uploaded_file.name
                base_name, ext = os.path.splitext(uploaded_file.name)
                ext = ext.lower()
                if ext in ['.jpg', '.jpeg']:
                    ext = '.jpg'
                filename = f"{base_name}{ext}"

                image = Image.open(uploaded_file)
                st.image(image, caption='Uploaded Image', use_column_width=True)
                saved_file_path = save_image(uploaded_file, filename)

                # Button to trigger model execution
                if st.button('Run Models'):
                    with st.spinner('Running models...'):
                        results = get_all_stats(saved_file_path)
                        display_results(results)
                        
        elif image_source == 'Sample Photos':
            # List all jpg files in the sample images directory
            available_samples = [file for file in os.listdir(sample_images_path) if file.endswith('.jpg')]
            # Move 'drone_shot.jpg' to the front of the list if it exists
            if 'drone_shot.jpg' in available_samples:
                available_samples.insert(0, available_samples.pop(available_samples.index('drone_shot.jpg')))

            selected_sample = st.selectbox('Select a sample photo:', available_samples, key=i)
            i+=1
            
            sample_image_path = os.path.join(sample_images_path, selected_sample)
            image = Image.open(sample_image_path)
            st.image(image, caption='Sample Image', use_column_width=True)
            
            # Button to trigger model execution on the selected sample
            if st.button('Run Models'):
                with st.spinner('Analyzing...'):
                    results = get_all_stats(sample_image_path)
                    display_results(results)
    except:
        st.error('Something went wrong. Reload and try again with the same image and then a different image.')

if __name__ == '__main__':
    main()