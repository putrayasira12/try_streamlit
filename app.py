import streamlit as st
from streamlit_option_menu import option_menu 
import home
import eda
# import prediction
from PIL import Image

st.set_page_config(
    page_title='Sentiment Analysis for Wardah Sunscreen', 
    layout='centered', #wide
    initial_sidebar_state='expanded'
)

col1, col2, col3 = st.columns([10, 1, 5])
# image_url = 'https://raw.githubusercontent.com/FTDS-assignment-bay/p2-final-project-ftds-001-sby-group-001/main/images/logo_crop_clean.png'
# col1.image(image_url, width=450)
st.write('# Sentiment Analysis for Wardah Sunscreen')
st.subheader('Analyze sentiment whether good or bad of our sunscreen products')
st.markdown('---')

selected = option_menu(None, ["About", "EDA", "Predict"], 
    icons=['house', 'file-earmark-bar-graph', 'search'], 
    menu_icon="cast", default_index=0, orientation="horizontal",
    styles={
        "icon": {"color": "orange", "font-size": "15px"}, 
        "nav-link": {"font-size": "15px", "text-align": "left", "margin":"1px", "--hover-color": "#eee"}, 
        "nav-link-selected": {"background-color": "grey"},
    }
)   

if selected == 'About':
    home.run()
elif selected == 'EDA':
    eda.run()
# else:
#     prediction.run()
