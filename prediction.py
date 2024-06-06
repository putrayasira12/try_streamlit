import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import word_tokenize
from tensorflow.keras.models import load_model
import plotly.graph_objects as go


nltk.download('stopwords')
nltk.download('punkt')

def run():
    # Load Model Affiliate
    model_affiliate = load_model('model_affiliate')

    # Load Model Sentiment
    model_sentiment = load_model('model_sentiment')  # Loading from SavedModel format

    # Choice of input: Upload or Manual Input
    inputType = st.selectbox("How would you like to input data ?", ["Upload Excel or CSV File", "Manual Input"])
    st.markdown('---')

    # Create Function for Prediction
    def predictData(df):
        totalSentences = len(df)

        if totalSentences < 1:
            st.write('## There is no sentences on this data, please check again.')
        else:
            # Preprocessing data
            # Define Stopwords
            stpwds_id = list(set(stopwords.words('indonesian')))
            stpwds_id = stpwds_id + ['list_of_additional_stopwords']
            # Define Stemming
            stemmer = StemmerFactory().create_stemmer()

            def text_preprocessing(text):
                # Case folding
                text = text.lower()

                # Mention removal
                text = re.sub("@[A-Za-z0-9_]+", " ", text)

                # Hashtags removal
                text = re.sub("#[A-Za-z0-9_]+", " ", text)

                # Newline removal (\n)
                text = re.sub(r"\\n", " ", text)

                # Whitespace removal
                text = text.strip()

                # Repeated letters removal
                text = re.sub(r'(\w)\1+\b', r'\1', text)

                # URL removal
                text = re.sub(r"http\S+", " ", text)
                text = re.sub(r"www.\S+", " ", text)

                # Non-letter removal (such as emoticon, symbol (like μ, $, 兀), etc)
                text = re.sub("[^A-Za-z\s']", " ", text)

                # Tokenization
                tokens = word_tokenize(text)

                # Stopwords removal
                tokens = [word for word in tokens if word not in stpwds_id]

                # Stemming
                tokens = [stemmer.stem(word) for word in tokens]

                # Combining Tokens
                text = ' '.join(tokens)

                return text

            df['text_processed'] = df['full_text'].apply(lambda x: text_preprocessing(x))
            # Prediction for affiliate model
            pred_test = model_affiliate(df['text_processed'])

            # Convert predictions from probabilities to class labels
            pred_test_class = np.where(pred_test.numpy() >= 0.5, 1, 0)
            df['Affiliate'] = pred_test_class

    # A. For CSV
    if inputType == "Upload Excel or CSV File":
        dl_1, dl_2, dl_3 = st.columns([3, 3, 3])
        with open('sunscreen_labeled_data_test_nolabel.csv', 'rb') as file:
            dl_2.download_button(
                label='💾 Download Data Example',
                data=file,
                file_name='data_example.csv',
                mime='text/csv'
            )

        uploaded_file = st.file_uploader("Choose Excel or CSV file", type=["csv", "xlsx"], accept_multiple_files=False)
        if uploaded_file is not None:
            split_file_name = os.path.splitext(uploaded_file.name)
            
            file_extension = split_file_name[1]

            if file_extension == '.csv':
                df = pd.read_csv(uploaded_file)
            else:    
                df = pd.read_excel(uploaded_file)
            predictData(df)
            df_sentiment_lanjut = df[df['Affiliate'] == 0]
            df_sentiment_galanjut = df[df['Affiliate'] == 1]
            st.write('## Dari total', len(df),' didapatkan :')
            st.write('## Jumlah data tergolong affiliate sebanyak', len(df_sentiment_galanjut))
            st.write('## Jumlah data tergolong non affiliate sebanyak', len(df_sentiment_lanjut))

            # Menghitung jumlah setiap kategori
            a = df['Affiliate'].value_counts()
            labels = ['Affiliate','Non Affiliate']  # Ambil label langsung dari value_counts
            values = a.values.tolist()  # Ambil nilai langsung dari value_counts

            # Membuat grafik pie dengan lubang
            fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
            # Menambahkan judul pada grafik
            fig.update_layout(title_text='Perbandingan Tweets Affiliate dan Non Affiliate',title_x=0.5,title_xanchor='center')
            # Menampilkan grafik di Streamlit
            st.plotly_chart(fig)
            if len(df_sentiment_galanjut)>len(df_sentiment_lanjut):
                st.write('''
                        ### Banyak orang yang memanfaatkan media sosial X untuk menyebarkan konten afiliasi untuk menggait keuntungan karena afiliasi adalah salah satu cara marketing yang digunakan oleh perusahaan.
                        ''')
            else:
                st.write('''
                        ### Banyak orang yang telah menggunakan produk dan memberikan pandangan mereka terkait produk yang digunakan di media sosial X.
                        ''')

            pred_test_sentiment = model_sentiment.predict(df_sentiment_lanjut['text_processed'])

            # Convert predictions from probabilities to class labels
            pred_test_sentiment_class = np.where(pred_test_sentiment >= 0.5, 1, 0)
            df_sentiment_lanjut['Sentiment'] = pred_test_sentiment_class
            
            df_sentiment_positive = df_sentiment_lanjut[df_sentiment_lanjut['Sentiment'] == 1]
            df_sentiment_negative = df_sentiment_lanjut[df_sentiment_lanjut['Sentiment'] == 0]
            st.write('## Dari total ', len(df_sentiment_lanjut), 'data non affiliate, didapati bahwa :')
            st.write('## Sentimen positif sebanyak', len(df_sentiment_positive))
            st.write('## Sentimen negatif sebanyak', len(df_sentiment_negative))
            
            # Menghitung jumlah setiap kategori
            a = df_sentiment_lanjut['Sentiment'].value_counts()
            labels = ['Negatif','Positif']  # Ambil label langsung dari value_counts
            values = a.values.tolist()  # Ambil nilai langsung dari value_counts

            # Membuat grafik pie dengan lubang
            fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
            # Menambahkan judul pada grafik
            fig.update_layout(title_text='Prediksi Perbandingan Sentiment Sunscreen',title_x=0.5,title_xanchor='center')
            # Menampilkan grafik di Streamlit
            st.plotly_chart(fig)

            if len(df_sentiment_positive)>len(df_sentiment_negative):
                st.write('''
                        ## Mayoritas orang menyukai produk.
                        ''')
                st.write('''
                        ### Saran yang dapat diberikan dari kami:
                         - Dapat mengembangkan produk baru atau tipe baru dari sunscreen yang telah dijual
                         - Memaksimalkan penjualan kesuluruh penjuru Indonesia dikarenakan kebanyakan orang menyukai produk.
                         - Memanfaatkan exposure baik yang ada di media sosial untuk marketing dan nama baik perusahaan lebih lanjut.
                        ''')
            else:
                st.write('''
                        ## Mayoritas orang tidak menyukai dengan produk.
                        ''')
                st.write('''
                        ### Saran yang dapat diberikan dari kami:
                        - Mengevaluasi kembali produk yang telah diedarkan dikarenakan sentimen terhadap produk sedang tidak baik.
                        - Dapat memperbaiki formula yang ada pada produk yang dijual.
                        - Menggencarkan proses marketing seperti iklan dan pemanfaatan endorsement atau progeram afiliasi.
                        ''')
                         
    # B. For Manual        
    else:
        # Create Form
        sentence = st.text_input('Kalimat', value='', help='silahkan tulis kalimat yang akan diprediksi sentimennya')

        df = {'full_text': sentence}

        if st.button('Predict'):
            df = pd.DataFrame([df])

            predictData(df)
            if df['Affiliate'][0] == 0:
                pred_test_sentiment = model_sentiment.predict(df['text_processed'])

                # Convert predictions from probabilities to class labels
                pred_test_sentiment_class = np.where(pred_test_sentiment >= 0.5, 1, 0)
                df['Sentiment'] = pred_test_sentiment_class
                if df['Sentiment'][0] == 1:
                    st.write('## Kalimat tergolong bukan affiliate dan memiliki sentiment Positive')
                else:
                    st.write('## Kalimat tergolong bukan affiliate dan memiliki sentiment Negative')
            else:
                st.write('## Kalimat tergolong iklan atau affiliate')

if __name__ == '__main__':
    run()
