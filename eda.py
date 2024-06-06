import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image

def run():
    # Show dataframe
    # st.title('Data Overview')
    df = pd.read_csv('sunscreen_labeled_data.csv')
    # st.dataframe(df.head())

    st.title('Exploratory Data Analysis')
    plot_selection = st.selectbox(label='Choose', 
                                  options=['Affiliate and non Affiliate', 
                                           'Netizen Sentiment from Tweets',
                                           'Non Affiliate Sentiments', 
                                           'WordClouds'])
    
    # Plot 1
    def plot_1():
        st.write('#### Bar Chart for Affiliate and non Affiliate')

        aff = len(df[(df['Affiliate']==1)])
        nonaff = len(df[(df['Affiliate']==0)])
        # Visualisasi pie perbandingan non affiliate tweets dengan sentiment netizen
        fig_1=plt.figure(figsize=(5, 5))
        plt.pie([aff,nonaff], labels=['Affiliate', 'Non Affiliate'], autopct='%1.2f%%', colors=['yellow', 'violet'])
        plt.title('Perbandingan Jumlah Tweets Berdasarkan Afiliasi')
        plt.show()

        st.pyplot(fig_1)
        st.write('''
                Data training berjumlah 5210 data sentimen masyarakat dari 9 merk sunscreen yang berbeda. 77.04% merupakan data asli sentimen customer dan 22,96% merupakan data iklan.

                Hal ini menunjukkan bahwa:

                - Mayoritas Sentimen Asli: Sebagian besar data berasal dari sentimen pelanggan asli, memberikan gambaran akurat tentang opini konsumen.
                - Pengaruh Iklan Signifikan: 22,96% data adalah iklan, menandakan perusahaan aktif mempromosikan produk mereka di platform.
                - Pemisahan Data Penting: Penting untuk membedakan antara sentimen asli dan iklan dalam analisis agar hasilnya akurat.
                - Perlunya Kualitas Data: Metode pembersihan dan pemrosesan data yang efektif dibutuhkan untuk mencerminkan pandangan konsumen yang sebenarnya.
                - Wawasan Mendalam: Fokus pada sentimen pelanggan asli memberikan feedback berharga untuk perbaikan produk dan strategi pemasaran.
                ''')
        st.markdown('---')
    
    # Plot 2
    def plot_2():
        st.write('#### Bar Chart of Netizen Sentiment')

        df.drop(df[df.Sentiment == 3].index, inplace=True)
        labels=["Positive","Neutral","Negative"]
        values = df['Sentiment'].value_counts()
        fig_2=plt.figure(figsize = (5,5))
        plt.bar(labels, values, color=['yellow', 'navy','violet'])
        plt.xlabel('Jenis Tweets')
        plt.ylabel('Total')
        plt.title('Perbandingan Sentiment Tweets Netizen')
        plt.show()
        st.pyplot(fig_2)
        st.write('''
                 - Mayoritas Sentimen Positif: Sebagian besar sentimen yang dianalisis adalah positif, menunjukkan bahwa banyak konsumen memiliki pengalaman baik dan puas dengan produk sunscreen yang mereka gunakan. Ini merupakan indikator positif bagi merek-merek sunscreen, menandakan keberhasilan produk mereka dalam memenuhi harapan dan kebutuhan konsumen.
                 - Sentimen Negatif Cukup Tinggi: Sentimen negatif menempati posisi kedua terbanyak. Ini menunjukkan adanya sejumlah konsumen yang tidak puas dengan produk yang mereka gunakan, mengindikasikan area yang perlu diperbaiki oleh perusahaan. Umpan balik negatif ini penting untuk diidentifikasi dan ditindaklanjuti agar merek dapat meningkatkan kualitas produk dan layanan mereka.
                 - Sentimen Netral Terendah: Sentimen netral berada di posisi terakhir, menandakan bahwa hanya sedikit konsumen yang memiliki opini yang tidak terlalu kuat atau tidak berpengaruh baik secara positif maupun negatif. Ini bisa berarti bahwa mayoritas konsumen memiliki pendapat yang jelas mengenai produk, baik itu positif atau negatif.
                 ''')
        st.markdown('---')

    # Plot 3
    def plot_3():
        nonaff_pos = len(df[(df['Affiliate']==0)&(df['Sentiment']==1)])
        nonaff_neg = len(df[(df['Affiliate']==0)&(df['Sentiment']==0)])
        nonaff_neu = len(df[(df['Affiliate']==0)&(df['Sentiment']==2)])
        # Visualisasi pie perbandingan non affiliate tweets dengan sentiment netizen
        fig_3=plt.figure(figsize=(5, 5))
        plt.pie([nonaff_pos,nonaff_neu,nonaff_neg,], labels=['Positive', 'Negative','Neutral'], autopct='%1.1f%%', colors=['yellow', 'violet', 'navy'])
        plt.title('Non Affiliate Sentiments')
        plt.show()
        st.pyplot(fig_3)
        st.write('''
                - Kepuasan Konsumen: Mayoritas sentimen positif menunjukkan tingkat kepuasan yang tinggi di kalangan konsumen non-iklan.
                - Area untuk Perbaikan: Adanya sentimen negatif yang signifikan mengidentifikasi area yang membutuhkan perbaikan.
                - Opini Konsumen: Rendahnya sentimen netral menunjukkan bahwa konsumen memiliki opini yang kuat mengenai produk sunscreen.
                 ''')
        st.markdown('---')
    
    # Plot 4
    def plot_4():
        image_url = 'all.png'
        st.image(image_url,use_column_width ="always")
        st.write('''
                1. Kulit: Kata "kulit" muncul sebagai salah satu kata paling dominan, menunjukkan bahwa banyak pengguna berbicara tentang efek sunscreen pada kulit mereka. Ini mencakup bagaimana produk mempengaruhi kondisi kulit, seperti perlindungan dari sinar UV dan reaksi kulit terhadap produk tersebut.
                2. Pakai: Kata "pakai" sering digunakan, menandakan bahwa banyak pengguna membahas penggunaan sunscreen dalam rutinitas sehari-hari mereka. Ini juga mencakup frekuensi penggunaan dan cara aplikasi produk.
                3. Beli: Kata "beli" sering muncul, mengindikasikan banyak diskusi tentang keputusan pembelian, tempat pembelian (seperti toko atau online), dan pengalaman pembelian produk sunscreen.
                4. Cocok: Kata "cocok" menunjukkan bahwa banyak pengguna memberikan feedback tentang kecocokan produk dengan jenis kulit mereka, apakah produk tersebut sesuai dengan kebutuhan kulit mereka atau tidak.
                5. Muka: Kata "muka" sering disebut, mengindikasikan banyak pengguna membicarakan efek produk pada wajah mereka, seperti perlindungan, tekstur, dan kenyamanan produk saat digunakan di area wajah.
                6. Banget: Kata "banget" sering digunakan untuk menekankan pendapat atau perasaan mereka terhadap produk. Misalnya, "sangat bagus" atau "sangat buruk."
                7. Harga: Kata "harga" menandakan banyak diskusi mengenai biaya produk sunscreen. Pengguna mungkin membicarakan apakah produk tersebut memiliki nilai yang sesuai dengan harganya atau membandingkan harga antar merek.
                8. Bagus: Kata "bagus" menunjukkan banyak ulasan positif, di mana pengguna mengekspresikan kepuasan mereka terhadap kualitas dan efektivitas produk.
                9. Udah: Kata "udah" menunjukkan bahwa pengguna membicarakan pengalaman mereka setelah menggunakan produk selama beberapa waktu, memberikan pendapat berdasarkan hasil yang mereka rasakan.
                10. Bikin: Kata "bikin" digunakan dalam konteks produk yang memberikan hasil tertentu, seperti "bikin kulit jadi lebih lembut" atau "bikin muka jadi berminyak."
                11. Coba: Kata "coba" menandakan bahwa banyak pengguna membicarakan tentang mencoba produk baru atau membandingkan dengan produk yang sudah pernah mereka coba sebelumnya.
                12. Shopee: Kata "Shopee" menunjukkan bahwa banyak pengguna membeli atau membicarakan pembelian produk sunscreen melalui platform e-commerce Shopee.
                13. Rp: Kata "Rp" (Rupiah) sering muncul dalam konteks diskusi harga produk, mengindikasikan bahwa harga produk adalah faktor penting dalam ulasan dan keputusan pembelian.
                14. Pa: Kata "pa" mungkin merupakan potongan kata dari "pakai" atau bisa juga singkatan dari kata lain yang digunakan oleh pengguna.
                 
                Word cloud ini menunjukkan fokus utama pengguna pada aspek penggunaan, kecocokan, harga, dan pengalaman pembelian produk sunscreen. Sebagian besar diskusi berkisar pada bagaimana produk berinteraksi dengan kulit mereka, baik positif maupun negatif, serta evaluasi harga dan nilai produk. Platform e-commerce seperti Shopee juga menjadi topik penting dalam ulasan produk sunscreen. 
                Word cloud ini memberikan wawasan berharga tentang topik dan kata kunci utama yang sering dibicarakan oleh pengguna dalam ulasan mereka, membantu perusahaan memahami fokus dan perhatian konsumen.
                ''')
        st.markdown('---')


    if plot_selection == "Affiliate and non Affiliate":
        plot_1()
    elif plot_selection == "Netizen Sentiment from Tweets":
        plot_2()
    elif plot_selection == "Non Affiliate Sentiments":
        plot_3()
    elif plot_selection == "WordClouds":
        plot_4()

if __name__ == '__main__':
    run()