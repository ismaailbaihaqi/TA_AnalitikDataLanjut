import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


# Streamlit App
st.title('Flight Price Prediction')

 # Baca file CSV
df = pd.read_csv('./Clean_Dataset.csv')
df_economy = df[df['class'] == 'Economy']

# Menambahkan navigasi di sidebar
st.write("KELOMPOK 4")
st.write("- Nayaka Wiryatama_065002200003")
st.write("- Rizky Ramadhan_065002200010")
st.write("- Ery Febrian_065002200011")
st.write("- Ismail Baihaqi_065002200025")
page = st.sidebar.radio("Pilih halaman", ["Dataset", "Perbandingan Class", "Class Vs Ticket Price","Flights Count", "Stops", "Departure and Arrival", "Source and Destination", "Duration", "Days Left", 
"Actual vs Predicted","Make Prediction"])

result = None
if page == "Dataset":
    st.header("Halaman Dataset")
    st.write(df)

elif page == "Perbandingan Class":
    st.title('Classes of Different Airlines')
    st.write('Kelas penerbangan atau kelas layanan dalam maskapai penerbangan mengacu pada tingkat kenyamanan, fasilitas, dan pelayanan yang ditawarkan kepada penumpang. Berikut adalah beberapa kelas yang umumnya ditemukan dalam berbagai maskapai penerbangan:')
    st.write('- Ekonomi: Kelas ini merupakan kelas paling umum dan terjangkau. Penumpang di kelas ekonomi ditempatkan di kursi-kursi yang biasanya memiliki ruang kaki yang lebih terbatas dibandingkan dengan kelas yang lebih tinggi. Mereka menerima pelayanan makanan dan minuman dasar selama penerbangan.')
    st.write('- Bisnis: Kelas bisnis menawarkan tingkat kenyamanan yang lebih tinggi dengan kursi yang dapat direbahkan, ruang kaki yang lebih luas, dan layanan makanan yang lebih baik. Penumpang kelas bisnis juga dapat menikmati fasilitas tambahan di bandara, seperti akses lounge.')
    # Plotting the pie chart
    fig, ax = plt.subplots()
    df['class'].value_counts().plot(kind='pie', ax=ax, autopct='%.2f', colors=sns.color_palette('pastel'), textprops={'color':'black'})
    ax.set_title('Classes of Different Airlines', fontsize=15)
    ax.legend(['Economy', 'Business'])

    # Display the plot in Streamlit
    st.pyplot(fig)
    

elif page == "Class Vs Ticket Price":
    st.title('Class Vs Ticket Price')
    st.write('Ketika membahas tentang kelas harga tiket pesawat, kita dapat merinci berbagai tipe tarif atau kelas harga yang ditawarkan oleh maskapai penerbangan. Berikut adalah beberapa kelas harga yang umumnya ditemui:')
    st.write('- Ekonomi: Kelas harga ini umumnya merupakan opsi yang lebih terjangkau, tetapi penumpang mungkin harus mengorbankan beberapa fasilitas, seperti hak untuk memilih tempat duduk atau membawa bagasi kabin tambahan. Ekonomi dasar sering kali memberikan pengalaman penerbangan yang lebih sederhana.')
    st.write('- Bisnis: Tiket kelas bisnis dan first class biasanya termasuk dalam kategori harga yang lebih tinggi. Namun, penumpang mendapatkan keuntungan dari fasilitas yang lebih mewah, kursi yang lebih nyaman, pilihan makanan yang lebih baik, dan layanan tambahan.')
    # Plotting the boxplot
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(x='class', y='price', data=df, palette='hls', ax=ax)
    ax.set_title('Class Vs Ticket Price', fontsize=15)
    ax.set_xlabel('Class', fontsize=15)
    ax.set_ylabel('Price', fontsize=15)

    # Display the plot in Streamlit
    st.pyplot(fig)

elif page == "Flights Count":
    st.title('Flights Count of Different Airlines (Economy)')
    st.write('Jumlah penerbangan dan harga tiket kelas ekonomi dari berbagai maskapai penerbangan memberikan gambaran tentang berbagai pilihan yang tersedia untuk penumpang. Berikut adalah deskripsi mengenai jumlah penerbangan dan harga tiket kelas ekonomi dari beberapa maskapai:')

    # Plotting the count plot
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(data=df_economy, x='airline', order=df_economy['airline'].value_counts().index[::-1], ax=ax)
    ax.set_title('Flights Count of Different Airlines (Economy)', fontsize=15)
    ax.set_xlabel('Airline', fontsize=15)
    ax.set_ylabel('Count', fontsize=15)

    # Display the plot in Streamlit
    st.pyplot(fig)

elif page == "Class Vs Ticket Price (Economy)":
    st.title('Airlines Vs Price (Economy)')

    # Plotting the boxplot
    fig, ax = plt.subplots(figsize=(15, 5))
    sns.boxplot(data=df_economy, x='airline', y='price', ax=ax)
    ax.set_title('Airlines Vs Price (Economy)', fontsize=15)
    ax.set_xlabel('Airline', fontsize=15)
    ax.set_ylabel('Price', fontsize=15)

    # Display the plot in Streamlit
    st.pyplot(fig)

elif page == "Economy":
    st.title('Flights Count and Airlines Vs Price (Economy)')

    # Plotting the count plot
    st.subheader('Flights Count of Different Airlines (Economy)')
    fig_count, ax_count = plt.subplots(figsize=(8, 5))
    sns.countplot(data=df_economy, x='airline', order=df_economy['airline'].value_counts().index[::-1], ax=ax_count)
    ax_count.set_title('Flights Count of Different Airlines (Economy)', fontsize=15)
    ax_count.set_xlabel('Airline', fontsize=15)
    ax_count.set_ylabel('Count', fontsize=15)
    st.pyplot(fig_count)

    # Plotting the boxplot
    st.subheader('Airlines Vs Price (Economy)')
    fig_boxplot, ax_boxplot = plt.subplots(figsize=(15, 5))
    sns.boxplot(data=df_economy, x='airline', y='price', ax=ax_boxplot)
    ax_boxplot.set_title('Airlines Vs Price (Economy)', fontsize=15)
    ax_boxplot.set_xlabel('Airline', fontsize=15)
    ax_boxplot.set_ylabel('Price', fontsize=15)
    st.pyplot(fig_boxplot)

elif page == "Business":
    st.title('Flights Count and Airlines Vs Price (Business)')

    # Plotting the count plot
    st.subheader('Flights Count of Different Airlines (Business)')
    fig_count, ax_count = plt.subplots(figsize=(8, 5))
    sns.countplot(data=df_economy, x='airline', order=df_economy['airline'].value_counts().index[::-1], ax=ax_count)
    ax_count.set_title('Flights Count of Different Airlines (Business)', fontsize=15)
    ax_count.set_xlabel('Airline', fontsize=15)
    ax_count.set_ylabel('Count', fontsize=15)
    st.pyplot(fig_count)

    # Plotting the boxplot
    st.subheader('Airlines Vs Price (Economy)')
    fig_boxplot, ax_boxplot = plt.subplots(figsize=(15, 5))
    sns.boxplot(data=df_economy, x='airline', y='price', ax=ax_boxplot)
    ax_boxplot.set_title('Airlines Vs Price (Economy)', fontsize=15)
    ax_boxplot.set_xlabel('Airline', fontsize=15)
    ax_boxplot.set_ylabel('Price', fontsize=15)
    st.pyplot(fig_boxplot)

elif page == "Stops":
    st.title('Stops Vs Ticket Price')
    st.write('Perbandingan antara jumlah persinggahan (stops) dalam suatu penerbangan dan harga tiket merupakan pertimbangan penting bagi penumpang yang merencanakan perjalanan mereka. Berikut adalah deskripsi tentang pengaruh jumlah persinggahan terhadap harga tiket penerbangan:')
    st.write('- 0 stop: Penerbangan langsung atau non-stop adalah pilihan terbaik bagi penumpang yang ingin mencapai tujuan mereka tanpa persinggahan. Meskipun tiket untuk penerbangan langsung mungkin cenderung lebih mahal, penumpang dapat menikmati kenyamanan dan waktu tempuh yang lebih singkat.')
    st.write('- 1 stop: Penerbangan dengan satu persinggahan (direct flight) melibatkan perhentian teknis tanpa perlu berganti pesawat. Harga tiket untuk penerbangan dengan satu persinggahan biasanya lebih terjangkau daripada non-stop, dan penumpang memiliki kesempatan untuk istirahat sejenak di bandara persinggahan.')
    st.write('- 2 or more stop: Penerbangan dengan lebih dari satu persinggahan (connecting flight) dapat menjadi pilihan yang lebih ekonomis, namun, penumpang harus mempertimbangkan waktu tempuh yang lebih lama dan kemungkinan ketidaknyamanan akibat perpindahan pesawat.')

    # Plotting the boxplot
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(x='stops', y='price', data=df, palette='hls', ax=ax)
    ax.set_title('Stops Vs Ticket Price', fontsize=15)
    ax.set_xlabel('Stops', fontsize=15)
    ax.set_ylabel('Price', fontsize=15)

    # Display the plot in Streamlit
    st.pyplot(fig)

elif page == "Departure and Arrival":
    st.title('Departure and Arrival Time Vs Ticket Price')
    st.write('Perbandingan antara waktu keberangkatan (departure) dan waktu kedatangan (arrival) terhadap harga tiket pesawat merupakan faktor penting yang memengaruhi keputusan penumpang dalam merencanakan perjalanan mereka. Berikut adalah deskripsi mengenai hubungan antara waktu keberangkatan dan kedatangan dengan harga tiket pesawat')
    st.write('- Keberangkatan Pagi atau Malam: Waktu keberangkatan pagi atau malam hari sering kali memiliki pengaruh pada harga tiket. Penerbangan pada waktu-waktu ini dapat memiliki harga yang lebih rendah karena cenderung kurang diminati dibandingkan dengan penerbangan pada jam sibuk di siang hari.')

    # Plotting the boxplot for Departure Time
    st.subheader('Departure Time Vs Ticket Price')
    fig_departure, ax_departure = plt.subplots(figsize=(10, 5))
    sns.boxplot(x='departure_time', y='price', data=df, palette='hls', ax=ax_departure)
    ax_departure.set_title('Departure Time Vs Ticket Price', fontsize=15)
    ax_departure.set_xlabel('Departure Time', fontsize=15)
    ax_departure.set_ylabel('Price', fontsize=15)
    st.pyplot(fig_departure)

    # Plotting the boxplot for Arrival Time
    st.subheader('Arrival Time Vs Ticket Price')
    fig_arrival, ax_arrival = plt.subplots(figsize=(10, 5))
    sns.boxplot(x='arrival_time', y='price', data=df, palette='hls', ax=ax_arrival)
    ax_arrival.set_title('Arrival Time Vs Ticket Price', fontsize=15)
    ax_arrival.set_xlabel('Arrival Time', fontsize=15)
    ax_arrival.set_ylabel('Price', fontsize=15)
    st.pyplot(fig_arrival)

elif page == "Source and Destination":
    st.title('Source and Destination City Vs Ticket Price')
    st.write('Perbandingan antara kota keberangkatan (source city) dan kota tujuan (destination city) dengan harga tiket pesawat adalah pertimbangan penting dalam merencanakan perjalanan.')
    st.write('')

    # Plotting the boxplot for Source City
    st.subheader('Source City Vs Ticket Price')
    fig_source, ax_source = plt.subplots(figsize=(10, 5))
    sns.boxplot(x='source_city', y='price', data=df, palette='hls', ax=ax_source)
    ax_source.set_title('Source City Vs Ticket Price', fontsize=15)
    ax_source.set_xlabel('Source City', fontsize=15)
    ax_source.set_ylabel('Price', fontsize=15)
    st.pyplot(fig_source)

    # Plotting the boxplot for Destination City
    st.subheader('Destination City Vs Ticket Price')
    fig_destination, ax_destination = plt.subplots(figsize=(10, 5))
    sns.boxplot(x='destination_city', y='price', data=df, palette='hls', ax=ax_destination)
    ax_destination.set_title('Destination City Vs Ticket Price', fontsize=15)
    ax_destination.set_xlabel('Destination City', fontsize=15)
    ax_destination.set_ylabel('Price', fontsize=15)
    st.pyplot(fig_destination)

elif page == "Duration":
    st.title('Duration Vs Price')
    st.write('Perbandingan antara durasi perjalanan dan harga tiket pesawat adalah aspek penting dalam pemilihan opsi perjalanan. Berikut adalah penjelasan mengenai hubungan antara durasi perjalanan dan harga tiket')
    st.write('- Durasi Penerbangan: Durasi perjalanan atau waktu yang dibutuhkan untuk mencapai tujuan merupakan faktor utama yang mempengaruhi harga tiket pesawat. Penerbangan dengan durasi yang lebih singkat seringkali memiliki harga yang lebih tinggi, terutama untuk rute-rute jarak jauh atau penerbangan langsung.')
    st.write('- Jarak Tempuh: Jarak tempuh antara kota keberangkatan dan tujuan memainkan peran penting. Destinasi jarak jauh cenderung memiliki harga tiket yang lebih tinggi, sementara destinasi yang lebih dekat umumnya memiliki harga yang lebih terjangkau.')
    st.write('Penumpang disarankan untuk mempertimbangkan dengan cermat faktor-faktor di atas dan menyesuaikan pilihan perjalanan mereka sesuai dengan kebutuhan, kenyamanan, dan anggaran yang tersedia.')
    # Plotting the regression plot
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.regplot(x='duration', y='price', data=df, line_kws={'color': 'blue'}, ax=ax)
    ax.set_title('Duration Vs Price', fontsize=20)
    ax.set_xlabel('Duration', fontsize=15)
    ax.set_ylabel('Price', fontsize=15)

    # Display the plot in Streamlit
    st.pyplot(fig)

elif page == "Days Left":
    st.title('Days Left Vs Price')
    st.write('Perbandingan antara jumlah hari yang tersisa sebelum tanggal keberangkatan (days left) dan harga tiket pesawat adalah faktor penting yang memengaruhi biaya perjalanan. Berikut adalah penjelasan mengenai hubungan antara jumlah hari yang tersisa dan harga tiket')
    st.write('- Waktu Pemesanan Awal: Harga tiket pesawat sering kali lebih terjangkau jika penumpang memesan tiket mereka jauh-jauh hari sebelum tanggal keberangkatan. Pemesanan awal dapat memberikan keuntungan diskon atau tarif yang lebih rendah.')
    st.write('- Last Minute Booking: Sebaliknya, jika penumpang memesan tiket dalam waktu mendekati keberangkatan, harga tiket cenderung lebih tinggi. Last-minute booking dapat mengakibatkan biaya yang lebih mahal karena maskapai penerbangan dapat memanfaatkan permintaan mendesak.')
    st.write('- Pemantauan Harga: Memonitor perubahan harga tiket secara rutin dapat membantu penumpang menangkap penawaran terbaik. Beberapa situs web dan aplikasi menyediakan layanan pemantauan harga untuk membantu penumpang memilih waktu yang tepat untuk membeli tiket.')
    # Plotting the line plot
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=df, x='days_left', y='price', color='blue', ax=ax)
    ax.set_title('Days Left Vs Price', fontsize=20)
    ax.set_xlabel('Days Left', fontsize=15)
    ax.set_ylabel('Price', fontsize=15)

    # Display the plot in Streamlit
    st.pyplot(fig)

elif page == "Actual vs Predicted":
    st.title('Actual Price Vs Predicted Price')
    st.write('Perbandingan antara harga aktual (Actual Price) dan harga yang diprediksi (Predicted Price) merujuk pada perbedaan antara harga sebenarnya yang dikenakan atau dibayar dan perkiraan harga yang mungkin telah dihitung sebelumnya. Berikut adalah penjelasan mengenai konsep Actual Price dan Predicted Price')
    st.write('- Actual Price(Harga Sebenarnya): Harga aktual adalah harga yang sebenarnya dikenakan atau dibayar oleh konsumen atau pihak yang melakukan pembelian. Ini mencakup seluruh biaya atau nilai transaksi yang terjadi pada suatu produk atau layanan pada saat pembelian dilakukan.')
    st.write('- Predicted Price(Harga yang diprediksi): Predicted Price adalah estimasi atau perhitungan mengenai harga yang mungkin diharapkan atau diperkirakan sebelumnya. Hal ini dapat melibatkan analisis data, model matematis, atau faktor-faktor lain yang memungkinkan untuk memprediksi harga suatu produk atau layanan dalam kondisi tertentu.')
    st.write('Pentingnya Perbandingan antara actual dan predicted price bermanfaat untuk mengevaluasi seberapa baik prediksi atau perkiraan tersebut sesuai dengan kenyataan. Jika perbedaan antara harga sebenarnya dan prediksi cukup besar, hal ini dapat menunjukkan adanya ketidakakuratan dalam model atau metode prediksi yang digunakan.')

    # Example: Train a simple linear regression model
    X = df[['duration', 'days_left']]
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Create a DataFrame for result
    result = pd.DataFrame({'Price_actual': y_test, 'Price_pred': y_pred})

    # Ensure 'Price_actual' and 'Price_pred' columns are numeric
    result['Price_actual'] = pd.to_numeric(result['Price_actual'])
    result['Price_pred'] = pd.to_numeric(result['Price_pred'])

    # Plotting the regression plot
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.regplot(x='Price_actual', y='Price_pred', data=result, ax=ax)
    ax.set_title('Actual Price Vs Predicted Price', fontsize=20)
    ax.set_xlabel('Actual Price', fontsize=15)
    ax.set_ylabel('Predicted Price', fontsize=15)

    # Display the plot in Streamlit
    st.pyplot(fig)

elif page == "Make Prediction":
    st.title('Make Flight Price Prediction')
    model = RandomForestRegressor()  # Use the same model for training and prediction
    joblib.dump(model, 'random_forest_model.pkl')
    # User input for prediction
    st.subheader("Input Features for Prediction")
    duration = st.number_input("Duration (hours)", min_value=0, max_value=24, value=1)
    days_left = st.number_input("Days Left for Departure", min_value=0, value=30)
    # Add more input features as needed for your model

    # Prepare the input data for prediction
    input_data = pd.DataFrame({'duration': [duration], 'days_left': [days_left]})
    # Add more input features as needed for your model

    # Load the saved model (assuming you saved it during training)
    model_path = './random_forest_model.pkl'
    try:
        model = joblib.load(model_path)
        if not hasattr(model, 'fit'):
            raise ValueError("The loaded model does not have a 'fit' method.")
         # Fit the model if it's not already fitted
        if not hasattr(model, 'estimators_') or len(model.estimators_) == 0:
        # Example: fit the model with your training data
            X_train, y_train = df[['duration', 'days_left']], df['price']
            model.fit(X_train, y_train)
    # Ensure the model is fitted before making predictions
        if hasattr(model, 'predict'):
            prediction = model.predict(input_data)
            st.subheader('Predicted Price using random forest regression:')
            st.write(prediction[0])  # Assuming the prediction is a single value
        else:
            st.warning("Model is not fitted.")
    except Exception as e:
        st.error(f"Error loading or fitting the model: {str(e)}")
#if hasattr(model, 'predict'):
#    prediction = model.predict(input_data)
#    st.subheader('Predicted Price:')
#    st.write(prediction[0])
#else:
#    st.warning("Model is not fitted.")
