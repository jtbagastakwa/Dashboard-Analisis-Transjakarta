import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn2
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import plotly.express as px
from matplotlib.ticker import MaxNLocator

# --- BAGIAN 1: KONFIGURASI TAMPILAN DAN FUNGSI BANTU ---

def set_page_style():
    """
    Fungsi untuk menerapkan style modern dengan efek bayangan pada setiap grafik.
    """
    st.markdown("""
    <style>
    .stApp {
        background-color: #F0F2F6; /* Warna latar belakang abu-abu muda */
        font-family: 'Segoe UI', sans-serif;
        font-size: 1.1rem; /* Ukuran font dasar diperbesar */
    }
    /* Hapus style dari kontainer utama agar tidak seperti kartu */
    .main .block-container {
        background: transparent;
        box-shadow: none;
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    
    /* Style untuk SETIAP GRAFIK sebagai kartu dengan bayangan */
    .stPlot, [data-testid="stImage"], .stDataFrame, .stMap, .stPlotlyChart, .stGraphVizChart {
        background: #FFFFFF;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.07);
        margin-bottom: 2rem; /* Jarak antar grafik */
        margin-top: 1rem;
    }

    /* Style untuk tabs agar bersih */
    .stTabs [data-baseweb="tab-list"] {
        background: transparent;
        border-bottom: 1px solid #ddd;
        font-size: 1.2rem; /* Ukuran font judul tab */
    }
    .stTabs [data-baseweb="tab-panel"] > div {
        background: transparent;
        padding: 0;
        box-shadow: none;
    }
    h1 {
        font-size: 2.8rem !important;
        color: #1A1A1A;
    }
    h2 {
       font-size: 2.2rem !important;
       color: #1A1A1A;
    }
    h3 {
       font-size: 1.75rem !important;
       color: #333333;
    }
    .stAlert {
        border-radius: 10px;
    }
    
    /* FIX: Style the selectbox */
    .stSelectbox div[data-baseweb="select"] > div {
        background-color: #FFFFFF; /* Warna abu-abu untuk filter */
        border-radius: 10px;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.07);
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_data
def load_data():
    """
    Memuat data dari file Excel. Menggunakan caching Streamlit untuk performa.
    """
    try:
        halte_df = pd.read_excel('Data Halte Transjakarta 2025_modified.xlsx', sheet_name='Sheet1')
        bus_penumpang_df = pd.read_excel('Data Jumlah Bus yang Beroperasi dan Jumlah Penumpang Layanan Transjakarta 2024_modified.xlsx', sheet_name='Sheet1')
        rute_df = pd.read_excel('Data Rute Jalur Transjakarta 2024_modified.xlsx', sheet_name='Sheet1')

        # FIX: Gabungkan kategori 'BRT' dan 'Bus Rapid Transit'
        bus_penumpang_df['jenis_layanan'] = bus_penumpang_df['jenis_layanan'].replace('Bus Rapid Transit', 'BRT')
        rute_df['kategori'] = rute_df['kategori'].replace('Bus Rapid Transit', 'BRT')

        # Persiapan data untuk peta geospasial
        halte_df['koordinat_x'] = pd.to_numeric(halte_df['koordinat_x'], errors='coerce')
        halte_df['koordinat_y'] = pd.to_numeric(halte_df['koordinat_y'], errors='coerce')
        halte_df.dropna(subset=['koordinat_x', 'koordinat_y'], inplace=True)
        
        halte_df['lat'] = halte_df['koordinat_x'] / 1000000
        halte_df['lon'] = halte_df['koordinat_y'].apply(lambda y: y / 100000 if y < 100000000 else y / 1000000)

        halte_df = halte_df[
            (halte_df['lat'] > -6.5) & (halte_df['lat'] < -6.0) &
            (halte_df['lon'] > 106.6) & (halte_df['lon'] < 107.0)
        ].copy()
        
        return halte_df, bus_penumpang_df, rute_df
    except FileNotFoundError as e:
        st.error(f"Gagal memuat file: {e.filename}. Pastikan file Excel asli ada di direktori yang sama.")
        return None, None, None
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses data: {e}")
        return None, None, None

# --- BAGIAN 2: LOGIKA CHATBOT DENGAN LANGCHAIN ---

def get_response(user_query, chat_history, model, data_context):
    """
    Mendapatkan respons dari model AI menggunakan LangChain.
    """
    template = """
    Anda adalah "Analis AI", seorang ahli analis data transportasi publik yang berspesialisasi dalam sistem Transjakarta.
    Anda sedang berdiskusi dengan seorang pengguna yang melihat dashboard interaktif.
    Berdasarkan konteks data yang telah dianalisis dan riwayat percakapan, berikan jawaban yang mendalam dan rekomendasi yang dapat ditindaklanjuti.

    Konteks Hasil Analisis Data:
    {data_context}

    Riwayat Percakapan:
    {chat_history}

    Pertanyaan Pengguna:
    {user_question}
    """
    
    prompt = PromptTemplate(
        input_variables=["data_context", "chat_history", "user_question"],
        template=template
    )
    
    chain = prompt | model
    
    response = chain.invoke({
        "data_context": data_context,
        "chat_history": chat_history,
        "user_question": user_query
    })
    
    return response.content

# --- BAGIAN 3: UTAMA APLIKASI STREAMLIT ---

# Konfigurasi halaman
st.set_page_config(page_title="Dashboard Analisis Transjakarta", page_icon="💡", layout="wide")
set_page_style() # Memanggil fungsi style baru

# Inisialisasi Model AI
try:
    google_api_key = st.secrets["GOOGLE_API_KEY"]
    chat_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=google_api_key, temperature=0.7)
    st.session_state.api_configured = True
except (FileNotFoundError, KeyError):
    st.error("Konfigurasi GOOGLE_API_KEY tidak ditemukan di Streamlit Secrets. Chatbot tidak akan berfungsi.")
    chat_model = None
    st.session_state.api_configured = False

# Judul Aplikasi
st.title("🚌 Dashboard Analisis Interaktif Transjakarta")
st.markdown("Aplikasi ini memvisualisasikan data operasional Transjakarta. Gunakan tab untuk navigasi dan chatbot untuk diskusi.")

# Memuat Data
halte_df, bus_penumpang_df, rute_df = load_data()

if halte_df is not None:
    # Pengaturan Tema dan Ukuran Font untuk Grafik Matplotlib
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.labelsize'] = 14  # Ukuran font label sumbu X dan Y
    plt.rcParams['xtick.labelsize'] = 12 # Ukuran font tick sumbu X
    plt.rcParams['ytick.labelsize'] = 12 # Ukuran font tick sumbu Y
    plt.rcParams['axes.titlesize'] = 18  # Ukuran font judul grafik
    plt.rcParams['legend.fontsize'] = 14 # Ukuran font legenda

    # Navigasi Menggunakan Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Distribusi Halte", "Penumpang & Bus", "Jaringan Rute", "Analisis Hub Halte", "Sebaran Halte", "Tren & Korelasi"
    ])

    with tab1:
        st.header("Analisis Distribusi Geografis Halte")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Jumlah Halte per Wilayah")
            fig1, ax1 = plt.subplots()
            halte_by_wilayah = halte_df['wilayah'].value_counts().sort_values(ascending=False)
            sns.barplot(x=halte_by_wilayah.values, y=halte_by_wilayah.index, palette='Blues_r', ax=ax1)
            ax1.set_xlabel('Jumlah Halte'); ax1.set_ylabel('')
            sns.despine(left=True, bottom=True); st.pyplot(fig1)
        with col2:
            st.subheader("Top 10 Jumlah Halte Terbanyak per Kecamatan")
            fig2, ax2 = plt.subplots()
            halte_by_kecamatan = halte_df['kecamatan'].value_counts().nlargest(10)
            sns.barplot(y=halte_by_kecamatan.index, x=halte_by_kecamatan.values, palette='Greens_r', ax=ax2)
            ax2.set_xlabel('Jumlah Halte'); ax2.set_ylabel('')
            sns.despine(left=True, bottom=True); st.pyplot(fig2)

    with tab2:
        st.header("Analisis Penumpang dan Armada Bus")
        penumpang_by_layanan = bus_penumpang_df.groupby('jenis_layanan')['jumlah_penumpang'].sum().sort_values(ascending=False)
        bus_by_layanan = bus_penumpang_df.groupby('jenis_layanan')['jumlah_bus'].sum().sort_values(ascending=False)
        ratio_df = pd.DataFrame({'Total Penumpang': penumpang_by_layanan, 'Total Bus': bus_by_layanan})
        ratio_df['Penumpang per Bus'] = ratio_df['Total Penumpang'] / ratio_df['Total Bus']
        ratio_df = ratio_df.sort_values(by='Penumpang per Bus', ascending=False)

        st.subheader("Rasio Penumpang per Bus (Efisiensi & Kepadatan)")
        r_col1, r_col2, r_col3 = st.columns([0.2, 1.5, 0.2])
        with r_col2:
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            sns.barplot(x=ratio_df['Penumpang per Bus'], y=ratio_df.index, palette='rocket_r', ax=ax3)
            ax3.set_xlabel('Rata-rata Penumpang Dilayani per Bus'); ax3.set_ylabel('')
            sns.despine(); st.pyplot(fig3)
        
        st.info("**Interpretasi:** Grafik ini menunjukkan efisiensi layanan. Bar yang lebih panjang berarti setiap bus melayani lebih banyak penumpang, menandakan efisiensi tinggi namun juga potensi kepadatan yang tinggi.", icon="💡")

    with tab3:
        st.header("Analisis Jaringan dan Rute Utama")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Top 10 Titik Keberangkatan")
            fig6, ax6 = plt.subplots()
            top_10_titik_a = rute_df['titik_a'].value_counts().nlargest(10)
            sns.barplot(y=top_10_titik_a.index, x=top_10_titik_a.values, palette='Purples_r', ax=ax6)
            ax6.set_xlabel('Jumlah Rute'); ax6.set_ylabel('')
            sns.despine(); st.pyplot(fig6)
        with col2:
            st.subheader("Top 10 Titik Tujuan")
            fig7, ax7 = plt.subplots()
            top_10_titik_b = rute_df['titik_b'].value_counts().nlargest(10)
            sns.barplot(y=top_10_titik_b.index, x=top_10_titik_b.values, palette='Oranges_r', ax=ax7)
            ax7.set_xlabel('Jumlah Rute'); ax7.set_ylabel('')
            sns.despine(); st.pyplot(fig7)

    with tab4:
        st.header("Analisis Hub Transit Utama")
        st.markdown("Hub adalah lokasi yang sering menjadi titik awal sekaligus titik akhir, menandakan perannya sebagai pusat transit.")
        
        set_a = set(rute_df['titik_a'].dropna())
        set_b = set(rute_df['titik_b'].dropna())
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("Irisan Lokasi")
            fig9, ax9 = plt.subplots()
            v = venn2([set_a, set_b], set_labels=('Keberangkatan (A)', 'Tujuan (B)'), ax=ax9)
            
            for text in v.subset_labels:
                if text:
                    text.set_fontsize(20)
            for text in v.set_labels:
                if text:
                    text.set_fontsize(16)
                    
            st.pyplot(fig9)
        with col2:
            st.subheader("Top 15 Lokasi Hub")
            intersection_locations = set_a.intersection(set_b)
            counts_a = rute_df[rute_df['titik_a'].isin(intersection_locations)]['titik_a'].value_counts()
            counts_b = rute_df[rute_df['titik_b'].isin(intersection_locations)]['titik_b'].value_counts()
            total_counts = counts_a.add(counts_b, fill_value=0).sort_values(ascending=False).nlargest(15)
            fig10, ax10 = plt.subplots(figsize=(10, 8))
            sns.barplot(y=total_counts.index, x=total_counts.values, palette='viridis', ax=ax10)
            ax10.set_xlabel('Total Frekuensi sebagai Titik Awal & Akhir'); ax10.set_ylabel('')
            sns.despine(); st.pyplot(fig10)
        
        st.markdown("---")
        st.subheader("Tipologi Hub Utama")
        st.markdown("Analisis ini mengklasifikasikan hub berdasarkan fungsinya: sebagai titik awal (Terminal), titik akhir (Tujuan), atau keduanya (Transit).")
        
        try:
            in_degree = rute_df['titik_b'].value_counts()
            out_degree = rute_df['titik_a'].value_counts()
            
            hub_analysis_df = pd.DataFrame({
                'Rute_Masuk': in_degree,
                'Rute_Keluar': out_degree
            }).fillna(0).astype(int)

            hub_analysis_df['Total_Koneksi'] = hub_analysis_df['Rute_Masuk'] + hub_analysis_df['Rute_Keluar']
            hub_analysis_df = hub_analysis_df[hub_analysis_df['Total_Koneksi'] > 5].copy()

            def get_typology(row):
                diff = row['Rute_Keluar'] - row['Rute_Masuk']
                if diff > 2:
                    return "Dominan Terminal"
                elif diff < -2:
                    return "Dominan Tujuan"
                else:
                    return "Seimbang (Transit)"
            
            if not hub_analysis_df.empty:
                hub_analysis_df['Tipologi'] = hub_analysis_df.apply(get_typology, axis=1)
                hub_analysis_df.index.name = 'Hub'
                hub_analysis_df.reset_index(inplace=True)
                
                st.dataframe(
                    hub_analysis_df[['Hub', 'Rute_Masuk', 'Rute_Keluar', 'Total_Koneksi', 'Tipologi']]
                    .sort_values(by='Total_Koneksi', ascending=False)
                )
            else:
                st.warning("Tidak ditemukan data hub yang memenuhi kriteria (minimal 5 koneksi rute).")
        except Exception as e:
            st.error(f"Terjadi kesalahan saat membuat analisis tipologi hub: {e}")

    with tab5:
        st.header("Analisis Sebaran Halte")
        st.subheader("Peta Sebaran Halte")
        st.markdown("Peta ini menunjukkan lokasi setiap halte untuk melihat cakupan layanan.")
        if not halte_df.empty:
            map_data = halte_df[['lat', 'lon']].copy().dropna()
            if not map_data.empty:
                st.map(map_data, zoom=10)
            else:
                st.warning("Tidak ada data koordinat valid untuk ditampilkan.")
        else:
            st.warning("Data halte tidak tersedia.")

    with tab6:
        st.header("Analisis Tren dan Korelasi")

        # Agregasi data per tahun
        yearly_df = bus_penumpang_df.groupby(['periode_data', 'jenis_layanan']).agg(
            jumlah_penumpang=('jumlah_penumpang', 'sum'),
            jumlah_bus=('jumlah_bus', 'mean') # Mengambil rata-rata bus per tahun
        ).reset_index()

        st.subheader("Filter Analisis Tren")
        service_options = ['Tampilkan Semua'] + yearly_df['jenis_layanan'].unique().tolist()
        selected_service = st.selectbox(
            "Pilih satu jenis layanan untuk melihat tren:",
            service_options
        )

        # Filter data based on selection for trend plots
        if selected_service == 'Tampilkan Semua':
            filtered_yearly_df = yearly_df
        else:
            filtered_yearly_df = yearly_df[yearly_df['jenis_layanan'] == selected_service]

        st.subheader("Tren Penumpang dan Armada per Layanan per Tahun")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Tren Penumpang per Layanan")
            fig_pass, ax_pass = plt.subplots(figsize=(10, 8))
            sns.lineplot(data=filtered_yearly_df, x='periode_data', y='jumlah_penumpang', hue='jenis_layanan', marker='o', ax=ax_pass)
            ax_pass.set_title("")
            ax_pass.set_xlabel("Tahun")
            ax_pass.set_ylabel("Jumlah Penumpang")
            ax_pass.xaxis.set_major_locator(MaxNLocator(integer=True)) 
            ax_pass.legend(title='Jenis Layanan', loc='upper right', fontsize='medium')
            sns.despine()
            st.pyplot(fig_pass)
            
        with col2:
            st.markdown("#### Tren Armada Bus per Layanan")
            fig_bus, ax_bus = plt.subplots(figsize=(10, 8))
            sns.lineplot(data=filtered_yearly_df, x='periode_data', y='jumlah_bus', hue='jenis_layanan', marker='o', ax=ax_bus)
            ax_bus.set_title("")
            ax_bus.set_xlabel("Tahun")
            ax_bus.set_ylabel("Rata-rata Jumlah Bus")
            ax_bus.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax_bus.legend(title='Jenis Layanan', loc='upper right', fontsize='medium')
            sns.despine()
            st.pyplot(fig_bus)

        st.markdown("---")
        
        st.subheader("Korelasi Jumlah Bus dan Penumpang per Layanan")
        st.markdown("Diagram ini menunjukkan hubungan antara penambahan armada dengan jumlah penumpang untuk semua jenis layanan.")
        
        corr_col1, corr_col2, corr_col3 = st.columns([0.3, 1, 0.3])
        with corr_col2:
            fig_corr_sns, ax_corr_sns = plt.subplots(figsize=(8, 6))
            
            # --- FIX: Prepare data with new column names for legend titles ---
            corr_plot_df = bus_penumpang_df.copy()
            corr_plot_df.rename(columns={
                'jenis_layanan': 'Jenis Layanan',
                'jumlah_penumpang': 'Jumlah Penumpang'
            }, inplace=True)

            # Draw the regression line first
            sns.regplot(
                data=corr_plot_df,
                x="jumlah_bus",
                y="Jumlah Penumpang",
                ax=ax_corr_sns,
                scatter=False, 
                color='darkred',
                line_kws={'linestyle':'--'}
            )
            
            # Overlay the scatter plot
            sns.scatterplot(
                data=corr_plot_df,
                x="jumlah_bus",
                y="Jumlah Penumpang",
                hue="Jenis Layanan",
                size="Jumlah Penumpang",
                sizes=(30, 400),
                ax=ax_corr_sns,
                palette="viridis"
            )
            
            ax_corr_sns.set_xlabel("Jumlah Bus", fontsize=10)
            ax_corr_sns.set_ylabel("Jumlah Penumpang", fontsize=10)
            ax_corr_sns.legend(bbox_to_anchor=(1.05, 1), loc=2, fontsize='medium')
            
            ax_corr_sns.grid(which='major', linestyle='-', linewidth='0.7', color='gray')
            
            plt.tight_layout()
            st.pyplot(fig_corr_sns)


    # Konteks Data untuk Chatbot
    data_context = """
    1. **Distribusi Halte**: Paling banyak di Jakarta Timur, Selatan, Pusat. Paling sedikit di Utara.
    2. **Efisiensi Layanan**: BRT punya rasio penumpang/bus tertinggi (efisien tapi padat).
    3. **Jaringan Rute**: Mayoritas rute adalah pengumpan (feeder) dan Mikrotrans.
    4. **Hub Utama**: Blok M, Pulo Gadung, Tanah Abang, Senen adalah hub utama dengan fungsi yang berbeda-beda (terminal, tujuan, transit).
    5. **Tren**: Ada sedikit fluktuasi jumlah penumpang dan bus antar tahun, yang bervariasi antar layanan.
    6. **Korelasi**: Terdapat hubungan positif antara jumlah bus dan penumpang, terutama pada layanan utama.
    """
    st.markdown("---")

    # --- Sesi Chatbot ---
    st.header("🤖 Diskusi dengan Analis AI")
    
    with st.expander("Klik di sini untuk melihat contoh pertanyaan", expanded=True):
        st.markdown("""
        **Rekomendasi & Strategi Umum:**
        - `Berikan 3 rekomendasi utama berdasarkan data ini.`
        - `Apa masalah terbesar yang bisa diidentifikasi dari data ini dan bagaimana solusinya?`
        
        **Pertanyaan Spesifik:**
        - `Layanan mana yang paling padat? Apa yang bisa dilakukan untuk mengatasinya?`
        - `Dari peta sebaran halte, area mana yang tampaknya kurang terlayani?`
        - `Apa fungsi utama dari hub Kampung Rambutan berdasarkan tipologinya?`
        - `Apakah penambahan bus selalu efektif menaikkan jumlah penumpang? Jelaskan dari grafik korelasi.`
        - `Bandingkan tren penumpang antara layanan BRT dan Mikrotrans per tahun.`
        """)

    if not st.session_state.api_configured:
        st.warning("Chatbot non-aktif. Mohon konfigurasikan GOOGLE_API_KEY Anda di pengaturan secrets Streamlit.")
    else:
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [
                AIMessage(content="Halo! Saya adalah Analis AI Anda. Silakan ajukan pertanyaan mengenai data Transjakarta ini untuk mendapatkan insight atau rekomendasi. Anda bisa menggunakan contoh di atas sebagai inspirasi."),
            ]

        for message in st.session_state.chat_history:
            if isinstance(message, AIMessage):
                with st.chat_message("AI"):
                    st.write(message.content)
            elif isinstance(message, HumanMessage):
                with st.chat_message("Human"):
                    st.write(message.content)

        if user_query := st.chat_input("Tanyakan sesuatu tentang data ini..."):
            st.session_state.chat_history.append(HumanMessage(content=user_query))
            with st.chat_message("Human"):
                st.write(user_query)

            with st.chat_message("AI"):
                with st.spinner("Analis AI sedang berpikir..."):
                    response = get_response(user_query, st.session_state.chat_history, chat_model, data_context)
                    st.write(response)
                    st.session_state.chat_history.append(AIMessage(content=response))

st.markdown('<div style="text-align: center; color: black; margin-top: 50px;">Dibuat dengan ❤️ oleh Jati Tepatasa Bagastakwa (dibantu AI)</div>', unsafe_allow_html=True)
st.markdown("")
st.markdown("")
st.markdown("")