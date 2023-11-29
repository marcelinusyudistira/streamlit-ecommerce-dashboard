import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import streamlit as st
from babel.numbers import format_currency

sns.set(style="dark")


# helper function
def create_daily_orders_df(df, category_col):
    daily_orders_df = df.resample(rule="D", on="order_purchase_timestamp").agg(
        {"order_id": "nunique", "total_price": "sum", category_col: "first"}
    )
    daily_orders_df = daily_orders_df.reset_index()
    daily_orders_df.rename(
        columns={
            "order_id": "order_count",
            "total_price": "revenue",
            category_col: "category",
        },
        inplace=True,
    )

    return daily_orders_df


def create_category_by_order(df):
    category_counts = (
        df.groupby("product_category_name_english")["order_id"]
        .count()
        .sort_values(ascending=False)
    )

    return category_counts


def create_payment_types_by_frequency(df):
    payment_stats_df = (
        df.groupby("payment_type")["order_id"].nunique().rename("frequency")
    )

    payment_stats_df = payment_stats_df.to_frame().join(
        df.groupby("payment_type")["payment_value"].sum()
    )
    payment_stats_df = payment_stats_df.sort_values(
        by=["frequency", "payment_value"], ascending=False
    )
    payment_stats_df["frequency_percentage"] = (
        payment_stats_df["frequency"] / payment_stats_df["frequency"].sum() * 100
    )
    payment_stats_df["payment_value_million"] = (
        payment_stats_df["payment_value"] / 1000000
    )

    return payment_stats_df


def create_category_by_rating(df):
    skor_kategori = df.groupby("product_category_name_english")["review_score"].mean()

    return skor_kategori


def create_city_by_revenue(df):
    revenue_per_city = (
        df.groupby("customer_city")[["payment_value"]]
        .sum()
        .sort_values(by="payment_value", ascending=False)
    )
    revenue_per_city.reset_index(inplace=True)

    return revenue_per_city


def create_rfm_df(df):
    rfm_df = df.groupby(by="customer_unique_id", as_index=False).agg(
        {
            "order_purchase_timestamp": "max",  # mengambil tanggal order terakhir
            "order_id": "nunique",
            "total_price": "sum",
        }
    )
    rfm_df.columns = [
        "customer_unique_id",
        "max_order_timestamp",
        "frequency",
        "monetary",
    ]

    rfm_df["max_order_timestamp"] = rfm_df["max_order_timestamp"].dt.date
    recent_date = df["order_purchase_timestamp"].dt.date.max()
    rfm_df["recency"] = rfm_df["max_order_timestamp"].apply(
        lambda x: (recent_date - x).days
    )
    rfm_df.drop("max_order_timestamp", axis=1, inplace=True)

    return rfm_df


def create_segmentation_df(segmentation_df):
    # Proses pengelompokan dan segmentasi
    segmentation_df["recency_score"] = pd.qcut(
        segmentation_df["recency"], 5, labels=[5, 4, 3, 2, 1]
    )
    segmentation_df["frequency_score"] = pd.qcut(
        segmentation_df["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]
    )
    segmentation_df["monetary_score"] = pd.qcut(
        rfm_df["monetary"], 5, labels=[1, 2, 3, 4, 5]
    )
    segmentation_df["score_rfm"] = (
        segmentation_df.recency_score.astype(str)
        + segmentation_df.frequency_score.astype(str)
        + segmentation_df.monetary_score.astype(str)
    )

    seg_map = {
        r"111|112|121|131|141|151": "Lost customers",
        r"332|322|233|232|223|222|132|123|122|212|211": "Hibernating customers",
        r"155|154|144|214|215|115|114|113|255|254|245|244|253|252|243|242|235|234|225|224|153|152|145|143|142|135|134|133|125|124": "At Risk",
        r"331|321|312|221|213|231|241|251|543|444|435|355|354|345|344|335|553|551|552|541|542|533|532|531|452|451|442|441|431|453|433|432|423|353|352|351|342|341|333|323": "Potential Loyalist",
        r"535|534|443|434|343|334|325|324|525|524|523|522|521|515|514|513|425|424|413|414|415|315|314|313": "Promising",
        r"512|511|422|421|412|411|311|543|444|435|355|354|345|344|335|553|551|552|541|542|533|532|531|452|451|442|441|431|453|433|432|423|353|352|351|342|341|333|323|555|554|544|545|454|455|445": "Champions",
    }

    segmentation_df["Segments"] = (
        segmentation_df["recency_score"].astype(str)
        + segmentation_df["frequency_score"].astype(str)
        + segmentation_df["monetary_score"].astype(str)
    )

    segmentation_df["Segments"] = segmentation_df["Segments"].replace(
        seg_map, regex=True
    )

    return segmentation_df


# load semua data
all_df = pd.read_csv("all_merged_df.csv")
local_image_path = "market_2.png"

datetime_columns = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_estimated_delivery_date",
    "order_delivered_customer_date",
]
all_df.sort_values(by="order_purchase_timestamp", inplace=True)
all_df.reset_index(inplace=True)

for column in datetime_columns:
    all_df[column] = pd.to_datetime(all_df[column])

# membuat komponen filter
min_date = all_df["order_purchase_timestamp"].min()
max_date = all_df["order_purchase_timestamp"].max()

with st.sidebar:
    # Menambahkan logo perusahaan
    st.image(
        local_image_path,
        caption="O'list E-Commerce",
    )

    # Mengambil start_date & end_date dari date_input
    start_date, end_date = st.date_input(
        label="Rentang Waktu",
        min_value=min_date,
        max_value=max_date,
        value=[min_date, max_date],
    )

    unique_categories = all_df["product_category_name_english"].drop_duplicates()

    # Menambahkan opsi "Semua Kategori"
    unique_categories = pd.concat(
        [pd.Series(["All Categories"]), unique_categories], ignore_index=True
    )

    # Menambahkan selectbox untuk kategori
    selected_category = st.selectbox(
        label="Pilih Kategori",
        options=unique_categories,
    )

if selected_category == "All Categories":
    main_df = all_df[
        (all_df["order_purchase_timestamp"] >= str(start_date))
        & (all_df["order_purchase_timestamp"] <= str(end_date))
    ]
else:
    main_df = all_df[
        (all_df["order_purchase_timestamp"] >= str(start_date))
        & (all_df["order_purchase_timestamp"] <= str(end_date))
        & (all_df["product_category_name_english"] == selected_category)
    ]

daily_orders_df = create_daily_orders_df(main_df, "product_category_name_english")
rfm_df = create_rfm_df(all_df)
segmentation_df = create_segmentation_df(rfm_df)
category_counts = create_category_by_order(all_df)
category_counts_rating = create_category_by_rating(all_df)
city_by_revenue = create_city_by_revenue(all_df)
payment_types_percentage = create_payment_types_by_frequency(all_df)

# ==========================plot number of daily orders (2021)===========================
st.header("O'list E-Commerce Dashboard	:earth_americas:")
# Menampilkan subheader dengan HTML
st.markdown(
    f"<span style='font-size: 28px;'>**Daily Orders for :** {selected_category}</span>",
    unsafe_allow_html=True,
)

col1, col2 = st.columns(2)

with col1:
    total_orders = daily_orders_df.order_count.sum()
    st.metric("Total orders", value=total_orders)

with col2:
    total_revenue = format_currency(
        daily_orders_df.revenue.sum(), "AUD", locale="es_CO"
    )
    st.metric("Total Revenue", value=total_revenue)

fig, ax = plt.subplots(figsize=(16, 8))
ax.plot(
    daily_orders_df["order_purchase_timestamp"],
    daily_orders_df["order_count"],
    marker="o",
    linewidth=2,
    color="#90CAF9",
)
ax.tick_params(axis="y", labelsize=20)
ax.tick_params(axis="x", labelsize=15)
plt.xlabel("Date")
plt.ylabel("Order Count")
plt.title("Daily Order Count")

st.pyplot(fig)


# ==========================Top Categories by Order count=================================
st.subheader("Top Categories by Order")
tertinggi_6 = category_counts.head(6)
terendah_6 = category_counts.tail(6)

# membuat 2 subplot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# membuat plot dari 6 kategori tertinggi disebelah kiri
sns.barplot(x=tertinggi_6.index, y=tertinggi_6.values, color="forestgreen", ax=ax1)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)
ax1.set_xlabel("Product Category")
ax1.set_ylabel("Number of Order")
ax1.set_title("Top 6 Highest Product Categories based on number of Orders")

# membuat plot dari 6 kategori terendah disebelah kanan
sns.barplot(x=terendah_6.index, y=terendah_6.values, color="firebrick", ax=ax2)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90)
ax2.set_xlabel("Product Category")
ax2.set_ylabel("Number of Order")
ax2.set_title("Top 6 Lowest Product Categories based on number of Orders")

st.pyplot(fig)


# ================================Top Seller City ===============================
st.subheader("10 Cities with the largest number of Sellers")
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(
    x=all_df.seller_city.value_counts().values[:10],
    y=all_df.seller_city.value_counts().index[:10],
    palette="crest_r",
    ax=ax,  # Menggunakan axes yang diberikan oleh fig
)
ax.set_title("10 Cities with the largest number of Sellers")

# Menampilkan gambar di Streamlit
st.pyplot(fig)


# ============================Payment Types Percentage==================================
st.subheader("Frequency and Percentage of each Payment Type based on Total Revenue")

# Membuat bar plot dari frekuensi dan persentase pembayaran
plt.figure(figsize=(8, 5))
sns.barplot(
    data=payment_types_percentage,
    x=payment_types_percentage.index,
    y="frequency",
    palette="flare",
)
plt.xlabel("Payment Type")
plt.ylabel("Frequency")
plt.title("Frequency and Percentage of each Payment Type based on Total Revenue")

# Menambahkan teks pada setiap batang diagram
for i, v in enumerate(payment_types_percentage["frequency_percentage"]):
    plt.text(i - 0.1, v + 10, f"{v:.1f}%", color="black")

# Menampilkan bar plot di Streamlit menggunakan st.pyplot
st.pyplot(plt)


# ==============================Top Categories by Order rating================================
st.subheader("Top 6 Categories by Rating")
kategori_tertinggi = category_counts_rating.nlargest(6).sort_values(ascending=False)
kategori_terendah = category_counts_rating.nsmallest(6).sort_values(ascending=False)

# membuat 2 subplot
fig, axs = plt.subplots(1, 2, figsize=(15, 6))

# membuat bar plot untuk 6 kategori tertinggi
axs[0].bar(kategori_tertinggi.index, kategori_tertinggi.values)
axs[0].set_title("6 Kategori Produk tertinggi berdasarkan Rating")
axs[0].set_xlabel("Kategori Produk")
axs[0].set_xticks(range(len(kategori_tertinggi)))
axs[0].set_xticklabels(kategori_tertinggi.index, rotation=90)
axs[0].set_ylabel("Rerata Skor Review")
axs[0].set_ylim([1, 5])

# membuat bar plot untuk 6 kategori terendah
axs[1].bar(kategori_terendah.index, kategori_terendah.values)
axs[1].set_title("6 Kategori Produk terendah berdasarkan Rating")
axs[1].set_xticks(range(len(kategori_terendah)))
axs[1].set_xticklabels(kategori_terendah.index, rotation=90)
axs[1].set_xlabel("Kategori Produk")
axs[1].set_ylabel("Rerata Skor Review")
axs[1].set_ylim([1, 5])

# jeda rentang antar plot
plt.subplots_adjust(wspace=0.4)
st.pyplot(fig)

# ============================= Top city with highest revenue ======================
st.subheader("Top 10 cities with the highest Revenue")
# Membuat bar plot dengan Altair
bar_plot = (
    alt.Chart(city_by_revenue[:10])
    .mark_bar()
    .encode(
        x=alt.X(
            "customer_city:N",
            sort=alt.EncodingSortField(
                field="payment_value", op="sum", order="descending"
            ),
        ),
        y="payment_value:Q",
        color=alt.value("steelblue"),
        tooltip=["customer_city:N", "payment_value:Q"],
    )
    .properties(title="Top 10 cities with the highest Revenue", width=600, height=400)
)

# Menampilkan bar plot di Streamlit
st.altair_chart(bar_plot, use_container_width=True)


# =======================Best Customer Based on RFM Parameters=============================
st.subheader("Best Customer Based on RFM Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    avg_recency = round(rfm_df.recency.mean(), 1)
    st.metric("Average Recency (days)", value=avg_recency)

with col2:
    avg_frequency = round(rfm_df.frequency.mean(), 2)
    st.metric("Average Frequency", value=avg_frequency)

with col3:
    avg_frequency = format_currency(rfm_df.monetary.mean(), "AUD", locale="es_CO")
    st.metric("Average Monetary", value=avg_frequency)

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(35, 15))
colors = ["#90CAF9", "#90CAF9", "#90CAF9", "#90CAF9", "#90CAF9"]

sns.barplot(
    y="recency",
    x="customer_unique_id",
    data=rfm_df.sort_values(by="recency", ascending=True).head(8),
    palette=colors,
    ax=ax[0],
)
ax[0].set_ylabel(None)
ax[0].set_xlabel("customer_unique_id", fontsize=30)
ax[0].set_title("By Recency (days)", loc="center", fontsize=50)
ax[0].tick_params(axis="y", labelsize=30)
ax[0].tick_params(axis="x", labelsize=35, rotation=90)

sns.barplot(
    y="frequency",
    x="customer_unique_id",
    data=rfm_df.sort_values(by="frequency", ascending=False).head(8),
    palette=colors,
    ax=ax[1],
)
ax[1].set_ylabel(None)
ax[1].set_xlabel("customer_unique_id", fontsize=30)
ax[1].set_title("By Frequency", loc="center", fontsize=50)
ax[1].tick_params(axis="y", labelsize=30)
ax[1].tick_params(axis="x", labelsize=35, rotation=90)

sns.barplot(
    y="monetary",
    x="customer_unique_id",
    data=rfm_df.sort_values(by="monetary", ascending=False).head(8),
    palette=colors,
    ax=ax[2],
)
ax[2].set_ylabel(None)
ax[2].set_xlabel("customer_unique_id", fontsize=30)
ax[2].set_title("By Monetary", loc="center", fontsize=50)
ax[2].tick_params(axis="y", labelsize=30)
ax[2].tick_params(axis="x", labelsize=35, rotation=90)

st.pyplot(fig)

st.subheader("RFM Analysis")

# ======================Menampilkan bar plot segmentasi pelanggan================================
st.markdown(
    "<span style='font-size: 24px;'>Customer Segmentation based on RFM analysis</span>",
    unsafe_allow_html=True,
)
fig, ax = plt.subplots(figsize=(10, 5))
plotSegment = sns.barplot(
    x=segmentation_df["Segments"].value_counts().index,
    y=segmentation_df["Segments"].value_counts().values,
)
for bar in ax.patches:
    ax.annotate(
        format(bar.get_height(), ".0f"),
        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
        ha="center",
        va="center",
        size=10,
        xytext=(0, 5),
        textcoords="offset points",
    )
plt.xticks(rotation=90)
ax.set_title("Customer Segmentation based on RFM analysis")
ax.set_xlabel("Segments")
st.pyplot(fig)

# Menampilkan pie chart segmentasi pelanggan
st.markdown(
    "<span style='font-size: 24px;'>Customer Segmentation Ratio based on RFM analysis</span>",
    unsafe_allow_html=True,
)
fig, ax = plt.subplots()
ax.pie(
    segmentation_df["Segments"].value_counts(),
    labels=segmentation_df["Segments"].value_counts().index,
    autopct="%.2f%%",
    pctdistance=0.8,
)
ax.set_title("Customer Segmentation Ratio based on RFM analysis")
st.pyplot(fig)

st.caption("Copyright © Marcelinus Yudistira Yoga Pratama 2023")


