import pandas as pd
import numpy as np
########## GÖREV 1 ##########
# Soru 1: persona.csv dosyasını okutunuz ve veri seti ile ilgili genel bilgileri gösteriniz.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.max_rows', 1000)
df = pd.read_csv("persona.csv")

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)

# Soru 2: Kaç unique SOURCE vardır? Frekansları nedir?
print("Unique SOURCE adeti:", df["SOURCE"].nunique())
print("Unique SOURCE isimleri:", df["SOURCE"].unique())
print("Unique SOURCE sayıları:\n", df["SOURCE"].value_counts())

# Soru 3: Kaç unique PRICE vardır?
price = df["PRICE"].nunique()
print(f"{price} adet unique PRICE vardır")
df.head()

# Soru 4: Hangi PRICE'dan kaçar tane satış gerçekleşmiş?
piece = df["PRICE"].value_counts().sort_index()
for price, count in piece.items():
    print(f"Fiyatı {price} olan PRICE'dan {count} adet satış gerçekleşmiştir.")

# Soru 5: Hangi ülkeden kaçar tane satış olmuş?
country = df["COUNTRY"].value_counts()
for country, count in country.items():
    print(f"{country.upper()} ülkesinden {count} adet satış gerçekleşmiştir.")
    #print(f"{country.capitalize()} ülkesinden {count} adet satış gerçekleşmiştir.")

# Soru 6: Ülkelere göre satışlardan toplam ne kadar kazanılmış?
sum = df.groupby("COUNTRY")["PRICE"].agg('sum')
for country, sum_price in sum.items():
    print(f"{country.upper()} ülkesinden toplam {sum_price} kazanılmıştır.")

# Soru 7: SOURCE türlerine göre satış sayıları nedir?
source = df["SOURCE"].value_counts()
for source, piece in source.items():
    print(f"{source.capitalize()} SOURCE türünden toplam {piece} adet satılmıştır.")

# Soru 8: Ülkelere göre PRICE ortalamaları nedir?
mean_country = df.groupby("COUNTRY")["PRICE"].agg('mean')
for country, m_price in mean_country.items():
    print(f"{country.upper()} ülkesinin PRICE ortalaması {m_price:.3f}'dir")

# Soru 9: SOURCE'lara göre PRICE ortalamaları nedir?
mean_source = df.groupby("SOURCE")["PRICE"].agg('mean')
for source, m_price in mean_source.items():
    print(f"{source.capitalize()} Source'nun PRICE ortalaması {m_price:.3f}'dir")

# Soru 10: COUNTRY-SOURCE kırılımında PRICE ortalamaları nedir?
country_source = df.groupby(["COUNTRY", "SOURCE"])["PRICE"].agg('mean')
for (country, source), m_price in country_source.items():
    print(f"{country.upper()} ülkesinde {source.capitalize()} kullanıcısının ortalama PRICE değeri: {m_price:.2f}'dir")

########## GÖREV 2 ##########
# Soru: COUNTRY, SOURCE, SEX, AGE kırılımında ortalama kazançlar nedir?
# YOL 1:
agg_df = df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"])[["PRICE"]].agg('mean')
agg_df.head()

# YOL 2:
pivot_table = df.pivot_table(values='PRICE', index=['COUNTRY', 'SOURCE', 'SEX', 'AGE'], aggfunc='mean')


print("COUNTRY, SOURCE, SEX, AGE kırılımında ortalama kazançlar:\n", pivot_table.head())

########################## Görev 3: Çıktıyı PRICE’a göre sıralayınız. BAKILACAK ##########################
# Önceki sorudaki çıktıyı daha iyi görebilmek için sort_values metodunu azalan olacak şekilde PRICE’a göre uygulayınız.
# Çıktıyı agg_df olarak kaydediniz.

# YOL 1:
agg_df = agg_df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"])[["PRICE"]].mean()
#ÇİFT KÖŞELİ PARANTEZ YAPARSAM DFRAME OLUR YAPMAZSAM SERIES OLUR
agg_df = agg_df.sort_values("PRICE", ascending=False)
agg_df.head(6)

# YOL 2:
agg_df2 = pivot_table.sort_values("PRICE", ascending=False)
agg_df2.head(6)

# Görev 4: Indekste yer alan isimleri değişken ismine çeviriniz.
# Üçüncü sorunun çıktısında yer alan PRICE dışındaki tüm değişkenler index isimleridir.
# Bu isimleri değişken isimlerine çeviriniz.
agg_df = agg_df.reset_index()
agg_df.info()
agg_df.head()

# Görev 5: Age değişkenini kategorik değişkene çeviriniz ve agg_df’e ekleyiniz.
values = [0, 18, 23, 30, 40, 70]
labels = ['0_18', '19_23', '24_30', '31_40', '41_70']
agg_df["AGE"] = agg_df["AGE"].astype("category")
#
agg_df["AGE_CAT"] = pd.cut(agg_df["AGE"], bins=values, labels=labels, include_lowest=True, right=True)
agg_df.head()
agg_df.info()
# cut ve qcut fonksiyonları, sayısal değişkenleri kategorik değişkenlere dönüştürür
# Sayısal değişkeni hangi kategorilere bölmek istediğimi biliyorsak, cut fonksiyonu kullanılır.
# Sayısal değişkeni kategorilere bölmek istediğimizi bilmiyorsak, qcut fonksiyonu kullanılır.
# Çeyreklik değerlerine göre bölünür.
# qcut, otomatik olarak değerleri küçükten büyüğe sıralar ve yüzdelik çeyrek değerlerine göre kategorik olarak böler.


# Görev 6: Yeni seviye tabanlı müşterileri (persona) tanımlayınız.
# Yeni seviye tabanlı müşterileri (persona) tanımlayınız ve veri setine değişken olarak ekleyiniz.
# Yeni eklenecek değişkenin adı: customers_level_based
# Önceki soruda elde edeceğiniz çıktıdaki gözlemleri bir araya getirerek customers_level_based değişkenini oluşturmanız gerekmektedir.

agg_df["customers_level_based"] = [
    f"{row['COUNTRY'].upper()}_{row['SOURCE'].upper()}_{row['SEX'].upper()}_{row['AGE_CAT']}"
    for index, row in agg_df.iterrows()
]

agg_df["customers_level_based"].unique()

agg_df.head()
persona = agg_df.groupby("customers_level_based")[["PRICE"]].mean().reset_index()
persona["customers_level_based"].value_counts()
# persona = persona.sort_values("PRICE", ascending=False)

# iterrows: Satırları ve satırların indexlerini döner.
#  print(index): Satırın index'i (0, 1, 2, ...)
#  print(row): İlgili satırdaki tüm veriler (Series formatında)

# items(): Sütunları ve değerlerini döndürür.
# iterrows: İlgili satırları döndürür.

# Görev 7: Yeni müşterileri (personaları) segmentlere ayırınız.
# Yeni müşterileri (Örnek: USA_ANDROID_MALE_0_18) PRICE’a göre 4 segmente ayırınız.
# Segmentleri SEGMENT isimlendirmesi ile değişken olarak agg_df’e ekleyiniz.
# Segmentleri betimleyiniz (Segmentlere göre group by yapıp price mean, max, sum’larını alınız).

persona["SEGMENT"] = pd.qcut(persona["PRICE"], 4, labels=['D', 'C', 'B', 'A'])
segment_sum = persona.groupby("SEGMENT", observed=True)[["PRICE"]].agg(['mean', 'max', 'sum'])


# Görev 8: Yeni gelen müşterileri sınıflandırıp, ne kadar gelir getirebileceklerini tahmin ediniz.
# 33 yaşında ANDROID kullanan bir Türk kadını hangi segmente aittir ve ortalama ne kadar gelir kazandırması beklenir?
# 35 yaşında IOS kullanan bir Fransız kadını hangi segmente aittir ve ortalama ne kadar gelir kazandırması beklenir?

new_user = "TUR_ANDROID_FEMALE_31_40"
new_user2 = "FRA_IOS_FEMALE_31_40"
result1 = persona[persona["customers_level_based"] == new_user]
result2 = persona[persona["customers_level_based"] == new_user2]

result = pd.concat([result1, result2], axis=0).reset_index()
result["expected_total_revenue"] = result["PRICE"]
results_final = result.drop(columns=["PRICE"])
