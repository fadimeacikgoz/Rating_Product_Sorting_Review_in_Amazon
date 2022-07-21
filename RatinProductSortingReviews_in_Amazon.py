# İŞ PROBLEMİ
# E-ticaretteki en önemli problemlerden bir tanesi ürünlere satış
# sonrası verilen puanların doğru şekilde hesaplanmasıdır. Bu problemin çözümü e-ticaret sitesi için
# daha fazla müşteri memnuniyeti sağlamak, satıcılar için ürünün öne çıkması ve satın alanlar için
# sorunsuz bir alışveriş deneyimi demektir. Bir diğer problem ise ürünlere verilen yorumların doğru
# bir şekilde sıralanması olarak karşımıza çıkmaktadır. Yanıltıcı yorumların öne çıkması ürünün satışını
# doğrudan etkileyeceğinden dolayı hem maddi kayıp hem de müşteri kaybına neden olacaktır. Bu 2 temel
# problemin çözümünde e-ticaret sitesi ve satıcılar satışlarını arttırırken müşteriler ise satın alma
# yolculuğunu sorunsuz olarak tamamlayacaktır.

# Veri Seti Hikayesi
#  Amazon ürün verilerini içeren bu veri seti ürün kategorileri ile çeşitli metadataları içermektedir.
#  Elektronik kategorisindeki en fazla yorum alan ürünün kullanıcı puanları ve yorumları vardır.


#Degiskenler
# reviewerID : Kullanıcı ID’si
# asin :Ürün ID’si
# reviewerName :Kullanıcı Adı
# helpful :Faydalı değerlendirme derecesi
# reviewText :Değerlendirme
# overall :Ürün rating’i
# summary :Değerlendirme özeti
# unixReviewTime :Değerlendirme zamanı
# reviewTime :Değerlendirme zamanı Raw
# day_diff :Değerlendirmeden itibaren geçen gün sayısı
# helpful_yes :Değerlendirmenin faydalı bulunma sayısı
# total_vote : Değerlendirmeye verilen oy sayısı


import pandas as pd
import math
import scipy.stats as st
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler # standarlastırma için
from sklearn.preprocessing import MinMaxScaler
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


df_ = pd.read_csv("/Users/fadimeacikgoz/PycharmProjects/pythonProject/Measurement Problems/measurement_problems/datasets/amazon_review.csv")
df= df_.copy()
print(df.shape)
df.head(10)



#################### Görev 1: Average Rating’i güncel yorumlara göre hesaplayınız###############
#Var olan average rating ile kıyaslayınız.
# Paylaşılan veri setinde kullanıcılar bir ürüne puanlar vermiş ve yorumlar yapmıştır.
# Bu görevde amacımız verilen puanları tarihe göre ağırlıklandırarak değerlendirmek.
# İlk ortalama puan ile elde edilecek tarihe göre ağırlıklı puanın karşılaştırılması gerekmektedir.


# Adım1: Ürünün ortalama puanını hesaplayınız

df["overall"].mean()
df["overall"].describe().T
df["overall"].value_counts()
df.info()
df["overall"].hist()
plt.show()

# Adım 2: Tarihe göre ağırlıklı puan ortalamasını hesaplayınız.

# • reviewTime değişkenini tarih değişkeni olarak tanıtalım
df["reviewTime"]= pd.to_datetime(df["reviewTime"])
df.dtypes


#  - reviewTime'ın max değerini current_date olarak kabul ediniz
current_date = df["reviewTime"].max()

df["days"] = (current_date- df["reviewTime"]).dt.days
df.head()

# • her bir puan-yorum tarihi ile current_date'in farkını gün cinsinden ifade ederek yeni değişken
# oluşturmanız ve gün cinsinden ifade edilen değişkeni quantile fonksiyonu ile 4'e bölüp
# (3 çeyrek verilirse 4 parça çıkar) çeyrekliklerden gelen değerlere göre ağırlıklandırma yapmanız gerekir.
# Örneğin q1 = 12 ise ağırlıklandırırken 12 günden az süre önce yapılan yorumların ortalamasını
# alıp bunlara yüksek ağırlık vermek gibi.

# zaman bazlı ortalama ağırlıkların belirlenmesi
q1 = df["days"].quantile(0.25)  # 280
q2 = df["days"].quantile(0.50)  # 430
q3 = df["days"].quantile(0.75)  # 600


# a,b,c değerlerine göre ağırlıklı puanı hesaplayınız.
df.loc[df["days"] < q1, "overall"].mean()
df.loc[(df["days"] > q1) & (df["days"] <= q2), "overall"].mean()
df.loc[(df["days"] > q2) & (df["days"] <= q3), "overall"].mean()
df.loc[(df["days"] > q3), "overall"].mean()


# Adım 3: Ağırlıklandırılmış puanlamada her bir zaman diliminin ortalamasını karşılaştırıp yorumlayınız.
df.loc[df["days"] < q1, "overall"].mean() * 50/100+ \
df.loc[(df["days"] > q1) & (df["days"] <= q2), "overall"].mean() * 25/100+ \
df.loc[(df["days"] > q2) & (df["days"] <= q3), "overall"].mean() * 15/100+ \
df.loc[(df["days"] > q3), "overall"].mean() * 10/100

### 2.yöntem fonksiyonlaştırılmış hali
def time_based_weighted_average(dataframe, w1=50, w2=25, w3=15, w4=10):
    return df.loc[df["days"] < q1, "overall"].mean() * w1/100+ \
           df.loc[(df["days"] > q1) & (df["days"] <= q2), "overall"].mean() * w2/100+ \
           df.loc[(df["days"] > q2) & (df["days"] <= q3), "overall"].mean() * w3/100+ \
           df.loc[(df["days"] > q3), "overall"].mean() * w4/100

time_based_weighted_average(df)

# ilk basta hesapladıgımız overall ortalaması : 4.587589013224822
# Burada dört ceyrege böldügümüzde gün sayısı 280 günden az olanların ortalamasının diger günlere göre daha yüksek oldugunu
# müşteri menuniyetinin fazla oldugu görülmektedir.



#####################################
#Görev 2: Ürün için ürün detay sayfasında görüntülenecek 20 review’i belirleyiniz.
#####################################



###################################################
# Adım 1. helpful_no Değişkenini Üretiniz
###################################################
df["helpful"].dtypes
df["helpful"]

# 1.yol
df["helpful"]=df["helpful"].str.strip('[ ]')
df["helpful_yes"]=df["helpful"].apply(lambda x:x.split(", ")[0]).astype(int)
df["total_vote"]=df["helpful"].apply(lambda x:x.split(", ")[1]).astype(int)
df["helpful_no"] = df["total_vote"] - df["helpful_yes"]

# 2.yol
df["helpful_yes"]=df[["helpful"]].applymap(lambda x:x.split(", ")[0].strip('[')).astype(int)
df["total_vote"]=df[["helpful"]].applymap(lambda x:x.split(", ")[1].strip(']')).astype(int)
df["helpful_no"] = df["total_vote"] - df["helpful_yes"]

df.head()

###################################################
# Adım 2. score_pos_neg_diff, score_average_rating ve wilson_lower_bound Skorlarını Hesaplayıp Veriye Ekleyiniz
###################################################

# score_pos_neg_diff

#1. yol
def score_pos_neg_diff(helpful_yes, helpful_no):
    if helpful_yes - helpful_no <0 :
        return 0
    return helpful_yes - helpful_no

df["score_pos_neg_diff"]= df["helpful_yes"]- df["helpful_no"]
#2. yol
def score_pos_neg_diff(up,down):
    return up - down

df["score_pos_neg_diff"]= df.apply(lambda x: score_pos_neg_diff(x["helpful_yes"], x["helpful_no"]), axis=1)
df.head()

# # score_average_rating
def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up/(up+down)

df["score_average_rating"] = df.apply(lambda x:score_average_rating(x["helpful_yes"], x["helpful_no"]), axis=1)
df.head()


# confidence= 0.95 güven aralıgı
def wilson_lower_bound(up, down, confidence=0.95):
    """
    Wilson Lower Bound Score hesapla

    - Bernoulli parametresi p için hesaplanacak güven aralığının alt sınırı WLB skoru olarak kabul edilir.
    - Hesaplanacak skor ürün sıralaması için kullanılır.
    - Not:
    Eğer skorlar 1-5 arasıdaysa 1-3 negatif, 4-5 pozitif olarak işaretlenir ve bernoulli'ye uygun hale getirilebilir.
    Bu beraberinde bazı problemleri de getirir. Bu sebeple bayesian average rating yapmak gerekir.

    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    """
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)


df["wilson_lower_bound"]= df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)
df["helpful_yes"]
df["helpful_no"]

df.head(1000)

df.sort_values("wilson_lower_bound", ascending= False)[:50]

