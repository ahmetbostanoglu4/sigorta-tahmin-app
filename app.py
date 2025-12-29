import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import streamlit as st

st.set_page_config(page_title="Sigorta Tahmin Uygulaması", page_icon="🛡️", layout="centered")

st.title("Sigorta Tahmin Uygulaması")
st.write("Hoş geldiniz! Bu uygulama araç ve sağlık sigortası prim tahmini yapar.")

# -------------------------------
# Kullanıcıdan temel inputlar
# -------------------------------
age = st.number_input("Yaşınızı girin:", min_value=18, max_value=100, step=1, key="age_input")
gender = st.radio("Cinsiyetinizi seçin:", ["Erkek", "Kadın"], key="gender_input")
insurance_type = st.selectbox("Sigorta türü:", ["Sağlık", "Araç"], key="insurance_type_input")

# -------------------------------
# Araç Sigortası Modeli
# -------------------------------
car_df = pd.read_csv("car_insurance_premium_dataset.csv")
X_car = car_df[["Driver Age", "Driver Experience", "Previous Accidents", "Annual Mileage (x1000 km)", "Car Age"]]
y_car = car_df["Insurance Premium ($)"]

X_train, X_test, y_train, y_test = train_test_split(X_car, y_car, test_size=0.2, random_state=42)
car_model = LinearRegression()
car_model.fit(X_train, y_train)
joblib.dump(car_model, "car_model.pkl")

if insurance_type == "Araç":
    st.subheader("Araç Sigortası Tahmini")

    # Araç için ek inputlar
    experience = st.number_input("Sürücü deneyimi (yıl):", min_value=0, max_value=80, step=1, key="experience_input")
    accidents = st.number_input("Geçmiş kaza sayısı:", min_value=0, max_value=10, step=1, key="accidents_input")
    mileage = st.number_input("Yıllık km (bin km):", min_value=1, max_value=50, step=1, key="mileage_input")
    car_age = st.number_input("Araç yaşı:", min_value=0, max_value=40, step=1, key="car_age_input")

    model = joblib.load("car_model.pkl")
    input_data = [[age, experience, accidents, mileage, car_age]]
    prediction = model.predict(input_data)[0]
    st.success(f"Tahmini prim (araç): {round(prediction, 2)} $")

# -------------------------------
# Sağlık Sigortası Modeli
# -------------------------------
health_df = pd.read_csv("insurance.csv")

# Kategorik değişkenleri encode et
health_df_encoded = pd.get_dummies(health_df, columns=["cinsiyet", "sigara", "bolge"], drop_first=True)
X_health = health_df_encoded.drop("Prim (TL)", axis=1)
y_health = health_df_encoded["Prim (TL)"]

X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(X_health, y_health, test_size=0.2, random_state=42)
health_model = LinearRegression()
health_model.fit(X_train_h, y_train_h)
joblib.dump(health_model, "health_model.pkl")

if insurance_type == "Sağlık":
    st.subheader("Sağlık Sigortası Tahmini")

    # Sağlık için ek inputlar
    bmi = st.number_input("Vücut kitle indeksiniz (BMI):", min_value=10.0, max_value=60.0, step=0.1, key="bmi_input")
    children = st.number_input("Çocuk sayınız:", min_value=0, max_value=10, step=1, key="children_input")
    smoker = st.radio("Sigara kullanıyor musunuz?", ["yes", "no"], key="smoker_input")
    region = st.selectbox("Yaşadığınız bölge:", ["northeast", "northwest", "southeast", "southwest"], key="region_input")

    model = joblib.load("health_model.pkl")

    # Kullanıcı verisini encode et
    input_data = pd.DataFrame({
        "yas": [age],
        "bmi": [bmi],
        "cocuk": [children],
        "cinsiyet_male": [1 if gender == "Erkek" else 0],
        "sigara_yes": [1 if smoker == "yes" else 0],
        "bolge_northwest": [1 if region == "northwest" else 0],
        "bolge_southeast": [1 if region == "southeast" else 0],
        "bolge_southwest": [1 if region == "southwest" else 0],
    })

    prediction = model.predict(input_data)[0]
    st.success(f"Tahmini prim (sağlık): {round(prediction, 2)} TL")