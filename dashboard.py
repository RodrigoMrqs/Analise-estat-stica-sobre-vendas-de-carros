import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# ===============================
# Configurações iniciais
# ===============================
st.set_page_config(page_title="Car Sales Dashboard", layout="wide")

st.title("🚗 Car Sales Dashboard")
st.markdown("Este dashboard interativo auxilia usuários e vendedores na análise de mercado de carros, "
            "com base nos dados fornecidos.")

# ===============================
# Carregando os dados
# ===============================
@st.cache_data
def load_data():
    df = pd.read_csv("car_sales_data.csv")
    # Engenharia de variáveis (já feita no notebook)
    current_year = 2025
    df["Car_Age"] = current_year - df["Year of manufacture"]
    df["Price_Category"] = pd.qcut(df["Price"], q=4, labels=["Economic", "Mid-Range", "Premium", "Luxury"])
    df["Fuel_Efficiency"] = df["Mileage"] / df["Engine size"]
    df["Engine_Category"] = pd.qcut(df["Engine size"], q=3, labels=["Small", "Medium", "Large"])
    df["Decada"] = (df["Year of manufacture"] // 10) * 10
    return df

df = load_data()

# ===============================
# Barra lateral de filtros
# ===============================
st.sidebar.header("Filtros")

fabricantes = st.sidebar.multiselect("Selecione Fabricantes:", options=df["Manufacturer"].unique(),
                                     default=df["Manufacturer"].unique())
combustiveis = st.sidebar.multiselect("Selecione Tipos de Combustível:", options=df["Fuel type"].unique(),
                                      default=df["Fuel type"].unique())
decadas = st.sidebar.multiselect("Selecione Décadas:", options=sorted(df["Decada"].unique()),
                                 default=sorted(df["Decada"].unique()))

df_filtered = df[(df["Manufacturer"].isin(fabricantes)) &
                 (df["Fuel type"].isin(combustiveis)) &
                 (df["Decada"].isin(decadas))]

st.sidebar.write(f"**Total de registros filtrados:** {len(df_filtered)}")

# ===============================
# Abas principais
# ===============================
tabs = st.tabs([
    "📊 Visão Geral",
    "🏭 Análises por Fabricante",
    "⛽ Análises por Combustível",
    "📈 Distribuições & Décadas",
    "📐 Correlações"
])

# ===============================
# Aba 1 - Visão Geral
# ===============================
with tabs[0]:
    st.subheader("Resumo Estatístico")
    st.dataframe(df_filtered.describe())

    col1, col2, col3 = st.columns(3)
    col1.metric("Preço Médio", f"${df_filtered['Price'].mean():,.2f}")
    col2.metric("Idade Média", f"{df_filtered['Car_Age'].mean():.1f} anos")
    col3.metric("Eficiência Média", f"{df_filtered['Fuel_Efficiency'].mean():.2f} km")

# ===============================
# Aba 2 - Fabricantes
# ===============================
with tabs[1]:
    st.subheader("Análises por Fabricante")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    sns.barplot(data=df_filtered, x='Manufacturer', y='Price', ax=axes[0, 0])
    axes[0, 0].set_title("Preço médio por fabricante")

    sns.barplot(data=df_filtered, x='Manufacturer', y='Engine size', ax=axes[0, 1])
    axes[0, 1].set_title("Tamanho do motor por fabricante")

    sns.barplot(data=df_filtered, x='Manufacturer', y='Mileage', ax=axes[1, 0])
    axes[1, 0].set_title("Quilometragem média por fabricante")

    sns.countplot(data=df_filtered, x='Manufacturer', ax=axes[1, 1])
    axes[1, 1].set_title("Quantidade de vendas por fabricante")

    plt.tight_layout()
    st.pyplot(fig)

# ===============================
# Aba 3 - Combustível
# ===============================
with tabs[2]:
    st.subheader("Análises por Tipo de Combustível")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.barplot(data=df_filtered, x='Fuel type', y='Price', ax=axes[0])
    axes[0].set_title("Preço médio do veículo por tipo de combustível")

    sns.countplot(data=df_filtered, x='Fuel type', ax=axes[1])
    axes[1].set_title("Distribuição de combustível")

    plt.tight_layout()
    st.pyplot(fig)

    st.write("### Frequência de Combustíveis")
    combustivel_freq = df_filtered['Fuel type'].value_counts()
    combustivel_perc = df_filtered['Fuel type'].value_counts(normalize=True) * 100
    st.dataframe(pd.DataFrame({'Frequência': combustivel_freq, 'Percentual (%)': combustivel_perc.round(2)}))

# ===============================
# Aba 4 - Décadas e Distribuições
# ===============================
with tabs[3]:
    st.subheader("Distribuições e Décadas")

    st.write("#### Distribuição por década de fabricação")
    decada_freq = df_filtered['Decada'].value_counts().sort_index()
    decada_perc = df_filtered['Decada'].value_counts(normalize=True) * 100
    st.bar_chart(decada_freq)
    st.dataframe(pd.DataFrame({'Frequência': decada_freq, 'Percentual (%)': decada_perc.round(2)}))

    st.write("#### Scatterplot: Tamanho do Motor × Quilometragem")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df_filtered, x='Engine size', y='Mileage', hue='Fuel type', ax=ax)
    st.pyplot(fig)

# ===============================
# Aba 5 - Correlações
# ===============================
with tabs[4]:
    st.subheader("Relação entre Idade do Carro e Preço por Tipo de Combustível")
    numeric_cols = ['Price', 'Engine size', 'Mileage', 'Car_Age', 'Fuel_Efficiency']
    corr = df_filtered[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.scatterplot(data=df_filtered, x='Car_Age', y='Price', hue='Fuel type', alpha=0.6, ax=ax)
    st.pyplot(fig)



