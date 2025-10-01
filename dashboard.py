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
    "📐 Correlações",
    "📉 Regressão Linear"
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

# ===============================
# Aba 7 - Regressão Linear
# ===============================
with tabs[5]:
    st.subheader("📉 Modelo de Regressão Linear Interativo")

    st.markdown("""
    O modelo de regressão linear permite **estimar o preço dos carros** com base em variáveis como tamanho do motor, quilometragem e idade do carro.
    Use os filtros laterais e selecione as variáveis abaixo para ajustar o modelo de forma dinâmica.
    """)

    # Seleção de variáveis preditoras
    all_features = ["Engine size", "Mileage", "Car_Age", "Fuel_Efficiency"]
    selected_features = st.multiselect(
        "Selecione as variáveis independentes (X):",
        all_features,
        default=["Engine size", "Mileage", "Car_Age"]
    )

    if selected_features:
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        import numpy as np

        X = df_filtered[selected_features]
        y = df_filtered["Price"]

        if len(X) > 20:  # Garantir que há dados suficientes
            # Treino e teste
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Resultados numéricos
            st.markdown("### 📊 Avaliação do Modelo")
            col1, col2, col3 = st.columns(3)
            col1.metric("R²", f"{r2_score(y_test, y_pred):.4f}")
            col2.metric("MAE", f"{mean_absolute_error(y_test, y_pred):.2f}")
            col3.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

            # Coeficientes
            st.markdown("### ⚖️ Coeficientes do Modelo")
            coef_df = pd.DataFrame(model.coef_, index=selected_features, columns=["Coeficiente"])
            coef_df.loc["Intercepto"] = model.intercept_
            st.dataframe(coef_df)

            # Visualização
            st.markdown("### 📈 Visualização Gráfica")

            if len(selected_features) == 1:
                # Regressão simples -> gráfico 2D
                feature = selected_features[0]
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(X_test[feature], y_test, alpha=0.6, label="Dados reais")
                ax.plot(X_test[feature], y_pred, color="red", linewidth=2, label="Linha de Regressão")
                ax.set_xlabel(feature)
                ax.set_ylabel("Preço")
                ax.set_title(f"Regressão Linear: {feature} x Preço")
                ax.legend()
                st.pyplot(fig)

                st.info("👉 A linha vermelha representa a relação prevista entre a variável selecionada e o preço do carro.")

            else:
                # Comparação valores reais vs previstos
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(y_test, y_pred, alpha=0.6, c="blue", label="Previsões")
                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2, label="Ideal (y=x)")
                ax.set_xlabel("Preço Real")
                ax.set_ylabel("Preço Previsto")
                ax.set_title("Comparação: Preço Real vs Preço Previsto")
                ax.legend()
                st.pyplot(fig)

                st.info("👉 Se os pontos estiverem próximos da linha vermelha tracejada, significa que o modelo faz boas previsões.")

        else:
            st.warning("Poucos dados disponíveis para treinar o modelo.")
    else:
        st.info("Selecione ao menos uma variável para treinar o modelo.")

