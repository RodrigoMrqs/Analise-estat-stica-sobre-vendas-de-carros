# Analise-estat-stica-sobre-vendas-de-carros
Projeto relacionado a matéria de modelagem estatística com o fim de incentivar alunos a aprimorarem suas habilidades com bibliotecas de análise de dados e trabalharem seu pensamento e análise crítica sobre os dados obtidos.

# Sobre o dashboard
Nosso dashboard trata de carros usados e suas características como: quilometragem, modelo, marca, tamanho do motor, idade, etc. A partir desses, realizamos análises e previsões para preços futuros, depreciação e tendências de mercado, permitindo tanto usuários comuns quanto vendedores formarem uma opinião embasada.

# Estrutura do projeto
O projeto deve conter os seguintes arquivos:
    car_sales_data.csv
    dashboard.py
    main.ipynb
    README.md
    requirements.txt

# Instalação 
Primeiramente deve ser feito um clone do repositório no seu ambiente local

    git clone <url>
    cd <projeto>

Crie um ambiente virtual (opcional)

    python -m venv venv
    source venv/bin/activate   # Linux/Mac
    venv\Scripts\activate      # Windows

Em seguida, os requerimentos que estão no arquivos requeriment.txt devem ser baixados

    pip install -r requirements.txt

Por mim, execute o comando para iniciar o dashboard por meio do streamlit

    streamlit run ./dashboard.py
