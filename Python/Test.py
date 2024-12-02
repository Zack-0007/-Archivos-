import plotly.express as px
from sklearn.datasets import load_iris
import pandas as pd

# Carregando o conjunto de dados Iris
iris = load_iris()

# Convertendo os dados para um DataFrame para facilitar a manipulação
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['Classe'] = iris.target
data['Classe'] = data['Classe'].apply(lambda x: iris.target_names[x])  # Mapeando os rótulos

# Criando o gráfico 3D interativo
fig = px.scatter_3d(
    data,
    x=iris.feature_names[0],  # Comprimento da sépala
    y=iris.feature_names[1],  # Largura da sépala
    z=iris.feature_names[2],  # Comprimento da pétala
    color='Classe',  # Diferenciação por classe
    title='Gráfico 3D do Conjunto de Dados Iris',
    labels={'Classe': 'Classe da Flor'}
)

# Configurando o layout do gráfico
fig.update_layout(
    scene=dict(
        xaxis_title="Comprimento da Sépala (cm)",
        yaxis_title="Largura da Sépala (cm)",
        zaxis_title="Comprimento da Pétala (cm)",
    ),
    legend_title="Classes",
)

# Exibindo o gráfico interativo
fig.show()
