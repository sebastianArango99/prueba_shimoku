from os import getenv
import shimoku_api_python as Shimoku
import pandas as pd
from joblib import load


access_token = getenv('SHIMOKU_TOKEN')
universe_id: str = getenv('UNIVERSE_ID')
workspace_id: str = getenv('WORKSPACE_ID')

s = Shimoku.Client(
    access_token=access_token,
    universe_id=universe_id,
    verbosity='INFO'
)
s.set_workspace(uuid=workspace_id)

s.set_board('Algoritmo de Lead Scoring')

s.set_menu_path('Resultados')

df=pd.read_csv("/mnt/c/Users/Sebastián/OneDrive/Desktop/testShimoku.csv")

df2=pd.read_csv("/mnt/c/Users/Sebastián/OneDrive/Desktop/data_original.csv")

model = load('/mnt/c/Users/Sebastián/OneDrive/Desktop/lead_classification_v3.joblib')

df3=pd.read_csv('/mnt/c/Users/Sebastián/OneDrive/Desktop/shap.csv')


results=pd.DataFrame(model.predict_proba(df)[:,1], columns=['probabilidad'])


def classify_probability(proba):
    if proba > 0.75:
        return 'Alta Probabilidad'
    elif proba > 0.5:
        return 'Probabilidad Media'
    else:
        return 'Probabilidad Baja'

results['result'] = results['probabilidad'].apply(classify_probability)

results= pd.concat([df2,results], axis=1)

results_agrupado= results.groupby('result')['probabilidad'].count().reset_index()
s.plt.doughnut(data=results_agrupado, names='result', values='probabilidad', order=2,rows_size=4, cols_size=6, title='Distribución de categoría de lead')


results_agrupado_ciudad= results.groupby('City')['probabilidad'].mean().reset_index()
print(results_agrupado_ciudad.columns)
s.plt.horizontal_bar(data=results_agrupado_ciudad, x='City', order=3,rows_size=4, cols_size=6, title='Probabilidad de compra por ciudad')

s.plt.horizontal_bar(data=df3, x='Feature', order=4,rows_size=4, cols_size=12, title='Valores Shap del modelo, importancia de variables para una clasificación')

resultados= results.loc[:,['Id','probabilidad','result']]

s.plt.table(data=resultados, order=5,label_columns = { 
     'result': {
          'Probabilidad Alta' :'green',
          'Probabilidad Media' : 'yellow',
          'Probabilidad Baja' : 'red'
     }
})

s.plt.html(
    html=(
    "<article>"
    "<h4>¿Qué se puede observar?</h4>"
    "<ol>"
    "<li>Se puede observar que la gran mayoría de los leads tienen una baja probabilidad de convertirse en clientes, esto es de esperarse incluso viendo la diferencia de tamaños entre los datasets de leads y de offers.</li>"
    "<li>Se puede ver una diferencia entre la probabilidad de compra y la ciudad. Como próximo paso sería interesante enteder porqué se presenta esa diferencia.</li>"
    "<li>En cuanto a la importancia podemos ver que no tener ofertas es una de las causales que reduce en mayor medida la probabilidad de compra y lo mismo tener un descuento.</li>"
    "<li>Por otro lado, buscar a la empresa por evento corporativo y que tu dolor sea operaciones aumenta en mayor proporción la probabilidad de compra.</li>"
    "</ol>"
    "</article>"
    ),
    order=6, rows_size=1, cols_size=12,
)



#s.plt.delete_chart_by_order(order=1)

s.set_menu_path('Modelo')

s.plt.html(
    html=(
    "<article>"
    "<h4>¿Qué modelo se escogió?</h4>"
    "<p>El modelo que se escogió es un XGBoost, dentro de 3 diferentes que se probaron.<p>"
    "<ol>"
    "<h4>Pasos para construir el modelo</h4>"
    "<ol>"
    "<li>Exploración de datos inicial con el uso de Pandas Profiling.</li>"
    "<li>Unión de los diferentes datasets mediante la columna de ID, eliminando registros sin columna de ID dado que no podemos identificar si el lead tuvo oferta o no.</li>"
    "<li>Creación de pipeline de limpieza de datos, feature engineering y selección del modelo con validación cruzada. Se evalúan Random Forest, LightGBM y XGBoost, que son algoritmos con muy buenos resultados en tareas de clasificación.</li>"
    "<li>Generación de predicciones en el dataset de prueba con el mejor modelo, simulando datos nuevos en el modelo de clasificación construido.</li>"
    "<li>Uso de Shap Values para generar la explicabilidad del modelo y entender el porqué de una decisión del mismo.</li>"
    "</ol>"
    "<h4>Pasos para la limpieza y ingeniería de variables:</h4>"
    "<ol>"
    "<li>Unir datasets mediante columna de Id, eliminando registros que tengan datos faltantes en la columna de Id</li>"
    "<li>Para variables numèricas se imputan los valores faltantes mediante la mediana de la feature. Se utiliza la mediana dado que la mayorìa de los features presentan un sesgo importante y por lo tanto, si utilizamos la media cargaríamos ese sesgo</li>"
    "<li>Para variables categóricas se imputan valores faltantes con la moda de la feature.</li>"
    "<li>Se crea una variable adicional de tiempo que es la duración entre la creación del lead hasta el cierre del contrato.</li>"
    "<li>Se crea una variable binaria de descuento, 1 si tiene un descuento y 0 de lo contrario</li>"
    "<li>Removemos variables que se llenan a posteriori como status de oferta y conversión del lead puesto que son variables que se llenan cuando ya se conoce si el cliente seguirá en el proceso de compra o no.</li>"
    "<li>Removemos variables de fechas, puesto que no tienen tanta importancia para el modelo de lead scoring.</li>"
    "</ol>"
    "<p>Ahora veremos los resultados del modelo, mediante la matriz de confusión:<p>"
    "</article>"
    ),
    order=2, rows_size=1, cols_size=12,
)

data=[
    {
        "Predicción": "Compra",
        "Real": "No Compra",
        "Valor": 28
    },
    {
        "Predicción": "Compra",
        "Real": "Compra",
        "Valor": 366
    },
    {
        "Predicción": "No Compra",
        "Real": "Compra",
        "Valor": 0
    },
    {
        "Predicción": "No Compra",
        "Real": "No Compra",
        "Valor": 8406
    }]

s.plt.heatmap(
    data=data, title='Matriz de Confusión',
    x='Predicción', y='Real', 
    values='Valor', order=3,x_axis_name="Predicción", y_axis_name="Real", color_range=(0,8500)
)

s.plt.html(
    html=(
    "<article>"
    "<ol>"
    "<p>Se puede observar de la matriz de confusión previa los buenos resultados del modelo de XGBoost para clasificar leads. Se puede observar que clasifica correctamente los verdaderos positivos y los verdadero negativos.<p>"
    "</article>"
    ),
    order=4, rows_size=1, cols_size=12,
)

data = [
    {'Categoría': 'No Compra', 'Precisión': 1, 'Recall': 1, 'F1': 1},
    {'Categoría': 'Compra', 'Precisión': 0.93, 'Recall': 1, 'F1': 0.96}
]

s.plt.table(data=data, order=5)

s.plt.html(
    html=(
    "<article>"
    "<ol>"
    "<p>Se puede observar de los resultados unos excelentes resultados. La precisión es probablemente una de las métricas más importantes a revisar dado el supuesto que hay una inversión grande de capital en el desarrollo de un cliente. <p>"
    "</article>"
    ),
    order=6, rows_size=1, cols_size=12,
)
s.set_menu_path('Conclusiones')
s.plt.html(
    html=(
        "<h4>Conclusiones</h4>"
        "<p>En general se realizó una limpieza de los datos, dada la gran cantidad de datos faltantes se tuvo que remover una gran parte de los datos. Sin embargo, con los datos faltantes logramos obtener un data set para el enetrenamiento del modelo. Se imputaron valores faltantes, y se codificaron a través de one hot encoding las variables categóricas para el procesamiento por parte del modelo. Posteriormente entrenamos el modelo y obtuvimos los resultados.</p>"
    ),
    order=2, rows_size=1, cols_size=12,
)

s.run()
