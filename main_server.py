# SAMPLE_SIZE_API
from flask import Flask, request, jsonify
from statsmodels.stats.power import TTestIndPower, NormalIndPower, FTestAnovaPower, FTestPower
import numpy as np
import math
from statsmodels.stats.power import TTestPower # Para la prueba t de datos pareados
from statsmodels.stats.power import GofChisquarePower # Prueba Chi-cuadrado para Tablas de Contingencia 
import scipy.stats as stats  # para df chi 
from scipy.stats import chi2, norm
from scipy.optimize import brentq

# BASIC_ANALYSIS_API
# from flask import Flask, request, jsonify  # Ya importado
# import numpy as np  # Ya importado
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from scipy.stats import kurtosis, skew

# GENERATE_CHARTS_API
# from flask import Flask, request, jsonify  # Ya importado
# import numpy as np  # Ya importado
# import pandas as pd  # Ya importado
# import matplotlib.pyplot as plt  # Ya importado
import seaborn as sns
# import io  # Ya importado
# import base64  # Ya importado

# RUN_REGRESSION_API
# from flask import Flask, request, jsonify  # Ya importado
# import pandas as pd  # Ya importado
# import numpy as np  # Ya importado
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import shapiro
# import io  # Ya importado
from statsmodels.stats.outliers_influence import variance_inflation_factor

# STAT TEST
import scipy.stats as stats
from scipy.stats import chi2_contingency
from scipy.stats import chisquare
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import f_oneway
import statsmodels.stats.multicomp as mc
from scipy.stats import friedmanchisquare
from scipy.stats import fisher_exact
from statsmodels.stats.contingency_tables import cochrans_q
from statsmodels.stats.contingency_tables import mcnemar
import scikit_posthocs as sp  # Importar la librería correcta

# MODELS
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score

# Cambiar el backend de matplotlib para evitar problemas de hilos en entornos de servidor
plt.switch_backend('Agg')

app = Flask(__name__)

###########################################################################################
####################################### SAMPLE SIZE #######################################
###########################################################################################

# Ruta para el cálculo de tamaño muestral para comparación de medias
@app.route('/calculate_sample_size', methods=['POST'])
def calculate_sample_size():
    try:
        data = request.get_json()

        # Obtener los parámetros de la solicitud
        alpha = data.get('alpha', 0.05)  # Nivel de significancia
        power = data.get('power', 0.8)  # Potencia estadística
        alternative = data.get('alternative', 'two-sided')  # Tipo de prueba ('two-sided', 'larger', 'smaller')

        effect_size = data.get('effect_size', None)  # Tamaño del efecto (Cohen's d), puede ser None
        mean1 = data.get('mean1', None)  # Media del grupo 1
        mean2 = data.get('mean2', None)  # Media del grupo 2
        std_dev_group1 = data.get('std_dev_group1', None)  # Desviación estándar del grupo 1
        std_dev_group2 = data.get('std_dev_group2', None)  # Desviación estándar del grupo 2

        # Si no se proporciona el tamaño del efecto, pero sí las medias y desviaciones estándar
        if effect_size is None:
            if mean1 is not None and mean2 is not None and std_dev_group1 is not None and std_dev_group2 is not None:
                # Calcular la desviación estándar combinada (pooled standard deviation)
                pooled_std_dev = math.sqrt((std_dev_group1 ** 2 + std_dev_group2 ** 2) / 2)
                # Calcular el tamaño del efecto
                effect_size = abs(mean1 - mean2) / pooled_std_dev
            else:
                raise ValueError("Si no se proporciona el tamaño del efecto, se deben proporcionar mean1, mean2, std_dev_group1 y std_dev_group2.")

        # Configurar el análisis de poder para prueba t de medias independientes
        analysis = TTestIndPower()

        # Calcular el tamaño muestral necesario
        sample_size = analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power, alternative=alternative)

        if isinstance(sample_size, np.ndarray):
            sample_size = sample_size.item()  # Convertir a escalar si es un array de un solo valor

        return jsonify({'sample_size': round(sample_size), 'effect_size': effect_size})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Nueva ruta para el cálculo de tamaño muestral para comparación de proporciones
@app.route('/calculate_sample_size_proportion', methods=['POST'])
def calculate_sample_size_proportion():
    try:
        data = request.get_json()

        # Obtener los parámetros de la solicitud
        alpha = data.get('alpha', 0.05)  # Nivel de significancia
        power = data.get('power', 0.8)  # Potencia estadística
        p1 = data.get('p1', 0.5)  # Proporción esperada en el grupo 1
        p2 = data.get('p2', 0.5)  # Proporción esperada en el grupo 2
        alternative = data.get('alternative', 'two-sided')  # Tipo de prueba ('two-sided', 'larger', 'smaller')

        # Calcular el tamaño del efecto utilizando Cohen's h
        effect_size = 2 * (math.asin(math.sqrt(p1)) - math.asin(math.sqrt(p2)))

        # Configurar el análisis de poder para prueba de proporciones independientes
        analysis = NormalIndPower()

        # Calcular el tamaño muestral necesario
        sample_size = analysis.solve_power(effect_size=abs(effect_size), alpha=alpha, power=power, alternative=alternative)

        if isinstance(sample_size, np.ndarray):
            sample_size = sample_size.item()  # Convertir a escalar si es un array de un solo valor

        return jsonify({'sample_size': round(sample_size), 'effect_size': effect_size})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Nueva ruta para el cálculo de tamaño muestral para ANOVA
@app.route('/calculate_sample_size_anova', methods=['POST'])
def calculate_sample_size_anova():
    try:
        data = request.get_json()

        # Obtener los parámetros de la solicitud
        alpha = data.get('alpha', 0.05)  # Nivel de significancia
        power = data.get('power', 0.8)  # Potencia estadística
        effect_size = data.get('effect_size', 0.25)  # Tamaño del efecto (f de Cohen)
        k = data.get('num_groups', 3)  # Número de grupos (mínimo 2)

        # Configurar el análisis de poder para ANOVA (one-way)
        analysis = FTestAnovaPower()

        # Calcular el tamaño muestral necesario
        sample_size = analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power, k_groups=k)

        if isinstance(sample_size, np.ndarray):
            sample_size = sample_size.item()  # Convertir a escalar si es un array de un solo valor

        return jsonify({'sample_size': round(sample_size), 'effect_size': effect_size})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Nueva ruta para el cálculo de tamaño muestral para regresión lineal
@app.route('/calculate_sample_size_regression', methods=['POST'])
def calculate_sample_size_regression():
    try:
        data = request.get_json()

        # Obtener los parámetros de la solicitud
        alpha = data.get('alpha', 0.05)  # Nivel de significancia
        power = data.get('power', 0.8)  # Potencia estadística
        effect_size = data.get('effect_size', None)  # Tamaño del efecto (f²)
        num_predictors = data.get('num_predictors', 1)  # Número de predictores en el modelo

        if effect_size is None:
            raise ValueError("El tamaño del efecto (f²) es requerido para el cálculo de regresión lineal.")

        # Configurar el análisis de poder para regresión lineal
        analysis = FTestPower()

        # Calcular el tamaño muestral necesario
        sample_size = analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power, df_num=num_predictors)

        if isinstance(sample_size, np.ndarray):
            sample_size = sample_size.item()  # Convertir a escalar si es un array de un solo valor

        return jsonify({'sample_size': round(sample_size), 'effect_size': effect_size})

    except Exception as e:
        return jsonify({'error': str(e)}), 400
    

# Nueva ruta para el cálculo de tamaño muestral para Two-Way ANOVA
@app.route('/calculate_sample_size_two_way_anova', methods=['POST'])
def calculate_sample_size_two_way_anova():
    try:
        data = request.get_json()

        # Obtener los parámetros de la solicitud
        alpha = data.get('alpha', 0.05)
        power = data.get('power', 0.8)
        effect_size = data.get('effect_size', 0.25)
        k1 = data.get('num_groups_factor1', 2)  # Número de niveles para el primer factor
        k2 = data.get('num_groups_factor2', 2)  # Número de niveles para el segundo factor

        # Configurar el análisis de poder para Two-Way ANOVA
        analysis = FTestAnovaPower()

        # Calcular el tamaño muestral necesario
        sample_size = analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power, k_groups=k1 * k2)

        if isinstance(sample_size, np.ndarray):
            sample_size = sample_size.item()  # Convertir a escalar si es un array de un solo valor

        return jsonify({'sample_size': round(sample_size), 'effect_size': effect_size})

    except Exception as e:
        return jsonify({'error': str(e)}), 400
    

# Nueva ruta para el cálculo de tamaño muestral para t-test de datos pareados
@app.route('/calculate_sample_size_paired_ttest', methods=['POST'])
def calculate_sample_size_paired_ttest():
    try:
        data = request.get_json()

        # Obtener los parámetros de la solicitud
        alpha = data.get('alpha', 0.05)  # Nivel de significancia
        power = data.get('power', 0.8)  # Potencia estadística
        alternative = data.get('alternative', 'two-sided')  # Tipo de prueba ('two-sided', 'larger', 'smaller')

        effect_size = data.get('effect_size', None)  # Tamaño del efecto (Cohen's d), puede ser None
        mean1 = data.get('mean1', None)  # Media del grupo 1
        mean2 = data.get('mean2', None)  # Media del grupo 2
        std_dev_group1 = data.get('std_dev_group1', None)  # Desviación estándar del grupo 1
        std_dev_group2 = data.get('std_dev_group2', None)  # Desviación estándar del grupo 2

        # Si no se proporciona el tamaño del efecto, pero sí las medias y desviaciones estándar
        if effect_size is None:
            if mean1 is not None and mean2 is not None and std_dev_group1 is not None and std_dev_group2 is not None:
                # Calcular la desviación estándar combinada (pooled standard deviation)
                pooled_std_dev = math.sqrt((std_dev_group1 ** 2 + std_dev_group2 ** 2) / 2)
                # Calcular el tamaño del efecto
                effect_size = abs(mean1 - mean2) / pooled_std_dev
            else:
                raise ValueError("Si no se proporciona el tamaño del efecto, se deben proporcionar mean1, mean2, std_dev_group1 y std_dev_group2.")

        # Configurar el análisis de poder para prueba t de datos pareados
        analysis = TTestPower()

        # Calcular el tamaño muestral necesario
        sample_size = analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power, alternative=alternative)

        if isinstance(sample_size, np.ndarray):
            sample_size = sample_size.item()

        return jsonify({'sample_size': round(sample_size), 'effect_size': effect_size})

    except Exception as e:
        return jsonify({'error': str(e)}), 400
    
# Nueva ruta para el cálculo de tamaño muestral para prueba Chi-cuadrado
@app.route('/calculate_sample_size_chi_square', methods=['POST'])
def calculate_sample_size_chi_square():
    try:
        data = request.get_json()

        # Obtener los parámetros de la solicitud
        alpha = data.get('alpha', 0.05)  # Nivel de significancia
        power = data.get('power', 0.8)  # Potencia estadística
        effect_size = data.get('effect_size', 0.3)  # Tamaño del efecto (w)
        df = data.get('df', None)  # Grados de libertad

        if df is None:
            raise ValueError("Los grados de libertad (df) son necesarios para el cálculo de Chi-cuadrado.")

        # Definir la función para calcular el poder basado en el tamaño de muestra
        def power_function(sample_size):
            non_centrality_param = sample_size * (effect_size ** 2)
            critical_value = chi2.ppf(1 - alpha, df)
            achieved_power = 1 - chi2.cdf(critical_value, df, non_centrality_param)
            return achieved_power - power

        # Usar brentq para resolver la ecuación y encontrar el tamaño de muestra adecuado
        sample_size = brentq(power_function, 1, 10000)  # Búsqueda entre 1 y 10,000

        return jsonify({'sample_size': round(sample_size), 'effect_size': effect_size})

    except Exception as e:
        return jsonify({'error': str(e)}), 400
    
@app.route('/calculate_sample_size_proportions_fisher', methods=['POST'])
def calculate_sample_size_fisher():
    try:
        data = request.get_json()

        # Obtener los parámetros de la solicitud
        alpha = data.get('alpha', 0.05)  # Nivel de significancia
        power = data.get('power', 0.8)  # Potencia estadística
        p1 = data.get('p1', 0.5)  # Proporción esperada en el grupo 1
        p2 = data.get('p2', 0.5)  # Proporción esperada en el grupo 2
        alternative = data.get('alternative', 'two-sided')  # Tipo de prueba ('two-sided', 'larger', 'smaller')

        # Calcular el tamaño del efecto utilizando un enfoque aproximado (Cohen's h)
        effect_size = 2 * (math.asin(math.sqrt(p1)) - math.asin(math.sqrt(p2)))

        # Utilizar NormalIndPower como aproximación
        analysis = NormalIndPower()
        sample_size = analysis.solve_power(effect_size=abs(effect_size), alpha=alpha, power=power, alternative=alternative)

        if isinstance(sample_size, np.ndarray):
            sample_size = sample_size.item()

        return jsonify({'sample_size': round(sample_size), 'effect_size': effect_size})

    except Exception as e:
        return jsonify({'error': str(e)}), 400


# Nueva ruta para el cálculo de tamaño muestral para la prueba de McNemar
@app.route('/calculate_sample_size_proportions_mcnemar', methods=['POST'])
def calculate_sample_size_mcnemar():
    try:
        data = request.get_json()

        # Obtener los parámetros de la solicitud
        alpha = data.get('alpha', 0.05)  # Nivel de significancia
        power = data.get('power', 0.8)  # Potencia estadística
        odds_ratio = data.get('odds_ratio', 1.5)  # Odds ratio esperado
        prop_discordant_pairs = data.get('prop_discordant_pairs', 0.3)  # Proporción de pares discordantes
        alternative = data.get('alternative', 'two-sided')  # Tipo de prueba ('two-sided', 'larger', 'smaller')

        # Cálculo de la potencia para McNemar basado en odds ratio y pares discordantes
        # Calcular el efecto a partir del odds ratio y la proporción de discordancias
        p12 = prop_discordant_pairs / (1 + odds_ratio)
        p21 = prop_discordant_pairs - p12
        effect_size = abs(p12 - p21)

        # Utilizar NormalIndPower como aproximación para la potencia
        z_alpha = norm.ppf(1 - alpha / 2) if alternative == 'two-sided' else norm.ppf(1 - alpha)
        z_beta = norm.ppf(power)

        # Calcular tamaño de muestra con la fórmula para McNemar
        n = (z_alpha + z_beta)**2 * (p12 + p21) / (p12 - p21)**2

        if isinstance(n, np.ndarray):
            n = n.item()

        return jsonify({'sample_size': round(n), 'effect_size': effect_size})

    except Exception as e:
        return jsonify({'error': str(e)}), 400
    

# Ruta para el cálculo de tamaño muestral para la correlación de Pearson
@app.route('/calculate_sample_size_pearson', methods=['POST'])
def calculate_sample_size_pearson():
    try:
        data = request.get_json()

        # Obtener los parámetros de la solicitud
        alpha = data.get('alpha', 0.05)  # Nivel de significancia
        power = data.get('power', 0.8)  # Potencia estadística
        r = data.get('coef_pearson', None)  # Tamaño del efecto (coeficiente de correlación)
        alternative = data.get('alternative', 'two-sided')  # Tipo de prueba ('two-sided', 'larger', 'smaller')

        if r is None:
            raise ValueError("El tamaño del efecto (coeficiente de correlación r) es necesario para el cálculo.")

        # Verificar que el coeficiente de correlación esté en el rango válido
        if not -1 <= r <= 1:
            raise ValueError("El coeficiente de correlación (r) debe estar entre -1 y 1.")

        # Calcular el tamaño del efecto para el análisis de poder
        effect_size = math.sqrt(r**2 / (1 - r**2))

        # Configurar el análisis de poder para la correlación de Pearson
        analysis = NormalIndPower()

        # Calcular el tamaño muestral necesario
        sample_size = analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power, alternative=alternative)

        if isinstance(sample_size, float):
            sample_size = math.ceil(sample_size)  # Redondear hacia arriba para asegurar tamaño suficiente

        return jsonify({'sample_size': sample_size, 'effect_size': effect_size})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

    


####################################################################################################
####################################### DESCRIPTIVE ANALYSIS #######################################
####################################################################################################

# Ruta para el análisis descriptivo básico a partir de una lista de datos proporcionada
@app.route('/calculate_basic_analysis', methods=['POST'])
def calculate_basic_analysis():
    try:
        data = request.get_json()
        data_list = data.get('data', None)

        if data_list is None or not isinstance(data_list, list):
            raise ValueError("Los datos proporcionados no son válidos o no se encuentran en el formato adecuado.")

        # Convertir los datos a una Serie de Pandas
        data_series = pd.Series(data_list)

        # Calcular estadísticas descriptivas
        result = calculate_descriptive_statistics(data_series)

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Variable global para almacenar el DataFrame cargado
dataframe = None

# Ruta para cargar un archivo y almacenar el DataFrame
@app.route('/upload_csv_descriptive', methods=['POST'])
def upload_file_descriptive():
    global dataframe
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No se encontró el archivo.'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'El archivo no tiene nombre.'}), 400

        # Determinar el tipo de archivo y leerlo en un DataFrame de Pandas
        filename = file.filename.lower()
        if filename.endswith('.csv'):
            dataframe = pd.read_csv(io.StringIO(file.stream.read().decode("utf-8")))
        elif filename.endswith(('.xls', '.xlsx')):
            dataframe = pd.read_excel(io.BytesIO(file.read()))  # Leer archivos Excel desde BytesIO
        elif filename.endswith('.txt'):
            dataframe = pd.read_csv(io.StringIO(file.stream.read().decode("utf-8")), delimiter=r'\s+')
        else:
            raise ValueError("Formato de archivo no soportado. Proporcione un archivo CSV, XLSX o TXT.")

        # Devolver las columnas disponibles para análisis
        columns = dataframe.columns.tolist()
        return jsonify({'message': 'Archivo cargado exitosamente', 'columns': columns})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Ruta para el análisis descriptivo de las columnas seleccionadas
@app.route('/analyze_selected_columns', methods=['POST'])
def analyze_selected_columns():
    global dataframe
    try:
        if dataframe is None:
            return jsonify({'error': 'No se ha cargado ningún archivo para analizar.'}), 400

        data = request.get_json()
        selected_columns = data.get('columns', [])

        if not selected_columns:
            raise ValueError("No se proporcionaron columnas para analizar.")

        result = {}
        for column in selected_columns:
            if column not in dataframe.columns:
                raise ValueError(f"La columna '{column}' no se encuentra en el DataFrame.")

            if pd.api.types.is_numeric_dtype(dataframe[column]):
                # Análisis para columnas numéricas
                data_series = dataframe[column]
                result[column] = calculate_descriptive_statistics(data_series, title=f'Histograma de {column}')
            else:
                # Análisis para columnas categóricas
                value_counts = dataframe[column].value_counts()
                percentages = dataframe[column].value_counts(normalize=True) * 100
                frequencies = value_counts.to_dict()
                percentages = percentages.to_dict()

                # Crear gráfico de barra horizontal única dividida
                labels = list(percentages.keys())
                values = list(percentages.values())
                colors = plt.get_cmap('Pastel1').colors  # Cambiar a colores más claros usando el colormap 'Pastel1'

                fig, ax = plt.subplots(figsize=(10, 3.5))  # Aumentar el alto del gráfico para más espacio para etiquetas
                left = 0

                # Dibuja cada sector con su porcentaje correspondiente
                for i, (label, value) in enumerate(zip(labels, values)):
                    ax.barh([0], value, left=left, color=colors[i % len(colors)], edgecolor='black')
                    left += value
                    # Ajustar el tamaño del texto y posición para que sea más visible
                    if value > 5:  # Solo mostramos las etiquetas si la sección tiene al menos 5% del ancho
                        # Mostrar el nombre de la categoría
                        ax.text(left - value / 2, 0.2, label, ha='center', va='center', fontsize=14, color='black', fontweight='bold')
                        # Mostrar el porcentaje debajo del nombre
                        ax.text(left - value / 2, -0.3, f'({value:.1f}%)', ha='center', va='center', fontsize=14, color='black', fontweight='bold')

                # Configurar el gráfico
                ax.set_xlim(0, 100)
                ax.set_xlabel('Porcentaje (%)', fontsize=14)
                ax.set_yticks([])  # Ocultar etiquetas del eje Y
                ax.set_title(f'Distribución de {column}', fontsize=16, fontweight='bold')

                plt.tight_layout()

                # Guardar el gráfico en formato de imagen
                img = io.BytesIO()
                plt.savefig(img, format='png')  # Cambiar 'svg' a 'png', 'jpeg', etc., según sea necesario.
                img.seek(0)
                encoded_img = base64.b64encode(img.getvalue()).decode()
                plt.close()  # Cerrar la figura para liberar memoria

                # Almacenar el resultado del análisis en el diccionario de resultados
                result[column] = {
                    'frequencies': frequencies,
                    'percentages': percentages,
                    'segmented_bar_chart': encoded_img  # Imagen codificada en base64 del gráfico de barra única
                }


        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Función para calcular estadísticas descriptivas comunes
def calculate_descriptive_statistics(data_series, title='Histograma de Datos'):
    try:
        # Calcular estadísticas descriptivas
        mean = float(data_series.mean())
        median = float(data_series.median())
        mode = data_series.mode().tolist()  # Puede tener más de un valor
        mode = [float(m) if isinstance(m, (np.integer, np.floating)) else m for m in mode]
        std = float(data_series.std())
        variance = float(data_series.var())
        min_value = float(data_series.min())
        max_value = float(data_series.max())
        range_value = float(max_value - min_value)
        coef_var = (std / mean) * 100 if mean != 0 else None
        coef_var = float(coef_var) if coef_var is not None else None

        # Medidas de forma
        skewness = float(skew(data_series, nan_policy='omit'))
        kurt = float(kurtosis(data_series, nan_policy='omit'))

        # Medidas de posición
        q1 = float(data_series.quantile(0.25))
        q3 = float(data_series.quantile(0.75))
        p10 = float(data_series.quantile(0.10))
        p90 = float(data_series.quantile(0.90))

        # Identificación de outliers (rango intercuartílico - IQR)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = data_series[(data_series < lower_bound) | (data_series > upper_bound)].tolist()
        outliers = [float(o) if isinstance(o, (np.integer, np.floating)) else o for o in outliers]

        # Crear un histograma y convertirlo a base64
        plt.figure(figsize=(6, 4))
        plt.hist(data_series.dropna(), bins=10, color='skyblue', edgecolor='black')
        plt.title(title)
        plt.xlabel('Valor')
        plt.ylabel('Frecuencia')
        plt.tight_layout()

        img = io.BytesIO()
        plt.savefig(img, format='png')  # Cambiar 'svg' a 'png', 'jpeg', etc., según sea necesario.
        img.seek(0)
        encoded_img = base64.b64encode(img.getvalue()).decode()
        plt.close()  # Cerrar la figura para liberar memoria

        # Devolver los resultados en formato JSON
        return {
            'mean': mean,
            'median': median,
            'mode': mode,
            'std': std,
            'variance': variance,
            'min': min_value,
            'max': max_value,
            'range': range_value,
            'coef_var': coef_var,
            'skewness': skewness,
            'kurtosis': kurt,
            'q1': q1,
            'q3': q3,
            'p10': p10,
            'p90': p90,
            'iqr': iqr,
            'outliers': outliers,
            'histogram': encoded_img  # Imagen codificada en base64
        }
    except Exception as e:
        return {'error': str(e)}


###############################################################################################
####################################### GENERATE CHARTS #######################################
###############################################################################################

# Variable global para almacenar el DataFrame cargado
dataframe = None

# Ruta para cargar un archivo y almacenar el DataFrame
@app.route('/upload_csv_charts', methods=['POST'])
def upload_file_charts():
    global dataframe
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No se encontró el archivo.'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'El archivo no tiene nombre.'}), 400

        # Determinar el tipo de archivo y leerlo en un DataFrame de Pandas
        filename = file.filename.lower()

        if filename.endswith('.csv'):
            try:
                dataframe = pd.read_csv(io.StringIO(file.stream.read().decode("utf-8")), delimiter=',')
            except UnicodeDecodeError:
                dataframe = pd.read_csv(io.StringIO(file.stream.read().decode("ISO-8859-1")), delimiter=',')
            except pd.errors.ParserError:
                dataframe = pd.read_csv(io.StringIO(file.stream.read().decode("utf-8")), delimiter=';')

        elif filename.endswith(('.xls', '.xlsx')):
            try:
                # Intentar leer el archivo Excel (.xls o .xlsx) usando un flujo de bytes
                file_stream = io.BytesIO(file.read())
                dataframe = pd.read_excel(file_stream, engine='openpyxl')
            except ValueError as e:
                return jsonify({'error': f'Error al leer el archivo Excel: {str(e)}'}), 400
            except Exception as e:
                return jsonify({'error': f'Error desconocido al leer el archivo Excel: {str(e)}'}), 400

        elif filename.endswith('.txt'):
            try:
                dataframe = pd.read_csv(io.StringIO(file.stream.read().decode("utf-8")), delimiter=r'\s+')
            except UnicodeDecodeError:
                dataframe = pd.read_csv(io.StringIO(file.stream.read().decode("ISO-8859-1")), delimiter=r'\s+')
        else:
            return jsonify({'error': "Formato de archivo no soportado. Proporcione un archivo CSV, XLSX, XLS o TXT."}), 400

        # Verificar si el DataFrame se cargó correctamente
        if dataframe.empty:
            return jsonify({'error': 'El archivo está vacío o no se pudo procesar correctamente.'}), 400

        # Normalizar encabezados quitando espacios adicionales
        dataframe.columns = dataframe.columns.str.strip()

        # Manejo de celdas vacías
        dataframe = dataframe.fillna("N/A")  # Rellenar celdas vacías con "N/A" o el valor que sea más apropiado

        # Clasificar las columnas entre numéricas y categóricas
        numeric_columns = dataframe.select_dtypes(include=['number']).columns.tolist()
        categorical_columns = dataframe.select_dtypes(exclude=['number']).columns.tolist()

        return jsonify({
            'message': 'Archivo cargado exitosamente',
            'numeric_columns': numeric_columns,
            'categorical_columns': categorical_columns
        })

    except UnicodeDecodeError:
        return jsonify({'error': 'Error de codificación. Asegúrese de que el archivo esté en formato UTF-8 o ISO-8859-1.'}), 400
    except pd.errors.ParserError:
        return jsonify({'error': 'Error al analizar el archivo. Verifique el delimitador y el formato del archivo.'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Nueva ruta para la generación de gráficos a partir de las columnas seleccionadas
@app.route('/generate_charts', methods=['POST'])
def generate_charts():
    global dataframe
    try:
        if dataframe is None:
            return jsonify({'error': 'No se ha cargado ningún archivo para analizar.'}), 400

        data = request.get_json()
        x_column = data.get('x_column')
        y_column = data.get('y_column')
        chart_type = data.get('chart_type')
        categorical_column = data.get('categorical_column')

        if not x_column or not chart_type:
            return jsonify({'error': 'Por favor, seleccione las columnas y el tipo de gráfico.'}), 400

        # Limpiar los datos eliminando valores nulos
        df_clean = dataframe.dropna(subset=[x_column, y_column] if y_column else [x_column])

        # Generar el gráfico
        img = io.BytesIO()
        plt.figure(figsize=(8, 6))

        if chart_type == 'Scatterplot':
            # Scatterplot necesita dos columnas numéricas
            if not y_column:
                return jsonify({'error': 'Para un scatterplot, debe seleccionar dos variables numéricas.'}), 400
            
            if pd.api.types.is_numeric_dtype(dataframe[x_column]) and pd.api.types.is_numeric_dtype(dataframe[y_column]):
                sns.scatterplot(x=df_clean[x_column], y=df_clean[y_column], color='blue', alpha=0.6, edgecolor='w', s=80)
                plt.xlabel(x_column)
                plt.ylabel(y_column)
                plt.title(f'Scatterplot de {x_column} vs {y_column}')
                plt.grid(True)
            else:
                return jsonify({'error': 'Ambas columnas seleccionadas deben ser numéricas para un scatterplot.'}), 400

        elif chart_type == 'Histograma':
            # Histograma para una columna numérica
            if pd.api.types.is_numeric_dtype(dataframe[x_column]):
                sns.histplot(df_clean[x_column], bins=20, kde=True, color='skyblue')
                plt.xlabel(x_column)
                plt.ylabel('Frecuencia')
                plt.title(f'Histograma de {x_column}')
            else:
                return jsonify({'error': 'La columna seleccionada debe ser numérica para un histograma.'}), 400

        elif chart_type == 'Boxplot':
            # Boxplot para una columna numérica, opcionalmente diferenciada por una categórica
            if pd.api.types.is_numeric_dtype(dataframe[x_column]):
                if categorical_column and categorical_column in dataframe.columns:
                    if pd.api.types.is_categorical_dtype(dataframe[categorical_column]) or dataframe[categorical_column].dtype == 'object':
                        sns.boxplot(x=categorical_column, y=x_column, data=df_clean, palette='Set3')
                        plt.title(f'Boxplot de {x_column} según {categorical_column}')
                    else:
                        return jsonify({'error': 'La columna categórica seleccionada no es válida.'}), 400
                else:
                    sns.boxplot(x=df_clean[x_column], color='lightgreen')
                    plt.title(f'Boxplot de {x_column}')
                plt.xlabel(x_column)
            else:
                return jsonify({'error': 'La columna seleccionada debe ser numérica para un boxplot.'}), 400

        plt.tight_layout()
        plt.savefig(img, format='png')
        img.seek(0)
        encoded_img = base64.b64encode(img.getvalue()).decode()
        plt.close()

        return jsonify({'chart': encoded_img})

    except Exception as e:
        return jsonify({'error': str(e)}), 400



###############################################################################################
####################################### STAT TESTS DATA #######################################
###############################################################################################

# Variable global para almacenar el DataFrame cargado
dataframe = None

@app.route('/upload_csv_stat', methods=['POST'])
def upload_file_stat():
    global dataframe
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No se encontró el archivo.'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'El archivo no tiene nombre.'}), 400

        filename = file.filename.lower()

        # Determinar el tipo de archivo y leerlo en un DataFrame de Pandas
        try:
            if filename.endswith('.csv'):
                try:
                    dataframe = pd.read_csv(io.StringIO(file.stream.read().decode("utf-8")), delimiter=',')
                except UnicodeDecodeError:
                    dataframe = pd.read_csv(io.StringIO(file.stream.read().decode("ISO-8859-1")), delimiter=',')
                except pd.errors.ParserError:
                    dataframe = pd.read_csv(io.StringIO(file.stream.read().decode("utf-8")), delimiter=';')

            elif filename.endswith(('.xls', '.xlsx')):
                file_stream = io.BytesIO(file.read())
                dataframe = pd.read_excel(file_stream, engine='openpyxl')

            elif filename.endswith('.txt'):
                try:
                    dataframe = pd.read_csv(io.StringIO(file.stream.read().decode("utf-8")), delimiter=r'\s+')
                except UnicodeDecodeError:
                    dataframe = pd.read_csv(io.StringIO(file.stream.read().decode("ISO-8859-1")), delimiter=r'\s+')
            else:
                return jsonify({'error': "Formato de archivo no soportado. Proporcione un archivo CSV, XLSX, XLS o TXT."}), 400

            # Verificar si el DataFrame se cargó correctamente
            if dataframe.empty:
                return jsonify({'error': 'El archivo está vacío o no se pudo procesar correctamente.'}), 400

            # Normalizar encabezados quitando espacios adicionales y caracteres no deseados
            dataframe.columns = dataframe.columns.str.strip()

            # Manejo de celdas vacías
            dataframe = dataframe.replace("N/A", np.nan)
            dataframe = dataframe.dropna()  # Eliminamos las filas con valores nulos para evitar errores en la regresión

            # Clasificar las columnas entre numéricas y categóricas
            numeric_columns = dataframe.select_dtypes(include=['number']).columns.tolist()
            categorical_columns = dataframe.select_dtypes(exclude=['number']).columns.tolist()

            return jsonify({
                'message': 'Archivo cargado exitosamente',
                'numeric_columns': numeric_columns,
                'categorical_columns': categorical_columns
            })

        except Exception as e:
            return jsonify({'error': str(e)}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/run_regression', methods=['POST'])
def run_regression():
    global dataframe
    try:
        if dataframe is None:
            return jsonify({'error': 'No se ha cargado ningún archivo para analizar.'}), 400

        data = request.get_json()
        response_variable = data.get('response_variable')
        covariates = data.get('covariates')

        if not response_variable or not covariates:
            return jsonify({'error': 'Variables insuficientes para la regresión.'}), 400

        # Verificar que la variable de respuesta y las covariables existen en el DataFrame
        if response_variable not in dataframe.columns:
            return jsonify({'error': f'La variable de respuesta {response_variable} no existe en los datos.'}), 400

        for covariate in covariates:
            if covariate not in dataframe.columns:
                return jsonify({'error': f'La covariable {covariate} no existe en los datos.'}), 400

        warnings = []

        # Preparar X e y para la regresión
        X = dataframe[covariates]
        y = dataframe[response_variable]

        # Verificación de multicolinealidad utilizando el VIF
        X_with_const = sm.add_constant(X)
        vif_data = pd.DataFrame()
        vif_data["Feature"] = X_with_const.columns
        vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i) for i in range(X_with_const.shape[1])]

        # Agregar advertencia si hay covariables con VIF alto
        high_vif_features = vif_data[vif_data['VIF'] > 10]
        if not high_vif_features.empty:
            warnings.append('Existen variables con alta colinealidad. Considere eliminarlas.')

        # División de los datos en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=1234, shuffle=True)

        # A la matriz de predictores se le añade una columna de 1s para el intercept del modelo
        X_train = sm.add_constant(X_train)
        X_test = sm.add_constant(X_test)

        # Crear el modelo de regresión lineal
        model = sm.OLS(endog=y_train, exog=X_train).fit()

        # Verificación de normalidad de los residuos
        shapiro_test = shapiro(model.resid)
        if shapiro_test.pvalue < 0.05:
            warnings.append('Los residuos no parecen estar distribuidos de manera normal. Considere transformar las variables.')

        # Predicciones para el conjunto de prueba y cálculo del error RMSE
        predictions = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))

        # Resumen del modelo y resultados en formato JSON
        regression_result = {
            "coefficients": model.params.to_dict(),
            "p_values": model.pvalues.to_dict(),
            "r_squared": model.rsquared,
            "adj_r_squared": model.rsquared_adj,
            "rmse": rmse
        }

        # Devolver advertencias y resultados juntos
        return jsonify({
            'regression_result': regression_result,
            'warnings': warnings
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400
    


##########################################################################################
####################################### STAT TESTS #######################################
##########################################################################################

# Ruta para la prueba Shapiro-Wilk
@app.route('/api/shapiro', methods=['POST'])
def shapiro_test():
    data = request.get_json()
    sample = data['sample']
    stat, p_value = stats.shapiro(sample)

    # Determine significance of the p-value
    if p_value < 0.05:
        significance = "significant"
        reject_null = "Reject the null hypothesis"
    elif p_value < 0.1:
        significance = "marginally significant"
        reject_null = "Potential rejection of the null hypothesis"
    else:
        significance = "not significant"
        reject_null = "Do not reject the null hypothesis"

    return jsonify({
        'test': 'Shapiro-Wilk',
        'statistic': stat,
        'pValue': p_value,
        'significance': significance,
        'decision': reject_null
    })


# Ruta para la prueba Kolmogorov-Smirnov
@app.route('/api/kolmogorov', methods=['POST'])
def kolmogorov_test():
    data = request.get_json()
    sample = data['sample']

    sample_mean = np.mean(sample)
    sample_std = np.std(sample, ddof=1)
    
    stat, p_value = stats.kstest(sample, 'norm', args=(sample_mean, sample_std))

    # Determine significance of the p-value
    if p_value < 0.05:
        significance = "significant"
        reject_null = "Reject the null hypothesis"
    elif p_value < 0.1:
        significance = "marginally significant"
        reject_null = "Potential rejection of the null hypothesis"
    else:
        significance = "not significant"
        reject_null = "Do not reject the null hypothesis"

    return jsonify({
        'test': 'Kolmogorov-Smirnov',
        'statistic': stat,
        'pValue': p_value,
        'significance': significance,
        'decision': reject_null
    })


# Ruta para la prueba de Levene (homogeneidad de varianzas)
@app.route('/api/levene', methods=['POST'])
def levene_test():
    try:
        # Obtener los datos del JSON recibido
        data = request.get_json()
        group1 = data.get('group1', [])
        group2 = data.get('group2', [])
        group3 = data.get('group3', [])
        group4 = data.get('group4', [])
        group5 = data.get('group5', [])
        group6 = data.get('group6', [])
        group7 = data.get('group7', [])
        group8 = data.get('group8', [])
        group9 = data.get('group9', [])
        group10 = data.get('group10', [])

        # Convertir los datos de cada grupo a float
        group1 = [float(x) for x in group1]
        group2 = [float(x) for x in group2]
        group3 = [float(x) for x in group3]
        group4 = [float(x) for x in group4]
        group5 = [float(x) for x in group5]
        group6 = [float(x) for x in group6]
        group7 = [float(x) for x in group7]
        group8 = [float(x) for x in group8]
        group9 = [float(x) for x in group9]
        group10 = [float(x) for x in group10]

        # Filtrar los grupos no vacíos
        groups = [group for group in [group1, group2, group3, group4, group5, group6, group7, group8, group9, group10] if group]

        # Validar que haya al menos dos grupos no vacíos
        if len(groups) < 2:
            return jsonify({'error': 'Se requieren al menos dos grupos con datos para realizar la prueba de Levene.'}), 400

        # Realizar la prueba de Levene
        stat, p_value = stats.levene(*groups)

        # Determine significance of the p-value
        if p_value < 0.05:
            significance = "significant"
            reject_null = "Reject the null hypothesis"
        elif p_value < 0.1:
            significance = "marginally significant"
            reject_null = "Potential rejection of the null hypothesis"
        else:
            significance = "not significant"
            reject_null = "Do not reject the null hypothesis"

        # Resultados de la prueba de Levene
        levene_results = {
            'test': 'Levene',
            'statistic': stat,
            'pValue': p_value,
            'significance': significance,
            'decision': reject_null,
            'num_groups': len(groups),
            'total_observations': sum(len(group) for group in groups),
        }

        return jsonify(levene_results)

    except Exception as e:
        # Registrar el error en el servidor para depuración
        print(f'Error al ejecutar la prueba de Levene: {str(e)}')
        return jsonify({'error': f'Error interno del servidor: {str(e)}'}), 500

# Ruta para la prueba t de Student con opción pareado
@app.route('/api/ttest', methods=['POST'])
def t_test():
    data = request.get_json()
    sample1 = data['sample1']
    sample2 = data['sample2']
    paired = data.get('paired', False)
    
    if paired:
        stat, p_value = stats.ttest_rel(sample1, sample2)
    else:
        stat, p_value = stats.ttest_ind(sample1, sample2)

    # Determine significance of the p-value
    if p_value < 0.05:
        significance = "significant"
        reject_null = "Reject the null hypothesis"
    elif p_value < 0.1:
        significance = "marginally significant"
        reject_null = "Potential rejection of the null hypothesis"
    else:
        significance = "not significant"
        reject_null = "Do not reject the null hypothesis"

    return jsonify({
        'test': 'T-Test' + (' paired' if paired else ''),
        'statistic': stat,
        'pValue': p_value,
        'significance': significance,
        'decision': reject_null
    })


# Ruta para la prueba Chi-Square de bondad de ajuste
@app.route('/api/chi_square/goodness_of_fit', methods=['POST'])
def chi_square_goodness_of_fit():
    data = request.get_json()
    observed = data.get('observed', [])
    expected = data.get('expected', None)
    
    # Convertir observaciones a floats
    try:
        observed = [float(x) for x in observed]
        if expected:
            expected = [float(x) for x in expected]
    except ValueError:
        return jsonify({'error': 'Los datos proporcionados deben ser numéricos.'}), 400

    # Validar que observed y expected tengan la misma longitud si expected está presente
    if expected and len(observed) != len(expected):
        return jsonify({'error': 'Las frecuencias observadas y esperadas deben tener el mismo tamaño.'}), 400

    try:
        # Realizar la prueba Chi-Cuadrado
        if expected:
            stat, p_value = chisquare(f_obs=observed, f_exp=expected)
        else:
            stat, p_value = chisquare(f_obs=observed)
        
        # Retornar los resultados
        return jsonify({'test': 'Chi-Square (Bondad de Ajuste)', 'statistic': stat, 'pValue': p_value})
    except Exception as e:
        # Manejo de errores en la ejecución del test
        return jsonify({'error': f'Error interno del servidor: {str(e)}'}), 500



# Ruta para la prueba Chi-Square de independencia
@app.route('/api/chi_square/independence', methods=['POST'])
def chi_square_independence():
    data = request.get_json()
    observed = data['observed']
    stat, p_value, _, _ = stats.chi2_contingency(observed)

    # Determine significance of the p-value
    if p_value < 0.05:
        significance = "significant"
        reject_null = "Reject the null hypothesis"
    elif p_value < 0.1:
        significance = "marginally significant"
        reject_null = "Potential rejection of the null hypothesis"
    else:
        significance = "not significant"
        reject_null = "Do not reject the null hypothesis"

    return jsonify({
        'test': 'Chi-Square (Independence)',
        'statistic': stat,
        'pValue': p_value,
        'significance': significance,
        'decision': reject_null
    })



# Ruta para la prueba ANOVA de una vía
@app.route('/api/anova_one_way', methods=['POST'])
def anova_one_way():
    try:
        # Obtener los datos del JSON recibido
        data = request.get_json()
        group1 = data.get('group1', [])
        group2 = data.get('group2', [])
        group3 = data.get('group3', [])
        group4 = data.get('group4', [])
        group5 = data.get('group5', [])
        group6 = data.get('group6', [])
        group7 = data.get('group7', [])
        group8 = data.get('group8', [])
        group9 = data.get('group9', [])
        group10 = data.get('group10', [])
        multiple_comparisons = data.get('multipleComparisons', False)

        # Convertir los datos de cada grupo a float
        group1 = [float(x) for x in group1]
        group2 = [float(x) for x in group2]
        group3 = [float(x) for x in group3]
        group4 = [float(x) for x in group4]
        group5 = [float(x) for x in group5]
        group6 = [float(x) for x in group6]
        group7 = [float(x) for x in group7]
        group8 = [float(x) for x in group8]
        group9 = [float(x) for x in group9]
        group10 = [float(x) for x in group10]

        # Filtrar los grupos no vacíos
        groups = [group for group in [group1, group2, group3, group4, group5, group6, group7, group8, group9, group10] if group]

        # Validar que haya al menos dos grupos no vacíos
        if len(groups) < 2:
            return jsonify({'error': 'Se requieren al menos dos grupos con datos para realizar ANOVA.'}), 400

        # Realizar el ANOVA de una vía
        f_statistic, p_value = f_oneway(*groups)

        # Resultados básicos de ANOVA
        anova_results = {
            'F': f_statistic,
            'pValue': p_value,
            'num_groups': len(groups),
            'total_observations': sum(len(group) for group in groups),
            'anovaType': 'One way'
        }

        # Realizar comparaciones múltiples si está habilitado
        if multiple_comparisons:
            # Verificar que haya al menos tres grupos para hacer Tukey HSD
            if len(groups) < 3:
                return jsonify({'error': 'Se requieren al menos tres grupos para realizar comparaciones múltiples (Tukey HSD).'}), 400
            
            try:
                # Preparar los datos para Tukey HSD
                all_data = []
                labels = []
                for i, group in enumerate(groups):
                    # Verificar que cada grupo tenga al menos un valor
                    if len(group) == 0:
                        return jsonify({'error': f'El grupo {i + 1} no contiene datos suficientes.'}), 400

                    all_data.extend(group)
                    labels.extend([f'Grupo {i+1}'] * len(group))  # Cambia a 'Grupo {i+1}' para una mejor presentación
                
                # Convertir los datos a un DataFrame
                df = pd.DataFrame({'value': all_data, 'group': labels})
                
                # Realizar Tukey HSD
                tukey = mc.pairwise_tukeyhsd(df['value'], df['group'], alpha=0.05)

                # Procesar los resultados de Tukey HSD y generar texto formateado
                tukey_summary = "Comparaciones múltiples (Tukey HSD):\n"
                for result in tukey.summary().data[1:]:  # Ignorar la cabecera
                    tukey_summary += (
                        f"----------------------------\n"  # Separador para mayor claridad
                        f"Comparación: {result[0]} vs {result[1]}\n"
                        f"  • Diferencia de Medias: {result[2]:.2f}\n"
                        f"  • p-Value ajustado: {result[3]:.3f}\n"
                        f"  • IC Inferior: {result[4]:.2f}, IC Superior: {result[5]:.2f}\n"
                        f"  • Rechazo H0: {'Sí' if result[6] else 'No'}\n"
                    )

                # Agregar el resumen de Tukey al resultado
                anova_results['tukey'] = tukey_summary

            except Exception as e:
                return jsonify({'error': f'Error al ejecutar Tukey HSD: {str(e)}'}), 500

        return jsonify(anova_results)
    
    except Exception as e:
        # Registrar el error en el servidor para depuración
        print(f'Error al ejecutar ANOVA: {str(e)}')
        return jsonify({'error': f'Error interno del servidor: {str(e)}'}), 500



# Ruta para la prueba ANOVA de dos vías
@app.route('/anova_two_way', methods=['POST'])
def anova_two_way():
    try:
        # Obtener los datos del cuerpo de la solicitud
        data = request.get_json()

        # Extraer los vectores enviados desde Flutter
        factor1 = data['factor1']  # Variable categórica 1
        factor2 = data['factor2']  # Variable categórica 2
        values = data['values']    # Variable numérica (valores dependientes)

        # Asegurarse de que los vectores tengan la misma longitud
        if len(factor1) != len(factor2) or len(factor1) != len(values):
            return jsonify({'error': 'Los vectores factor1, factor2 y values deben tener la misma longitud.'}), 400

        # Crear un DataFrame con los datos, asegurando que factor1 y factor2 sean categóricos
        df = pd.DataFrame({
            'factor1': pd.Categorical(factor1),  # Convertir a categórico
            'factor2': pd.Categorical(factor2),  # Convertir a categórico
            'values': values                     # Continuo
        })

        # Realizar ANOVA de dos vías
        model = ols('values ~ C(factor1) + C(factor2) + C(factor1):C(factor2)', data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        
        # Reemplazar NaN e infinitos por 0 o por cualquier valor adecuado antes de convertir a JSON
        anova_table = anova_table.replace([np.inf, -np.inf, np.nan], 0)

        # Extraer valores de interés
        F_factor1 = anova_table.loc['C(factor1)', 'F']
        pValue_factor1 = anova_table.loc['C(factor1)', 'PR(>F)']
        F_factor2 = anova_table.loc['C(factor2)', 'F']
        pValue_factor2 = anova_table.loc['C(factor2)', 'PR(>F)']
        F_interaction = anova_table.loc['C(factor1):C(factor2)', 'F']
        pValue_interaction = anova_table.loc['C(factor1):C(factor2)', 'PR(>F)']

        # Devolver los resultados formateados
        return jsonify({
            'anovaType': 'Two way',
            'F_values': {
                'Factor1': F_factor1,
                'Factor2': F_factor2,
                'Interaction': F_interaction
            },
            'p_values': {
                'Factor1': pValue_factor1,
                'Factor2': pValue_factor2,
                'Interaction': pValue_interaction
            }
        })
    
    except Exception as e:
        return jsonify({'error': f'Error interno del servidor: {str(e)}'}), 500

# Mann Whitey U test
@app.route('/api/mannwhitney', methods=['POST'])
def mann_whitney():
    data = request.get_json()
    sample1 = data['sample1']
    sample2 = data['sample2']
    stat, p_value = stats.mannwhitneyu(sample1, sample2)

    # Determine significance of the p-value
    if p_value < 0.05:
        significance = "significant"
        reject_null = "Reject the null hypothesis"
    elif p_value < 0.1:
        significance = "marginally significant"
        reject_null = "Potential rejection of the null hypothesis"
    else:
        significance = "not significant"
        reject_null = "Do not reject the null hypothesis"

    return jsonify({
        'test': 'Mann-Whitney U',
        'statistic': stat,
        'pValue': p_value,
        'significance': significance,
        'decision': reject_null
    })



# Kruskal Wallis H test con comparaciones múltiples
@app.route('/api/kruskal', methods=['POST'])
def kruskal_wallis():
    try:
        data = request.get_json()
        # Obtener los grupos de datos
        group1 = data.get('group1', [])
        group2 = data.get('group2', [])
        group3 = data.get('group3', [])
        group4 = data.get('group4', [])
        group5 = data.get('group5', [])
        group6 = data.get('group6', [])
        group7 = data.get('group7', [])
        group8 = data.get('group8', [])
        group9 = data.get('group9', [])
        group10 = data.get('group10', [])
        multiple_comparisons = data.get('multipleComparisons', False)

        # Convertir los datos de cada grupo a float
        groups = [[float(x) for x in group] for group in [group1, group2, group3, group4, group5, group6, group7, group8, group9, group10] if group]

        # Validar que haya al menos dos grupos no vacíos
        if len(groups) < 2:
            return jsonify({'error': 'Se requieren al menos dos grupos con datos para realizar la prueba Kruskal-Wallis.'}), 400

        # Realizar la prueba Kruskal-Wallis
        stat, p_value = stats.kruskal(*groups)

        # Determine significance of the p-value
        if p_value < 0.05:
            significance = "significant"
            reject_null = "Reject the null hypothesis"
        elif p_value < 0.1:
            significance = "marginally significant"
            reject_null = "Potential rejection of the null hypothesis"
        else:
            significance = "not significant"
            reject_null = "Do not reject the null hypothesis"

        # Resultados básicos de Kruskal-Wallis
        kw_results = {
            'test': 'Kruskal-Wallis',
            'statistic': stat,
            'pValue': p_value,
            'significance': significance,
            'decision': reject_null,
            'num_groups': len(groups),
            'total_observations': sum(len(group) for group in groups)
        }

        # Comparaciones múltiples si está habilitado
        if multiple_comparisons:
            # Verificar que haya al menos tres grupos para realizar comparaciones múltiples
            if len(groups) < 3:
                return jsonify({'error': 'Se requieren al menos tres grupos para realizar comparaciones múltiples.'}), 400

            try:
                # Preparar los datos para las comparaciones múltiples
                all_data = []
                labels = []
                for i, group in enumerate(groups):
                    all_data.extend(group)
                    labels.extend([f'Grupo {i+1}'] * len(group))

                # Convertir a DataFrame para usar en las pruebas
                df = pd.DataFrame({'value': all_data, 'group': labels})

                # Realizar la prueba de Dunn
                dunn = sp.posthoc_dunn(df, val_col='value', group_col='group', p_adjust='bonferroni')

                # Procesar los resultados de Dunn
                dunn_summary = "Comparaciones múltiples (Dunn con corrección de Bonferroni):\n"
                for i in range(len(dunn)):
                    for j in range(i+1, len(dunn)):
                        dunn_summary += (
                            f"----------------------------\n"
                            f"Comparación: Grupo {i+1} vs Grupo {j+1}\n"
                            f"  • p-Value ajustado: {dunn.iloc[i, j]:.3f}\n"
                        )

                # Agregar el resumen de Dunn al resultado
                kw_results['dunn'] = dunn_summary

            except Exception as e:
                return jsonify({'error': f'Error al ejecutar la prueba de Dunn: {str(e)}'}), 500

        return jsonify(kw_results)

    except Exception as e:
        return jsonify({'error': f'Error interno del servidor: {str(e)}'}), 500


# Friedman Test
@app.route('/api/friedman', methods=['POST'])
def friedman_test():
    try:
        data = request.get_json()
        groups = data['groups']

        if len(groups) < 3:
            return jsonify({'error': 'At least 3 groups are required for the Friedman test.'}), 400

        num_observations = len(groups[0])
        if not all(len(g) == num_observations for g in groups):
            return jsonify({'error': 'All groups must have the same number of observations.'}), 400

        stat, p_value = stats.friedmanchisquare(*groups)

        # Determine significance of the p-value
        if p_value < 0.05:
            significance = "significant"
            reject_null = "Reject the null hypothesis"
        elif p_value < 0.1:
            significance = "marginally significant"
            reject_null = "Potential rejection of the null hypothesis"
        else:
            significance = "not significant"
            reject_null = "Do not reject the null hypothesis"

        return jsonify({
            'test': 'Friedman',
            'statistic': stat,
            'pValue': p_value,
            'significance': significance,
            'decision': reject_null,
            'num_groups': len(groups),
            'total_observations': num_observations * len(groups)
        })

    except Exception as e:
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500


# Fisher exact test
@app.route('/api/fisher', methods=['POST'])
def fisher_test():
    try:
        data = request.get_json()
        observed = data['observed']

        if len(observed) != 2 or len(observed[0]) != 2:
            return jsonify({'error': 'Fisher’s exact test requires a 2x2 contingency table.'}), 400

        oddsratio, p_value = stats.fisher_exact(observed)

        # Determine significance of the p-value
        if p_value < 0.05:
            significance = "significant"
            reject_null = "Reject the null hypothesis"
        elif p_value < 0.1:
            significance = "marginally significant"
            reject_null = "Potential rejection of the null hypothesis"
        else:
            significance = "not significant"
            reject_null = "Do not reject the null hypothesis"

        return jsonify({
            'test': 'Fisher',
            'oddsratio': oddsratio,
            'pValue': p_value,
            'significance': significance,
            'decision': reject_null
        })

    except Exception as e:
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500




# Mcnemar test
@app.route('/api/mcnemar', methods=['POST'])
def mcnemar_test():
    try:
        data = request.get_json()
        observed = data['observed']

        if len(observed) != 2 or len(observed[0]) != 2:
            return jsonify({'error': 'McNemar’s test requires a 2x2 contingency table.'}), 400

        result = mcnemar(observed, exact=True)

        # Determine significance of the p-value
        if result.pvalue < 0.05:
            significance = "significant"
            reject_null = "Reject the null hypothesis"
        elif result.pvalue < 0.1:
            significance = "marginally significant"
            reject_null = "Potential rejection of the null hypothesis"
        else:
            significance = "not significant"
            reject_null = "Do not reject the null hypothesis"

        return jsonify({
            'test': 'McNemar',
            'statistic': result.statistic,
            'pValue': result.pvalue,
            'significance': significance,
            'decision': reject_null
        })

    except Exception as e:
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500


# Cochran's Q Test
@app.route('/api/cochran', methods=['POST'])
def cochran_test():
    try:
        data = request.get_json()
        observed = data['observed']

        if len(observed[0]) < 3:
            return jsonify({'error': 'Cochran’s Q test requires at least 3 treatments/conditions.'}), 400

        # Ejecutar la prueba de Cochran
        result = cochrans_q(observed)

        # Determine significance of the p-value
        if result.pvalue < 0.05:
            significance = "significant"
            reject_null = "Reject the null hypothesis"
        elif result.pvalue < 0.1:
            significance = "marginally significant"
            reject_null = "Potential rejection of the null hypothesis"
        else:
            significance = "not significant"
            reject_null = "Do not reject the null hypothesis"

        return jsonify({
            'test': 'Cochran\'s Q',
            'statistic': result.statistic,
            'pValue': result.pvalue,
            'significance': significance,
            'decision': reject_null
        })

    except Exception as e:
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/api/wilcoxon', methods=['POST'])
def wilcoxon_test():
    try:
        data = request.get_json()
        sample1 = data['sample1']
        sample2 = data['sample2']

        # Asegurarse de que ambas muestras tengan el mismo número de observaciones
        if len(sample1) != len(sample2):
            return jsonify({'error': 'Las dos muestras deben tener el mismo número de observaciones para la prueba Wilcoxon.'}), 400

        # Ejecutar la prueba de Wilcoxon
        stat, p_value = stats.wilcoxon(sample1, sample2)

        # Determinar significancia del p-valor
        if p_value < 0.05:
            significance = "significant"
            reject_null = "Reject the null hypothesis"
        elif p_value < 0.1:
            significance = "marginally significant"
            reject_null = "Potential rejection of the null hypothesis"
        else:
            significance = "not significant"
            reject_null = "Do not reject the null hypothesis"

        return jsonify({
            'test': 'Wilcoxon Signed-Rank',
            'statistic': stat,
            'pValue': p_value,
            'significance': significance,
            'decision': reject_null
        })
    
    except Exception as e:
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

##########################################################################################
####################################### MODELOS ##########################################
##########################################################################################


# Función para reemplazar NaN en los resultados
def replace_nan_with_none(values):
    return [None if np.isnan(x) else x for x in values]

# Regresión Lineal
@app.route('/api/linear_regression', methods=['POST'])
def linear_regression():
    try:
        data = request.get_json()
        
        # Predictores y respuesta desde el JSON
        predictors = np.array(data['predictors'])
        response = np.array(data['response'])

        # Asegurarse de que los predictores sean una matriz 2D
        if predictors.ndim == 1:
            predictors = predictors.reshape(-1, 1)

        # Validar que el número de observaciones sea el mismo para predictores y respuesta
        if predictors.shape[0] != response.shape[0]:
            raise ValueError("El número de predictores y respuestas debe coincidir.")

        # Añadir una constante (intercepto) a los predictores
        X = sm.add_constant(predictors)

        # Ajustar el modelo de regresión lineal
        model = sm.OLS(response, X)
        result = model.fit()

        # Separar el intercepto del resto de los coeficientes
        intercept = result.params[0]  # El intercepto es el primer valor
        coefficients = result.params[1:].tolist()  # Coeficientes sin el intercepto

        # Separar p-valores
        intercept_pvalue = replace_nan_with_none([result.pvalues[0]])[0]  # p-valor del intercepto
        p_values = replace_nan_with_none(result.pvalues[1:].tolist())  # p-valores sin el intercepto

        # Estadísticos adicionales
        t_values = replace_nan_with_none(result.tvalues[1:].tolist())
        confidence_intervals = result.conf_int().tolist()[1:]  # Intervalos de confianza sin el intercepto
        r_squared = result.rsquared

        # Devolver los resultados en formato JSON
        return jsonify({
            'model': 'Regresión Lineal',
            'intercept': intercept,
            'intercept_pvalue': intercept_pvalue,
            'coefficients': coefficients,
            'p_values': p_values,
            't_values': t_values,
            'confidence_intervals': confidence_intervals,
            'r_squared': r_squared
        })
    except Exception as e:
        print(f"Error en regresión lineal: {str(e)}")
        return jsonify({'error': str(e)}), 500


# Regresión Logística
@app.route('/api/logistic_regression', methods=['POST'])
def logistic_regression():
    try:
        data = request.get_json()

        # Predictores y respuesta desde el JSON
        predictors = np.array(data['predictors'])
        response = np.array(data['response'])

        # Asegurarse de que los predictores sean una matriz 2D
        if predictors.ndim == 1:
            predictors = predictors.reshape(-1, 1)

        # Validar que el número de observaciones sea el mismo para predictores y respuesta
        if predictors.shape[0] != response.shape[0]:
            raise ValueError("El número de predictores y respuestas debe coincidir.")

        # Verificar que la respuesta sea binaria (0 o 1)
        if not np.array_equal(np.unique(response), [0, 1]):
            raise ValueError("La variable de respuesta debe ser binaria (0 o 1).")

        # Añadir una constante (intercepto) a los predictores
        X = sm.add_constant(predictors)

        # Ajustar el modelo de regresión logística
        model = sm.Logit(response, X)
        result = model.fit(disp=False)

        # Separar el intercepto del resto de los coeficientes
        intercept = result.params[0]  # El intercepto es el primer valor
        coefficients = result.params[1:].tolist()  # Coeficientes sin el intercepto

        # Separar p-valores
        intercept_pvalue = replace_nan_with_none([result.pvalues[0]])[0]  # p-valor del intercepto
        p_values = replace_nan_with_none(result.pvalues[1:].tolist())  # p-valores sin el intercepto

        # Estadísticos adicionales
        z_values = replace_nan_with_none(result.tvalues[1:].tolist())  # En logística, son z-values
        confidence_intervals = result.conf_int().tolist()[1:]  # Intervalos de confianza sin el intercepto

        # Precisión del modelo
        predictions = (result.predict(X) > 0.5).astype(int)
        accuracy = accuracy_score(response, predictions)

        # Matriz de confusión
        cm = confusion_matrix(response, predictions).tolist()

        # Devolver los resultados en formato JSON
        return jsonify({
            'model': 'Regresión Logística',
            'intercept': intercept,
            'intercept_pvalue': intercept_pvalue,
            'coefficients': coefficients,
            'p_values': p_values,
            'z_values': z_values,
            'confidence_intervals': confidence_intervals,
            'accuracy': accuracy,
            'confusion_matrix': cm
        })
    except Exception as e:
        print(f"Error en regresión logística: {str(e)}")
        return jsonify({'error': str(e)}), 500



# Ruta para llamar a Render y que no se apague
@app.route('/ping', methods=['HEAD', 'GET'])
def ping():
    return '', 200

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)











