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
from lifelines import CoxPHFitter

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import csv

import statsmodels.formula.api as smf

from flask_cors import CORS

import re  # Para limpiar los nombres de columnas

# Cambiar el backend de matplotlib para evitar problemas de hilos en entornos de servidor
plt.switch_backend('Agg')

app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": "*"}})

###########################################################################################
####################################### SAMPLE SIZE #######################################
###########################################################################################

# Ruta para el cálculo de tamaño muestral para comparación de medias (ttest)
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
        effect_size = abs(2 * (math.asin(math.sqrt(p1)) - math.asin(math.sqrt(p2))))

        # Configurar el análisis de poder para prueba de proporciones independientes
        analysis = NormalIndPower()

        # Calcular el tamaño muestral necesario
        sample_size = analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power, alternative=alternative)

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
        effect_size = abs(2 * (math.asin(math.sqrt(p1)) - math.asin(math.sqrt(p2))))

        # Utilizar NormalIndPower como aproximación
        analysis = NormalIndPower()
        sample_size = analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power, alternative=alternative)

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

    
# Ruta para el cálculo de tamaño muestral para estudios de Supervivencia (Log-rank test)
@app.route('/calculate_sample_size_logrank', methods=['POST'])
def calculate_sample_size_logrank():
    try:
        data = request.get_json()

        # Obtener los parámetros de la solicitud
        alpha = data.get('alpha', 0.05)  # Nivel de significancia
        power = data.get('power', 0.8)  # Potencia estadística
        p1 = data.get('p1', 0.5)  # Proporción esperada de eventos en el grupo control
        p2 = data.get('p2', 0.3)  # Proporción esperada de eventos en el grupo tratamiento
        alternative = data.get('alternative', 'two-sided')  # Tipo de hipótesis (unilateral o bilateral)

        # Ajustar el valor crítico según el tipo de hipótesis
        if alternative == 'two-sided':
            z_alpha = norm.ppf(1 - alpha / 2)
        else:
            z_alpha = norm.ppf(1 - alpha)

        z_beta = norm.ppf(power)

        # Proporción combinada de eventos
        p_combined = (p1 + p2) / 2

        # Tamaño del efecto para log-rank test
        effect_size = abs(p1 - p2) / math.sqrt(p_combined * (1 - p_combined))

        # Calcular el tamaño muestral necesario
        sample_size = ((z_alpha + z_beta) ** 2) / (effect_size ** 2)

        return jsonify({'sample_size': math.ceil(sample_size), 'effect_size': effect_size})

    except Exception as e:
        return jsonify({'error': str(e)}), 400


# Ruta para el cálculo de tamaño muestral para estudios Longitudinales o de Medidas Repetidas
@app.route('/calculate_sample_size_longitudinal', methods=['POST'])
def calculate_sample_size_longitudinal():
    try:
        data = request.get_json()

        # Obtener los parámetros de la solicitud
        alpha = data.get('alpha', 0.05)  # Nivel de significancia
        power = data.get('power', 0.8)  # Potencia estadística
        effect_size = data.get('effect_size', 0.5)  # Tamaño del efecto
        n_measurements = data.get('n_measurements', 3)  # Número de mediciones
        rho = data.get('rho', 0.5)  # Correlación intra-sujeto

        # Ajustar el tamaño del efecto considerando la correlación intra-sujeto
        adjusted_effect_size = effect_size / math.sqrt(1 + (n_measurements - 1) * rho)

        # Configurar el análisis de poder
        analysis = TTestPower()
        sample_size = analysis.solve_power(effect_size=adjusted_effect_size, alpha=alpha, power=power)

        return jsonify({'sample_size': round(sample_size), 'adjusted_effect_size': adjusted_effect_size})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Ruta para el cálculo de tamaño muestral para pruebas de No Inferioridad
@app.route('/calculate_sample_size_non_inferiority', methods=['POST'])
def calculate_sample_size_non_inferiority():
    try:
        data = request.get_json()

        # Obtener los parámetros de la solicitud
        alpha = data.get('alpha', 0.05)  # Nivel de significancia
        power = data.get('power', 0.8)  # Potencia estadística
        margin = data.get('margin', 0.1)  # Margen de no inferioridad
        effect_size = data.get('effect_size', None)  # Tamaño del efecto (Cohen's d)
        alternative = data.get('alternative', 'larger')  # Tipo de hipótesis (unilateral)

        if effect_size is None:
            raise ValueError("El tamaño del efecto es necesario para el cálculo.")

        # Configurar el análisis de poder para no inferioridad
        analysis = TTestIndPower()
        sample_size = analysis.solve_power(
            effect_size=effect_size - margin,
            alpha=alpha,
            power=power,
            alternative=alternative
        )

        # Convertir a escalar si es un array
        if isinstance(sample_size, np.ndarray):
            sample_size = sample_size.item()

        return jsonify({'sample_size': round(sample_size), 'effect_size': effect_size})

    except Exception as e:
        return jsonify({'error': str(e)}), 400



# Ruta para el cálculo de tamaño muestral para pruebas de Equivalencia (TOST)
@app.route('/calculate_sample_size_tost', methods=['POST'])
def calculate_sample_size_tost():
    try:
        data = request.get_json()

        # Obtener los parámetros de la solicitud
        alpha = data.get('alpha', 0.05)  # Nivel de significancia
        power = data.get('power', 0.8)  # Potencia estadística
        lower_margin = data.get('lower_margin', -0.1)  # Margen inferior de equivalencia
        upper_margin = data.get('upper_margin', 0.1)  # Margen superior de equivalencia
        effect_size = data.get('effect_size', None)  # Tamaño del efecto (Cohen's d)

        if effect_size is None:
            raise ValueError("El tamaño del efecto es necesario para el cálculo.")

        # Configurar el análisis de poder para prueba TOST
        analysis = TTestIndPower()
        sample_size_lower = analysis.solve_power(effect_size=effect_size - lower_margin, alpha=alpha / 2, power=power, alternative='larger')
        sample_size_upper = analysis.solve_power(effect_size=effect_size - upper_margin, alpha=alpha / 2, power=power, alternative='smaller')

        # Convertir los resultados a escalares si son arrays
        if isinstance(sample_size_lower, np.ndarray):
            sample_size_lower = sample_size_lower.item()
        if isinstance(sample_size_upper, np.ndarray):
            sample_size_upper = sample_size_upper.item()

        # Tomar el tamaño muestral máximo de ambas pruebas
        sample_size = max(sample_size_lower, sample_size_upper)

        return jsonify({'sample_size': round(sample_size), 'effect_size': effect_size})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

####################################################################################################
####################################### DESCRIPTIVE ANALYSIS #######################################
####################################################################################################

# Función para calcular estadísticas descriptivas para una o dos muestras
def calculate_descriptive_statistics(request_body):
    try:
        # Extraer datos de la solicitud
        data_series1 = pd.Series(request_body.get('data1', []))
        data_series2 = request_body.get('data2')
        category_series = request_body.get('categories')

        # Verificar si el cliente solicitó gráficos específicos
        show_boxplot = request_body.get('showBoxplot', False)
        show_violinplot = request_body.get('showViolinPlot', False)
        show_raincloudplot = request_body.get('showRaincloudPlot', False)
        show_histogram = request_body.get('showHistogram', False)
        show_density = request_body.get('showDensity', False)
        show_scatter = request_body.get('showScatter', False)

        results = {}  # Inicializar el diccionario de resultados

        if category_series is not None:
            category_series = pd.Series(category_series)

            # Calcular estadísticas descriptivas por categoría
            grouped = data_series1.groupby(category_series)
            stats_by_category = {}
            for category, group in grouped:
                stats_by_category[category] = {
                    'count': float(group.size),
                    'mean': float(group.mean()) if not group.mean().isna() else None,
                    'median': float(group.median()) if not group.median().isna() else None,
                    'std': float(group.std()) if not group.std().isna() else None,
                    'min': float(group.min()) if not group.min().isna() else None,
                    'max': float(group.max()) if not group.max().isna() else None,
                }
            
                # Asegurarse de que categorías con un solo valor sean manejadas correctamente
                if group.size == 1:
                    stats_by_category[category]['std'] = None  # Desviación estándar no aplicable
            
            results['stats_by_category'] = stats_by_category


            # Generar gráficos específicos para el análisis con categorías
            if show_boxplot:
                plt.figure(figsize=(8, 6))
                sns.boxplot(x=category_series, y=data_series1, palette="Set2", width=0.4)
                plt.title('Boxplot por Categorías')
                plt.xlabel('Categoría')
                plt.ylabel('Valor')
                plt.tight_layout()
                boxplot_img = io.BytesIO()
                plt.savefig(boxplot_img, format='png', bbox_inches='tight', dpi=100)
                boxplot_img.seek(0)
                results['boxplot_by_category'] = base64.b64encode(boxplot_img.getvalue()).decode()
                plt.close()

            if show_violinplot:
                plt.figure(figsize=(8, 6))
                sns.violinplot(x=category_series, y=data_series1, palette="Set2", width=0.8)
                plt.title('Gráfico de Violín por Categorías')
                plt.xlabel('Categoría')
                plt.ylabel('Valor')
                plt.tight_layout()
                violin_img = io.BytesIO()
                plt.savefig(violin_img, format='png', bbox_inches='tight', dpi=100)
                violin_img.seek(0)
                results['violin_plot_by_category'] = base64.b64encode(violin_img.getvalue()).decode()
                plt.close()

            if show_raincloudplot:
                fig, ax = plt.subplots(figsize=(10, 8))
                palette = sns.color_palette("Set2", len(category_series.unique()))
                for i, category in enumerate(category_series.unique()):
                    cat_data = data_series1[category_series == category]
                    bp = ax.boxplot([cat_data], positions=[i + 1], patch_artist=True, vert=False, widths=0.2)
                    bp['boxes'][0].set_facecolor(palette[i])
                    bp['boxes'][0].set_alpha(0.4)
                    vp = ax.violinplot([cat_data], positions=[i + 1], showmeans=False, showextrema=False, showmedians=False, vert=False)
                    for b in vp['bodies']:
                        b.get_paths()[0].vertices[:, 1] = np.clip(b.get_paths()[0].vertices[:, 1], i + 1, i + 1.5)
                        b.set_color(palette[i])
                    y_jitter = np.full(len(cat_data), i + 1) + np.random.uniform(-0.05, 0.05, len(cat_data))
                    ax.scatter(cat_data, y_jitter, s=3, color=palette[i], alpha=0.5)
                ax.set_yticks(np.arange(1, len(category_series.unique()) + 1))
                ax.set_yticklabels(category_series.unique())
                ax.set_xlabel('Valor')
                ax.set_title('Raincloud Plot')
                plt.tight_layout()
                raincloud_img = io.BytesIO()
                plt.savefig(raincloud_img, format='png', bbox_inches='tight', dpi=100)
                raincloud_img.seek(0)
                results['raincloud_plot'] = base64.b64encode(raincloud_img.getvalue()).decode()
                plt.close()
            return results

        # Análisis para una sola muestra sin categorías
        def analyze_single_series(data_series, title_suffix):
            stats = {
                'count': float(data_series.size),
                'mean': float(data_series.mean()),
                'median': float(data_series.median()),
                'mode': data_series.mode().tolist(),
                'std': float(data_series.std()),
                'variance': float(data_series.var()),
                'min': float(data_series.min()),
                'max': float(data_series.max()),
                'range': float(data_series.max() - data_series.min()),
                'coef_var': float((data_series.std() / data_series.mean()) * 100) if data_series.mean() != 0 else None,
                'skewness': float(skew(data_series, nan_policy='omit')),
                'kurtosis': float(kurtosis(data_series, nan_policy='omit')),
                'q1': float(data_series.quantile(0.25)),
                'q3': float(data_series.quantile(0.75)),
                'iqr': float(data_series.quantile(0.75) - data_series.quantile(0.25)),
                'p10': float(data_series.quantile(0.10)),
                'p90': float(data_series.quantile(0.90)),
                'outliers': data_series[(data_series < (data_series.quantile(0.25) - 1.5 * (data_series.quantile(0.75) - data_series.quantile(0.25)))) |
                                         (data_series > (data_series.quantile(0.75) + 1.5 * (data_series.quantile(0.75) - data_series.quantile(0.25))))].tolist()
            }

            # Generar gráficos si están solicitados
            if show_histogram:
                plt.figure(figsize=(6, 4))
                plt.hist(data_series.dropna(), bins=10, color='skyblue', edgecolor='black')
                plt.title(f'Histograma de Datos {title_suffix}')
                plt.xlabel('Valor')
                plt.ylabel('Frecuencia')
                plt.tight_layout()
                img = io.BytesIO()
                plt.savefig(img, format='png')
                img.seek(0)
                stats['histogram'] = base64.b64encode(img.getvalue()).decode()
                plt.close()

            if show_boxplot:
                plt.figure(figsize=(6, 4))
                plt.boxplot(data_series.dropna(), vert=False, patch_artist=True, boxprops=dict(facecolor='lightblue'))
                plt.title(f'Boxplot de Datos {title_suffix}')
                plt.xlabel('Valor')
                plt.tight_layout()
                boxplot_img = io.BytesIO()
                plt.savefig(boxplot_img, format='png')
                boxplot_img.seek(0)
                stats['boxplot'] = base64.b64encode(boxplot_img.getvalue()).decode()
                plt.close()

            if show_density:
                plt.figure(figsize=(6, 4))
                sns.kdeplot(data_series.dropna(), shade=True, color='green')
                plt.title(f'Gráfico de Densidad {title_suffix}')
                plt.xlabel('Valor')
                plt.tight_layout()
                density_img = io.BytesIO()
                plt.savefig(density_img, format='png')
                density_img.seek(0)
                stats['density_plot'] = base64.b64encode(density_img.getvalue()).decode()
                plt.close()
            if show_violinplot:
                plt.figure(figsize=(6, 4))
                sns.violinplot(data=data_series.dropna(), color='lightcoral')
                plt.title(f'Gráfico de Violín {title_suffix}')
                plt.xlabel('Valor')
                plt.tight_layout()
                violin_img = io.BytesIO()
                plt.savefig(violin_img, format='png')
                violin_img.seek(0)
                stats['violin_plot'] = base64.b64encode(violin_img.getvalue()).decode()
                plt.close()

            if show_raincloudplot:
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.violinplot(data=data_series.dropna(), ax=ax, inner=None, color='lightblue', alpha=0.5)
                sns.stripplot(data=data_series.dropna(), ax=ax, color='darkblue', size=4, jitter=True, alpha=0.6)
                ax.set_title(f'Raincloud Plot {title_suffix}')
                ax.set_xlabel('Valor')
                plt.tight_layout()
                raincloud_img = io.BytesIO()
                plt.savefig(raincloud_img, format='png')
                raincloud_img.seek(0)
                stats['raincloud_plot'] = base64.b64encode(raincloud_img.getvalue()).decode()
                plt.close()
            return stats

        # Si solo hay una muestra, devolver las estadísticas para una muestra
        if data_series2 is None:
            results.update(analyze_single_series(data_series1, ""))
            return results

        # Si hay dos muestras, realizar análisis conjunto
        mean1 = float(data_series1.mean())
        mean2 = float(data_series2.mean())
        correlation, _ = stats.pearsonr(data_series1, data_series2)

        results.update({
            'mean1': mean1,
            'mean2': mean2,
            'correlation': correlation
        })

        if show_scatter:
            plt.figure(figsize=(6, 4))
            plt.scatter(data_series1, data_series2, color='purple', alpha=0.6)
            plt.title('Gráfico de Dispersión con Línea de Tendencia')
            plt.xlabel('Muestra 1')
            plt.ylabel('Muestra 2')
            plt.tight_layout()
            m, b = np.polyfit(data_series1, data_series2, 1)
            plt.plot(data_series1, m * data_series1 + b, color='red')
            scatter_img = io.BytesIO()
            plt.savefig(scatter_img, format='png')
            scatter_img.seek(0)
            results['scatter_plot'] = base64.b64encode(scatter_img.getvalue()).decode()
            plt.close()
        return results

    except Exception as e:
        return {'error': str(e)}

# Ruta para el análisis descriptivo básico a partir de una lista de datos proporcionada
@app.route('/calculate_basic_analysis', methods=['POST'])
def calculate_basic_analysis():
    try:
        data = request.get_json()

        # Obtener la primera muestra
        data_list1 = data.get('data1', None)
        if data_list1 is None or not isinstance(data_list1, list):
            raise ValueError("Los datos de la primera muestra no son válidos o no se encuentran en el formato adecuado.")

        data_series1 = pd.Series(data_list1)

        data_list2 = data.get('data2')
        category_list = data.get('categories')

        show_boxplot = data.get('showBoxplot', False)
        show_violinplot = data.get('showViolinPlot', False)
        show_raincloudplot = data.get('showRaincloudPlot', False)
        show_histogram = data.get('showHistogram', False)
        show_density = data.get('showDensity', False)
        show_scatter = data.get('showScatter', False)

        # Caso 1: Análisis con categorías
        if category_list is not None:
            if not isinstance(category_list, list):
                raise ValueError("Las categorías no son válidas o no se encuentran en el formato adecuado.")
            category_series = pd.Series(category_list)
            result = calculate_descriptive_statistics({
                'data1': data_series1, 
                'categories': category_series,
                'showBoxplot': show_boxplot,
                'showViolinPlot': show_violinplot,
                'showRaincloudPlot': show_raincloudplot
            })

        # Caso 2: Análisis para dos muestras
        elif data_list2 is not None:
            if not isinstance(data_list2, list):
                raise ValueError("Los datos de la segunda muestra no son válidos o no se encuentran en el formato adecuado.")
            data_series2 = pd.Series(data_list2)
            result = calculate_descriptive_statistics({
                'data1': data_series1, 
                'data2': data_series2,
                'showScatter': show_scatter
            })

        # Caso 3: Análisis para una sola muestra
        else:
            result = calculate_descriptive_statistics({
                'data1': data_series1,
                'showBoxplot': show_boxplot,
                'showHistogram': show_histogram,
                'showDensity': show_density,
                'showViolinPlot': show_violinplot,
                'showRaincloudPlot': show_raincloudplot
            })

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 400


    
############################## DESDE BASE DE DATOS ###########################################

# Función para reemplazar NaN en los resultados
def replace_nan_with_none(values):
    return [None if pd.isna(x) else x for x in values]

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

        filename = file.filename.lower()

        if filename.endswith('.csv'):
            try:
                # Leer contenido del archivo para detectar delimitador
                content = file.stream.read().decode("utf-8")
                # Detectar delimitador dinámicamente
                dialect = csv.Sniffer().sniff(content[:1024], delimiters=";,")
                delimiter = dialect.delimiter
                dataframe = pd.read_csv(io.StringIO(content), delimiter=delimiter)
            except UnicodeDecodeError:
                # Si UTF-8 falla, intentar con ISO-8859-1
                content = file.stream.read().decode("ISO-8859-1")
                dialect = csv.Sniffer().sniff(content[:1024], delimiters=";,")
                delimiter = dialect.delimiter
                dataframe = pd.read_csv(io.StringIO(content), delimiter=delimiter)
            except pd.errors.ParserError as e:
                return jsonify({'error': f'Error al analizar el archivo CSV: {str(e)}'}), 400

        elif filename.endswith(('.xls', '.xlsx')):
            try:
                # Leer archivos Excel
                file_stream = io.BytesIO(file.read())
                dataframe = pd.read_excel(file_stream, engine='openpyxl')
            except ValueError as e:
                return jsonify({'error': f'Error al leer el archivo Excel: {str(e)}'}), 400
            except Exception as e:
                return jsonify({'error': f'Error desconocido al leer el archivo Excel: {str(e)}'}), 400

        elif filename.endswith('.txt'):
            try:
                content = file.stream.read().decode("utf-8")
                dataframe = pd.read_csv(io.StringIO(content), delimiter=r'\s+')
            except UnicodeDecodeError:
                content = file.stream.read().decode("ISO-8859-1")
                dataframe = pd.read_csv(io.StringIO(content), delimiter=r'\s+')
        else:
            return jsonify({'error': 'Formato de archivo no soportado. Proporcione un archivo CSV, XLSX, XLS o TXT.'}), 400

        # Verificar si el DataFrame se cargó correctamente
        if dataframe.empty:
            return jsonify({'error': 'El archivo está vacío o no se pudo procesar correctamente.'}), 400

        # Normalizar encabezados quitando espacios adicionales
        dataframe.columns = dataframe.columns.str.strip()

        # Manejo de celdas vacías
        dataframe = dataframe.fillna("N/A")  # Rellenar celdas vacías con "N/A"

        # Intentar convertir columnas numéricas que puedan haber sido interpretadas como texto
        for column in dataframe.columns:
            try:
                dataframe[column] = pd.to_numeric(dataframe[column].str.replace(',', '.'), errors='ignore')
            except AttributeError:
                # Si no es un string, continuar sin cambios
                continue

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

# Ruta para el análisis descriptivo de columnas seleccionadas
@app.route('/analyze_selected_columns', methods=['POST'])
def analyze_selected_columns():
    global dataframe
    try:
        if dataframe is None:
            return jsonify({'error': 'No se ha cargado ningún archivo para analizar.'}), 400

        data = request.get_json()
        analysis_type = data.get('analysis_type', 'Una muestra')
        selected_columns = data.get('numeric_columns', [])
        category_column = data.get('category')

        # Parámetros para los gráficos solicitados
        show_boxplot = data.get('showBoxplot', False)
        show_histogram = data.get('showHistogram', False)
        show_scatter = data.get('showScatter', False)
        show_violinplot = data.get('showViolinPlot', False)
        show_raincloudplot = data.get('showRaincloudPlot', False)

        if not selected_columns:
            return jsonify({'error': "No se proporcionaron columnas numéricas para analizar."}), 400

        result = {}

        # Análisis de una muestra (histograma y boxplot)
        if analysis_type == "Una muestra":
            if len(selected_columns) != 1:
                return jsonify({'error': "Seleccione exactamente una columna numérica para este análisis."}), 400
            
            data_series = dataframe[selected_columns[0]].dropna()

            # Calcular estadísticas descriptivas
            result[selected_columns[0]] = calculate_descriptive_statistics_from_data(data_series)

            # Crear histograma si solicitado
            if show_histogram:
                plt.figure(figsize=(8, 6))
                sns.histplot(data_series, bins=20, kde=True, color='skyblue')
                plt.xlabel(selected_columns[0])
                plt.ylabel('Frecuencia')
                plt.title(f'Histograma de {selected_columns[0]}')
                histogram_img = io.BytesIO()
                plt.savefig(histogram_img, format='png', bbox_inches='tight', dpi=100)
                histogram_img.seek(0)
                result['histogram'] = base64.b64encode(histogram_img.getvalue()).decode()
                plt.close()

            # Crear boxplot si solicitado
            if show_boxplot:
                plt.figure(figsize=(8, 6))
                sns.boxplot(x=data_series, color='orange')
                plt.title(f'Boxplot de {selected_columns[0]}')
                boxplot_img = io.BytesIO()
                plt.savefig(boxplot_img, format='png', bbox_inches='tight', dpi=100)
                boxplot_img.seek(0)
                result['boxplot'] = base64.b64encode(boxplot_img.getvalue()).decode()
                plt.close()

        # Análisis de dos muestras (scatter plot y correlación)
        elif analysis_type == "Dos muestras":
            if len(selected_columns) != 2:
                return jsonify({'error': "Seleccione exactamente dos columnas numéricas para este análisis."}), 400

            data_series1 = dataframe[selected_columns[0]].dropna()
            data_series2 = dataframe[selected_columns[1]].dropna()

            if len(data_series1) != len(data_series2):
                return jsonify({'error': "Las columnas seleccionadas tienen diferentes cantidades de datos válidos."}), 400

            result['mean1'] = float(data_series1.mean())
            result['mean2'] = float(data_series2.mean())
            result['correlation'], _ = stats.pearsonr(data_series1, data_series2)

            # Crear scatter plot si solicitado
            if show_scatter:
                plt.figure(figsize=(8, 6))
                sns.scatterplot(x=data_series1, y=data_series2, color='blue', alpha=0.6, s=80, edgecolor='w')
                plt.title('Gráfico de Dispersión con Línea de Tendencia')
                plt.xlabel(selected_columns[0])
                plt.ylabel(selected_columns[1])
                m, b = np.polyfit(data_series1, data_series2, 1)
                plt.plot(data_series1, m * data_series1 + b, color='red')
                scatter_img = io.BytesIO()
                plt.savefig(scatter_img, format='png', bbox_inches='tight', dpi=100)
                scatter_img.seek(0)
                result['scatter_plot'] = base64.b64encode(scatter_img.getvalue()).decode()
                plt.close()

        # Análisis en función de una categórica
        elif analysis_type == "En función de una categórica":
            if len(selected_columns) != 1 or not category_column:
                return jsonify({'error': "Seleccione una columna numérica y una categórica para este análisis."}), 400
            
            data_series = dataframe[selected_columns[0]].dropna()
            category_series = dataframe[category_column].dropna()

            # Agrupar y calcular estadísticas descriptivas por categoría
            grouped = data_series.groupby(category_series)
            stats_by_category = {
                str(category): {
                    'count': float(group.size),
                    'mean': float(group.mean()) if not np.isnan(group.mean()) else None,
                    'median': float(group.median()) if not np.isnan(group.median()) else None,
                    'std': float(group.std()) if not np.isnan(group.std()) else None,
                    'min': float(group.min()),
                    'max': float(group.max())
                }
                for category, group in grouped
            }

            # Gráficos específicos por categoría si solicitados
            if show_boxplot:
                plt.figure(figsize=(8, 6))
                sns.boxplot(x=category_series, y=data_series, palette="Set2", width=0.4)
                plt.title(f'Boxplot de {selected_columns[0]} según {category_column}')
                plt.xlabel(category_column)
                plt.ylabel(selected_columns[0])
                boxplot_img = io.BytesIO()
                plt.savefig(boxplot_img, format='png', bbox_inches='tight', dpi=100)
                boxplot_img.seek(0)
                result['boxplot_by_category'] = base64.b64encode(boxplot_img.getvalue()).decode()
                plt.close()

            if show_violinplot:
                plt.figure(figsize=(8, 6))
                sns.violinplot(x=category_series, y=data_series, palette="Set2", width=0.8)
                plt.title(f'Violin Plot de {selected_columns[0]} según {category_column}')
                plt.xlabel(category_column)
                plt.ylabel(selected_columns[0])
                violin_img = io.BytesIO()
                plt.savefig(violin_img, format='png', bbox_inches='tight', dpi=100)
                violin_img.seek(0)
                result['violin_by_category'] = base64.b64encode(violin_img.getvalue()).decode()
                plt.close()

            if show_raincloudplot:
                fig, ax = plt.subplots(figsize=(10, 8))
                palette = sns.color_palette("Set2", len(category_series.unique()))
                for i, category in enumerate(category_series.unique()):
                    cat_data = data_series[category_series == category]
                    bp = ax.boxplot([cat_data], positions=[i + 1], patch_artist=True, vert=False, widths=0.2)
                    bp['boxes'][0].set_facecolor(palette[i])
                    bp['boxes'][0].set_alpha(0.4)
                    vp = ax.violinplot([cat_data], positions=[i + 1], points=500, showmeans=False,
                                        showextrema=False, showmedians=False, vert=False)
                    for b in vp['bodies']:
                        b.get_paths()[0].vertices[:, 1] = np.clip(b.get_paths()[0].vertices[:, 1], i + 1, i + 1.5)
                        b.set_color(palette[i])
                    y = np.full(len(cat_data), i + 1)
                    y_jitter = y + np.random.uniform(-0.05, 0.05, size=len(cat_data))
                    ax.scatter(cat_data, y_jitter, s=3, color=palette[i], alpha=0.5)
                ax.set_yticks(np.arange(1, len(category_series.unique()) + 1))
                ax.set_yticklabels(category_series.unique())
                ax.set_xlabel('Valores')
                ax.set_title(f'Raincloud Plot de {selected_columns[0]} según {category_column}')
                raincloud_img = io.BytesIO()
                plt.savefig(raincloud_img, format='png', bbox_inches='tight', dpi=100)
                raincloud_img.seek(0)
                result['raincloud_plot'] = base64.b64encode(raincloud_img.getvalue()).decode()
                plt.close()

            result['stats_by_category'] = stats_by_category

        else:
            return jsonify({'error': "Tipo de análisis no válido."}), 400

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Función para calcular estadísticas descriptivas
def calculate_descriptive_statistics_from_data(data_series):
    try:
        # Ignorar NaN en lugar de reemplazarlos con 0
        count = float(data_series.dropna().size)
        mean = float(data_series.mean(skipna=True))
        median = float(data_series.median(skipna=True))
        mode = replace_nan_with_none(data_series.mode(dropna=True).tolist())  # Usar dropna=True para la moda
        std = float(data_series.std(skipna=True))
        variance = float(data_series.var(skipna=True))
        min_value = float(data_series.min(skipna=True))
        max_value = float(data_series.max(skipna=True))
        range_value = max_value - min_value
        coef_var = (std / mean * 100) if mean != 0 else None

        # Sanitizar skewness y kurtosis
        skewness = None if np.isnan(stats.skew(data_series.dropna())) else float(stats.skew(data_series.dropna()))
        kurtosis_value = None if np.isnan(stats.kurtosis(data_series.dropna())) else float(stats.kurtosis(data_series.dropna()))

        q1 = float(data_series.quantile(0.25, interpolation='linear'))
        q3 = float(data_series.quantile(0.75, interpolation='linear'))
        p10 = float(data_series.quantile(0.10, interpolation='linear'))
        p90 = float(data_series.quantile(0.90, interpolation='linear'))

        return {
            'count': count,
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
            'kurtosis': kurtosis_value,
            'q1': q1,
            'q3': q3,
            'p10': p10,
            'p90': p90
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
                # Leer contenido del archivo para detectar delimitador
                content = file.stream.read().decode("utf-8")
                # Intentar detectar delimitador automáticamente
                dialect = csv.Sniffer().sniff(content[:1024], delimiters=";,")
                delimiter = dialect.delimiter
                dataframe = pd.read_csv(io.StringIO(content), delimiter=delimiter)
            except UnicodeDecodeError:
                # Si falla la codificación UTF-8, intentar ISO-8859-1
                content = file.stream.read().decode("ISO-8859-1")
                dialect = csv.Sniffer().sniff(content[:1024], delimiters=";,")
                delimiter = dialect.delimiter
                dataframe = pd.read_csv(io.StringIO(content), delimiter=delimiter)
            except pd.errors.ParserError as e:
                return jsonify({'error': f'Error al analizar el archivo CSV: {str(e)}'}), 400

        elif filename.endswith(('.xls', '.xlsx')):
            try:
                # Leer archivos Excel
                file_stream = io.BytesIO(file.read())
                dataframe = pd.read_excel(file_stream, engine='openpyxl')
            except ValueError as e:
                return jsonify({'error': f'Error al leer el archivo Excel: {str(e)}'}), 400
            except Exception as e:
                return jsonify({'error': f'Error desconocido al leer el archivo Excel: {str(e)}'}), 400

        elif filename.endswith('.txt'):
            try:
                content = file.stream.read().decode("utf-8")
                dataframe = pd.read_csv(io.StringIO(content), delimiter=r'\s+')
            except UnicodeDecodeError:
                content = file.stream.read().decode("ISO-8859-1")
                dataframe = pd.read_csv(io.StringIO(content), delimiter=r'\s+')
        else:
            return jsonify({'error': "Formato de archivo no soportado. Proporcione un archivo CSV, XLSX, XLS o TXT."}), 400

        # Verificar si el DataFrame se cargó correctamente
        if dataframe.empty:
            return jsonify({'error': 'El archivo está vacío o no se pudo procesar correctamente.'}), 400

        # Normalizar encabezados quitando espacios adicionales
        dataframe.columns = dataframe.columns.str.strip()

        # Manejo de celdas vacías
        dataframe = dataframe.fillna("N/A")  # Rellenar celdas vacías con "N/A"

        # Intentar convertir columnas numéricas que puedan haber sido interpretadas como texto
        for column in dataframe.columns:
            try:
                dataframe[column] = pd.to_numeric(dataframe[column].str.replace(',', '.'), errors='ignore')
            except AttributeError:
                # Si no es un string, continuar sin cambios
                continue

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
        plt.figure(figsize=(10, 8))

        if chart_type == 'Scatterplot':
            if not y_column:
                return jsonify({'error': 'Para un scatterplot, debe seleccionar dos variables numéricas.'}), 400
            
            add_trendline = data.get('add_trendline', False)

            if pd.api.types.is_numeric_dtype(dataframe[x_column]) and pd.api.types.is_numeric_dtype(dataframe[y_column]):
                sns.scatterplot(x=df_clean[x_column], y=df_clean[y_column], color='blue', alpha=0.6, edgecolor='w', s=80)
                plt.xlabel(x_column)
                plt.ylabel(y_column)
                plt.title(f'Scatterplot de {x_column} vs {y_column}')
                plt.grid(True)

                # Añadir línea de tendencia si se solicita
                if add_trendline:
                    m, b = np.polyfit(df_clean[x_column], df_clean[y_column], 1)
                    plt.plot(df_clean[x_column], m * df_clean[x_column] + b, color='red', linestyle='--', linewidth=2)
                    plt.legend(['Línea de tendencia', 'Datos'])
                    
            else:
                return jsonify({'error': 'Ambas columnas seleccionadas deben ser numéricas para un scatterplot.'}), 400

        elif chart_type == 'Histograma':
            if pd.api.types.is_numeric_dtype(dataframe[x_column]):
                sns.histplot(df_clean[x_column], bins=20, kde=True, color='skyblue')
                plt.xlabel(x_column)
                plt.ylabel('Frecuencia')
                plt.title(f'Histograma de {x_column}')
            else:
                return jsonify({'error': 'La columna seleccionada debe ser numérica para un histograma.'}), 400

        elif chart_type == 'Boxplot':
            if pd.api.types.is_numeric_dtype(dataframe[x_column]):
                if categorical_column and categorical_column in dataframe.columns:
                    # Asegurar que la columna categórica sea tratada como categoría
                    df_clean[categorical_column] = df_clean[categorical_column].astype('category')

                    # Verificar si la columna categórica es válida
                    if pd.api.types.is_categorical_dtype(df_clean[categorical_column]) or df_clean[categorical_column].dtype == 'object':
                        sns.boxplot(
                            x=categorical_column, 
                            y=x_column, 
                            data=df_clean, 
                            palette='Set3', 
                            width=0.4  # Ajustar el ancho de las barras para hacerlas más finas
                        )
                        plt.title(f'Boxplot de {x_column} según {categorical_column}')
                        plt.xlabel(categorical_column)  # Etiqueta correcta para el eje X
                        plt.ylabel(x_column)  # Etiqueta correcta para el eje Y
                    else:
                        return jsonify({'error': 'La columna categórica seleccionada no es válida.'}), 400
                else:
                    # Boxplot sin categorización
                    sns.boxplot(x=df_clean[x_column], color='lightgreen', width=0.4)
                    plt.title(f'Boxplot de {x_column}')
                    plt.xlabel(x_column)
                    plt.ylabel('Frecuencia')  # Etiqueta adecuada para el eje Y cuando no hay categorización
            else:
                return jsonify({'error': 'La columna seleccionada debe ser numérica para un boxplot.'}), 400

        elif chart_type == 'Raincloud Plot':
            if pd.api.types.is_numeric_dtype(dataframe[x_column]):
                fig, ax = plt.subplots(figsize=(10, 8))

                if categorical_column and categorical_column in dataframe.columns:
                    # Raincloud Plot con categorización
                    palette = sns.color_palette("Set2", len(df_clean[categorical_column].unique()))
                    for i, category in enumerate(df_clean[categorical_column].unique()):
                        cat_data = df_clean[df_clean[categorical_column] == category][x_column]
                        
                        # Crear boxplot
                        bp = ax.boxplot([cat_data], positions=[i + 1], patch_artist=True, vert=False, widths=0.2)
                        bp['boxes'][0].set_facecolor(palette[i])
                        bp['boxes'][0].set_alpha(0.4)

                        # Crear violin plot
                        vp = ax.violinplot([cat_data], positions=[i + 1], points=500, showmeans=False,
                                            showextrema=False, showmedians=False, vert=False)
                        for b in vp['bodies']:
                            b.get_paths()[0].vertices[:, 1] = np.clip(b.get_paths()[0].vertices[:, 1], i + 1, i + 1.5)
                            b.set_color(palette[i])

                        # Añadir puntos con jitter
                        y_jitter = np.full(len(cat_data), i + 1) + np.random.uniform(-0.05, 0.05, size=len(cat_data))
                        ax.scatter(cat_data, y_jitter, s=10, color=palette[i], alpha=0.6)

                    ax.set_yticks(np.arange(1, len(df_clean[categorical_column].unique()) + 1))
                    ax.set_yticklabels(df_clean[categorical_column].unique())
                    ax.set_xlabel(x_column)
                    ax.set_title(f'Raincloud Plot de {x_column} según {categorical_column}')
                    plt.grid(True)
                else:
                    # Raincloud Plot sin categorización
                    sns.violinplot(x=df_clean[x_column], inner=None, color='skyblue', alpha=0.5)
                    sns.boxplot(x=df_clean[x_column], color='lightgreen', width=0.2)
                    y_jitter = np.random.uniform(-0.05, 0.05, size=len(df_clean[x_column]))
                    plt.scatter(df_clean[x_column], y_jitter, s=10, color='blue', alpha=0.6)
                    plt.xlabel(x_column)
                    plt.title(f'Raincloud Plot de {x_column}')

            else:
                return jsonify({'error': 'La columna seleccionada debe ser numérica para un Raincloud Plot.'}), 400

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
                    # Leer contenido del archivo para detectar delimitador
                    content = file.stream.read().decode("utf-8")
                    dialect = csv.Sniffer().sniff(content[:1024], delimiters=";,")
                    delimiter = dialect.delimiter
                    dataframe = pd.read_csv(io.StringIO(content), delimiter=delimiter)
                except UnicodeDecodeError:
                    # Intentar con ISO-8859-1 si UTF-8 falla
                    content = file.stream.read().decode("ISO-8859-1")
                    dialect = csv.Sniffer().sniff(content[:1024], delimiters=";,")
                    delimiter = dialect.delimiter
                    dataframe = pd.read_csv(io.StringIO(content), delimiter=delimiter)
                except pd.errors.ParserError as e:
                    return jsonify({'error': f'Error al analizar el archivo CSV: {str(e)}'}), 400

            elif filename.endswith(('.xls', '.xlsx')):
                try:
                    # Leer archivo Excel (.xls o .xlsx)
                    file_stream = io.BytesIO(file.read())
                    dataframe = pd.read_excel(file_stream, engine='openpyxl')
                except ValueError as e:
                    return jsonify({'error': f'Error al leer el archivo Excel: {str(e)}'}), 400
                except Exception as e:
                    return jsonify({'error': f'Error desconocido al leer el archivo Excel: {str(e)}'}), 400

            elif filename.endswith('.txt'):
                try:
                    content = file.stream.read().decode("utf-8")
                    dataframe = pd.read_csv(io.StringIO(content), delimiter=r'\s+')
                except UnicodeDecodeError:
                    content = file.stream.read().decode("ISO-8859-1")
                    dataframe = pd.read_csv(io.StringIO(content), delimiter=r'\s+')
            else:
                return jsonify({'error': "Formato de archivo no soportado. Proporcione un archivo CSV, XLSX, XLS o TXT."}), 400

            # Verificar si el DataFrame se cargó correctamente
            if dataframe.empty:
                return jsonify({'error': 'El archivo está vacío o no se pudo procesar correctamente.'}), 400

            # Normalizar encabezados quitando espacios adicionales
            dataframe.columns = dataframe.columns.str.strip()

            # Manejo de celdas vacías
            dataframe = dataframe.replace("N/A", np.nan)
            dataframe = dataframe.dropna()  # Eliminar filas con valores nulos

            # Intentar convertir columnas numéricas interpretadas como texto
            for column in dataframe.columns:
                try:
                    dataframe[column] = pd.to_numeric(dataframe[column].str.replace(',', '.'), errors='ignore')
                except AttributeError:
                    # Si no es un string, continuar sin cambios
                    continue

            # Clasificar las columnas entre numéricas y categóricas
            numeric_columns = dataframe.select_dtypes(include=['number']).columns.tolist()
            categorical_columns = dataframe.select_dtypes(exclude=['number']).columns.tolist()

            # Filtrar solo las columnas categóricas con exactamente dos categorías
            binary_categorical_columns = [
                col for col in categorical_columns if dataframe[col].nunique() == 2
            ]

            return jsonify({
                'message': 'Archivo cargado exitosamente',
                'numeric_columns': numeric_columns,
                'categorical_columns': categorical_columns,
                'binary_categorical_columns': binary_categorical_columns
            })

        except UnicodeDecodeError:
            return jsonify({'error': 'Error de codificación. Asegúrese de que el archivo esté en formato UTF-8 o ISO-8859-1.'}), 400
        except pd.errors.ParserError:
            return jsonify({'error': 'Error al analizar el archivo. Verifique el delimitador y el formato del archivo.'}), 400
        except Exception as e:
            return jsonify({'error': str(e)}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 400


# Función para gestionar NA
def clean_results(result_dict):
    """Reemplaza NaN e infinitos por None en un diccionario de resultados."""
    for key, value in result_dict.items():
        if isinstance(value, dict):
            # Llamada recursiva para limpiar diccionarios anidados
            clean_results(value)
        elif isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            result_dict[key] = None
    return result_dict

# Función para reemplazar NaN e infinitos por None
def replace_invalid_values(data):
    if isinstance(data, dict):
        return {k: (None if pd.isna(v) or v in [np.inf, -np.inf] else v) for k, v in data.items()}
    return data

# Modificar los nombres de las columnas para que sean válidos
def clean_column_names(df):
    """Limpia los nombres de las columnas en un DataFrame para que sean válidos"""
    original_names = df.columns
    clean_names = [
        re.sub(r'[^a-zA-Z0-9_]', '_', col).lower()  # Reemplaza caracteres inválidos con '_'
        for col in original_names
    ]
    df.columns = clean_names
    name_mapping = dict(zip(clean_names, original_names))  # Mapeo entre nombres limpios y originales
    return name_mapping

# REGRESIÓN SIMPLE
@app.route('/run_regression', methods=['POST'])
def run_regression():
    global dataframe
    try:
        if dataframe is None:
            return jsonify({'error': 'No se ha cargado ningún archivo para analizar.'}), 400

        # Limpia los nombres de las columnas antes de procesar
        name_mapping = clean_column_names(dataframe)

        data = request.get_json()
        response_variable = data.get('response_variable')
        covariates = data.get('covariates')

        if not response_variable or not covariates:
            return jsonify({'error': 'Variables insuficientes para la regresión.'}), 400
        
        # Normaliza los nombres de las variables
        response_variable_clean = re.sub(r'[^a-zA-Z0-9_]', '_', response_variable).lower()
        covariates_clean = [re.sub(r'[^a-zA-Z0-9_]', '_', cov).lower() for cov in covariates]

        # Verificar que la variable de respuesta y las covariables existen en el DataFrame
        if response_variable_clean not in dataframe.columns:
            return jsonify({'error': f'La variable de respuesta {response_variable} no existe en los datos.'}), 400

        for cov_clean, cov_original in zip(covariates_clean, covariates):
            if cov_clean not in dataframe.columns:
                return jsonify({'error': f'La covariable {cov_original} no existe en los datos.'}), 400

        # Preparar el DataFrame para el modelo
        df = dataframe[covariates_clean + [response_variable_clean]].copy()
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)

        # Definir la fórmula para el modelo completo
        formula = f"{response_variable_clean} ~ " + " + ".join(covariates_clean)

        # Ajustar el modelo completo usando statsmodels
        model_full = smf.ols(formula=formula, data=df).fit()

        # Calcular el RMSE
        predictions = model_full.predict(df)
        rmse = np.sqrt(mean_squared_error(df[response_variable_clean], predictions))

        # Extraer los coeficientes e intercepto
        coefficients = model_full.params.to_dict()
        intercept = coefficients.pop('Intercept', None)

        # Calcular R-cuadrado
        r_squared = model_full.rsquared

        # Realizar ANOVA para obtener p-valores generales
        anova_results = sm.stats.anova_lm(model_full, typ=2)
        # Excluir el valor de 'Residual' al crear el diccionario de p-valores generales
        p_values_general = {k: v for k, v in anova_results["PR(>F)"].to_dict().items() if k != "Residual"}


        # Extraer p-valores específicos para cada nivel de las variables categóricas
        p_values_specific = model_full.pvalues.to_dict()
        p_values_specific.pop('Intercept', None)

        # Reemplazar NaN e infinitos por None
        coefficients = replace_invalid_values(coefficients)
        p_values_general = replace_invalid_values(p_values_general)
        p_values_specific = replace_invalid_values(p_values_specific)

        # Reconstruir los nombres originales para los resultados
        coefficients_original = {name_mapping.get(k, k): v for k, v in coefficients.items()}
        p_values_specific_original = {name_mapping.get(k, k): v for k, v in p_values_specific.items()}
        p_values_general_original = {name_mapping.get(k, k): v for k, v in p_values_general.items()}

        # Preparar el resumen del modelo
        regression_result = {
            "coefficients": coefficients_original,
            "intercept": intercept,
            "rmse": rmse,
            "r_squared": r_squared,
            "p_values_general": p_values_general_original,
            "p_values_specific": p_values_specific_original
        }

        return jsonify({
            'regression_result': regression_result,
            'warnings': []
        })

    except Exception as e:
        if "invalid syntax" in str(e).lower():
            error_message = (
                "Error en los nombres de las variables. Revisa que no incluyan caracteres especiales "
                "(tildes, espacios, símbolos)."
            )
        else:
            error_message = str(e)
        return jsonify({'error': error_message}), 400


    
# 3. T-Test
@app.route('/run_ttest', methods=['POST'])
def run_ttest():
    global dataframe
    try:
        data = request.get_json()
        numeric_column = data.get('numeric_column')
        categorical_column = data.get('categorical_column')
        paired = data.get('paired', False)
        alternative = data.get('alternative', 'two-sided')

        if numeric_column not in dataframe.columns or categorical_column not in dataframe.columns:
            return jsonify({'error': 'Las columnas especificadas no se encontraron en los datos.'}), 400

        # Agrupar los datos por la columna categórica
        groups = dataframe.groupby(categorical_column)[numeric_column].apply(list)

        # Verificar que haya al menos dos grupos
        if len(groups) < 2:
            return jsonify({'error': 'Datos insuficientes para realizar el T-Test. Se requieren al menos dos categorías.'}), 400

        category_names = groups.index.tolist()

        # Realizar la prueba T con el tipo de prueba especificado
        if paired:
            t_stat, p_value = stats.ttest_rel(groups.iloc[0], groups.iloc[1], alternative=alternative)
        else:
            t_stat, p_value = stats.ttest_ind(groups.iloc[0], groups.iloc[1], alternative=alternative, equal_var=False)

        # Evaluar la significancia según el valor p
        significance = "significativo" if p_value < 0.05 else "no significativo"
        decision = "Rechazar la hipótesis nula" if p_value < 0.05 else "No rechazar la hipótesis nula"

        result = {
            'test': 'T-Test' + (' pareado' if paired else ''),
            't_statistic': t_stat,
            'p_value': p_value,
            'significance': significance,
            'decision': decision,
            'category1': category_names[0],
            'category2': category_names[1],
            'alternative': alternative
        }

        # Limpiar los resultados antes de enviarlos
        result = clean_results(result)

        return jsonify({'result': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 400


# DEVOLVER A LA APP EL NOMBRE DE LAS CATEGORÍAS
@app.route('/get_category_names', methods=['POST'])
def get_category_names():
    global dataframe
    try:
        data = request.get_json()
        categorical_column = data.get('categorical_column')

        if categorical_column not in dataframe.columns:
            return jsonify({'error': 'La columna categórica especificada no se encontró.'}), 400

        # Obtener los nombres de las categorías
        category_names = list(dataframe[categorical_column].dropna().unique())

        if len(category_names) < 2:
            return jsonify({'error': 'Datos insuficientes para obtener los nombres de las categorías.'}), 400

        return jsonify({'category_names': category_names[:2]})

    except Exception as e:
        return jsonify({'error': str(e)}), 400



# 4. ANOVA
# Ruta para ANOVA
@app.route('/run_anova', methods=['POST'])
def run_anova():
    global dataframe
    try:
        data = request.get_json()
        numeric_column = data.get('numeric_column')
        categorical_column = data.get('categorical_column')
        multiple_comparisons = data.get('multipleComparisons', False)

        # Verificar que las columnas existan
        if numeric_column not in dataframe.columns or categorical_column not in dataframe.columns:
            return jsonify({'error': 'Las columnas especificadas no se encontraron en los datos.'}), 400

        # Agrupar los datos por la columna categórica
        groups = dataframe.groupby(categorical_column)[numeric_column].apply(list)

        # Verificar si hay suficientes datos para cada grupo
        if any(len(values) < 2 for values in groups):
            return jsonify({'error': 'Hay grupos con datos insuficientes para realizar ANOVA.'}), 400

        # Realizar ANOVA
        f_statistic, p_value = f_oneway(*groups)

        # Crear respuesta inicial
        result = {
            'f_statistic': f_statistic,
            'p_value': p_value,
            'num_groups': len(groups),
            'total_observations': sum(len(group) for group in groups),
        }

        # Si se habilitan las comparaciones múltiples, realizar Tukey HSD
        if multiple_comparisons:
            if len(groups) < 3:
                return jsonify({'error': 'Se requieren al menos tres grupos para realizar comparaciones múltiples (Tukey HSD).'}), 400

            try:
                # Preparar los datos para Tukey HSD
                all_data = []
                labels = []
                for i, group in enumerate(groups):
                    # Verificar que el grupo tenga al menos un valor
                    if len(group) == 0:
                        return jsonify({'error': f'El grupo {groups.index[i]} no contiene datos suficientes.'}), 400

                    all_data.extend(group)
                    labels.extend([groups.index[i]] * len(group))

                # Crear DataFrame para Tukey HSD
                df = pd.DataFrame({'value': all_data, 'group': labels})

                # Realizar Tukey HSD
                tukey = mc.pairwise_tukeyhsd(df['value'], df['group'], alpha=0.05)

                # Formatear los resultados de Tukey HSD
                tukey_summary = []
                for res in tukey.summary().data[1:]:  # Ignorar la cabecera
                    comparison = f"{res[0]} vs {res[1]}"
                    mean_diff = f"{res[2]:.2f}"
                    p_adj = f"{res[3]:.3f}"
                    ci_lower = f"{res[4]:.2f}"
                    ci_upper = f"{res[5]:.2f}"
                    reject_h0 = "Sí" if res[6] else "No"

                    tukey_summary.append({
                        'comparison': comparison,
                        'mean_difference': mean_diff,
                        'p_value_adjusted': p_adj,
                        'ci_lower': ci_lower,
                        'ci_upper': ci_upper,
                        'reject_h0': reject_h0
                    })

                # Agregar el resumen de Tukey al resultado
                result['tukey'] = tukey_summary

            except Exception as e:
                return jsonify({'error': f'Error al ejecutar Tukey HSD: {str(e)}'}), 500

        # Limpiar los resultados antes de enviarlos
        result = clean_results(result)

        return jsonify({'result': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 400



# 5. Chi-Square
@app.route('/run_chisquare', methods=['POST'])
def run_chisquare():
    global dataframe
    try:
        data = request.get_json()
        categorical_column1 = data.get('categorical_column1')
        categorical_column2 = data.get('categorical_column2')
        show_contingency_table = data.get('show_contingency_table', False)

        if categorical_column1 not in dataframe.columns or categorical_column2 not in dataframe.columns:
            return jsonify({'error': 'Las columnas especificadas no se encontraron en los datos.'}), 400

        # Crear la tabla de contingencia
        contingency_table = pd.crosstab(dataframe[categorical_column1], dataframe[categorical_column2])

        # Realizar el test de Chi-Cuadrado
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)

        # Evaluar la significancia
        significance = "significativo" if p_value < 0.05 else "no significativo"
        decision = "Rechazar la hipótesis nula" if p_value < 0.05 else "No rechazar la hipótesis nula"

        # Preparar el resultado
        result = {
            'test': 'Chi-Square Test',
            'chi_square_statistic': chi2,
            'p_value': p_value,
            'degrees_of_freedom': dof,
            'significance': significance,
            'decision': decision
        }

        # Incluir la tabla de contingencia si se solicitó
        if show_contingency_table:
            result['contingency_table'] = contingency_table.to_dict()

        return jsonify({'result': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 400


# 6. Shapiro-Wilk 
@app.route('/run_shapiro', methods=['POST'])
def run_shapiro():
    global dataframe
    try:
        # Obtener los parámetros de la solicitud
        data = request.get_json()
        numeric_column = data.get('sample')
        group_column = data.get('group_column', None)

        # Validar que la columna numérica exista
        if numeric_column not in dataframe.columns:
            return jsonify({'error': 'La columna numérica especificada no se encontró en los datos.'}), 400

        # Si se especifica una columna de grupo, hacer Shapiro-Wilk por grupo
        if group_column and group_column in dataframe.columns:
            groups = dataframe.groupby(group_column)[numeric_column].apply(list)
            shapiro_results = {}

            for group, values in groups.items():
                if len(values) < 3:
                    shapiro_results[group] = {'w_statistic': None, 'p_value': 'Datos insuficientes'}
                else:
                    w_stat, p_value = shapiro(values)
                    # Resultado por grupos
                    shapiro_results[group] = clean_results({'w_statistic': w_stat, 'p_value': p_value})

            return jsonify({'type': 'grouped', 'result': shapiro_results})

        # Si no se especifica grupo, hacer Shapiro-Wilk global
        values = dataframe[numeric_column].dropna()
        if len(values) < 3:
            return jsonify({'error': 'Datos insuficientes para realizar la prueba de Shapiro-Wilk.'}), 400

        w_stat, p_value = shapiro(values)
        # Resultado global
        result = {'w_statistic': w_stat, 'p_value': p_value}
        result = clean_results(result)
        return jsonify({'type': 'global', 'result': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 400
    

# 7. Kolmogorov 
@app.route('/run_kolmogorov', methods=['POST'])
def run_kolmogorov():
    global dataframe
    try:
        # Obtener los parámetros de la solicitud
        data = request.get_json()
        numeric_column = data.get('sample')
        group_column = data.get('group_column', None)

        # Validar que la columna numérica exista
        if numeric_column not in dataframe.columns:
            return jsonify({'error': 'La columna numérica especificada no se encontró en los datos.'}), 400

        # Si se especifica una columna de grupo, hacer Kolmogorov-Smirnov por grupo
        if group_column and group_column in dataframe.columns:
            groups = dataframe.groupby(group_column)[numeric_column].apply(list)
            ks_results = {}

            for group, values in groups.items():
                if len(values) < 3:
                    ks_results[group] = {'ks_statistic': None, 'p_value': 'Datos insuficientes'}
                else:
                    # Realizar prueba de Kolmogorov-Smirnov asumiendo distribución normal
                    ks_stat, p_value = stats.kstest(values, 'norm', args=(np.mean(values), np.std(values)))
                    # Resultado por grupos
                    ks_results[group] = clean_results({'ks_statistic': ks_stat, 'p_value': p_value})

            return jsonify({'type': 'grouped', 'result': ks_results})

        # Si no se especifica grupo, hacer Kolmogorov-Smirnov global
        values = dataframe[numeric_column].dropna()
        if len(values) < 3:
            return jsonify({'error': 'Datos insuficientes para realizar la prueba de Kolmogorov-Smirnov.'}), 400

        # Realizar prueba de Kolmogorov-Smirnov asumiendo distribución normal
        ks_stat, p_value = stats.kstest(values, 'norm', args=(np.mean(values), np.std(values)))
        # Resultado global
        result = {'ks_statistic': ks_stat, 'p_value': p_value}
        result = clean_results(result)
        return jsonify({'type': 'global', 'result': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# 8. Mann-Whitney U Test
@app.route('/run_mannwhitney', methods=['POST'])
def run_mannwhitney():
    global dataframe
    try:
        data = request.get_json()
        numeric_column = data.get('numeric_column')
        categorical_column = data.get('categorical_column')
        alternative = data.get('alternative', 'two-sided')

        if numeric_column not in dataframe.columns or categorical_column not in dataframe.columns:
            return jsonify({'error': 'Las columnas especificadas no se encontraron en los datos.'}), 400

        # Agrupar los datos por la columna categórica
        groups = dataframe.groupby(categorical_column)[numeric_column].apply(list)

        # Verificar que haya al menos dos grupos
        if len(groups) < 2:
            return jsonify({'error': 'Datos insuficientes para realizar el Mann-Whitney U Test. Se requieren al menos dos categorías.'}), 400

        category_names = groups.index.tolist()

        # Ejecutar el test Mann-Whitney U
        u_stat, p_value = stats.mannwhitneyu(groups.iloc[0], groups.iloc[1], alternative=alternative)

        # Evaluar la significancia según el valor p
        significance = "significativo" if p_value < 0.05 else "no significativo"
        decision = "Rechazar la hipótesis nula" if p_value < 0.05 else "No rechazar la hipótesis nula"

        result = {
            'test': 'Mann-Whitney U Test',
            'u_statistic': u_stat,
            'p_value': p_value,
            'significance': significance,
            'decision': decision,
            'category1': category_names[0],
            'category2': category_names[1],
            'alternative': alternative
        }

        # Limpiar los resultados antes de enviarlos
        result = clean_results(result)

        return jsonify({'result': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# 9. Wilcoxon
@app.route('/run_wilcoxon', methods=['POST'])
def run_wilcoxon():
    global dataframe
    try:
        data = request.get_json()
        numeric_column = data.get('numeric_column')
        categorical_column = data.get('categorical_column')
        alternative = data.get('alternative', 'two-sided')

        if numeric_column not in dataframe.columns or categorical_column not in dataframe.columns:
            return jsonify({'error': 'Las columnas especificadas no se encontraron en los datos.'}), 400

        # Agrupar los datos por la columna categórica
        groups = dataframe.groupby(categorical_column)[numeric_column].apply(list)

        # Verificar que haya exactamente dos grupos
        if len(groups) != 2:
            return jsonify({'error': 'La variable categórica debe tener exactamente dos categorías.'}), 400

        category_names = groups.index.tolist()

        # Ajustar los tamaños de muestra si son diferentes
        min_size = min(len(groups.iloc[0]), len(groups.iloc[1]))
        group1 = groups.iloc[0][:min_size]
        group2 = groups.iloc[1][:min_size]

        # Ejecutar el test de Wilcoxon
        w_stat, p_value = stats.wilcoxon(group1, group2, alternative=alternative)

        # Evaluar la significancia según el valor p
        significance = "significativo" if p_value < 0.05 else "no significativo"
        decision = "Rechazar la hipótesis nula" if p_value < 0.05 else "No rechazar la hipótesis nula"

        # Crear el resultado con el mismo formato que Mann-Whitney U
        result = {
            'test': 'Wilcoxon Signed-Rank Test',
            'w_statistic': w_stat,
            'p_value': p_value,
            'significance': significance,
            'decision': decision,
            'category1': category_names[0],
            'category2': category_names[1],
            'alternative': alternative
        }

        return jsonify({'result': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# 10. Kruskal Wallis
@app.route('/run_kruskal_wallis', methods=['POST'])
def run_kruskal_wallis():
    global dataframe
    try:
        data = request.get_json()
        numeric_column = data.get('numeric_column')
        categorical_column = data.get('categorical_column')
        multiple_comparisons = data.get('multiple_comparisons', False)

        # Verificar que las columnas existan
        if numeric_column not in dataframe.columns or categorical_column not in dataframe.columns:
            return jsonify({'error': 'Las columnas especificadas no se encontraron en los datos.'}), 400

        # Agrupar los datos por la columna categórica
        groups = dataframe.groupby(categorical_column)[numeric_column].apply(list)

        # Verificar si hay suficientes datos para cada grupo
        if any(len(values) < 2 for values in groups):
            return jsonify({'error': 'Hay grupos con datos insuficientes para realizar Kruskal-Wallis.'}), 400

        # Realizar el test de Kruskal-Wallis
        h_statistic, p_value = stats.kruskal(*groups)

        # Crear respuesta inicial
        result = {
            'h_statistic': h_statistic,
            'p_value': p_value,
            'num_groups': len(groups),
            'total_observations': sum(len(group) for group in groups),
        }

        # Si se habilitan las comparaciones múltiples, realizar el test de Dunn
        if multiple_comparisons:
            try:
                # Preparar los datos para el test de Dunn
                all_data = []
                labels = []
                for i, group in enumerate(groups):
                    all_data.extend(group)
                    labels.extend([groups.index[i]] * len(group))

                # Crear DataFrame para el test de Dunn
                df = pd.DataFrame({'value': all_data, 'group': labels})

                # Realizar el test de Dunn
                dunn = sp.posthoc_dunn(df, val_col='value', group_col='group', p_adjust='bonferroni')

                # Formatear los resultados del test de Dunn
                dunn_summary = []
                for i, row in dunn.iterrows():
                    for j, p_val in row.items():
                        if i != j:
                            comparison = f"{i} vs {j}"
                            p_adj = f"{p_val:.3f}"
                            reject_h0 = "Sí" if p_val < 0.05 else "No"
                            dunn_summary.append({
                                'comparison': comparison,
                                'p_value_adjusted': p_adj,
                                'reject_h0': reject_h0
                            })


                # Agregar el resumen del test de Dunn al resultado
                result['dunn'] = dunn_summary

            except Exception as e:
                return jsonify({'error': f'Error al ejecutar el test de Dunn: {str(e)}'}), 500

        return jsonify({'result': result})

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

# Ruta para la prueba t de Student con opción pareado y unilateral/bilateral
@app.route('/api/ttest', methods=['POST'])
def t_test():
    data = request.get_json()
    sample1 = data['sample1']
    sample2 = data['sample2']
    paired = data.get('paired', False)
    alternative = data.get('alternative', 'two-sided')  # Configuración de unilateral/bilateral

    if paired:
        stat, p_value = stats.ttest_rel(sample1, sample2, alternative=alternative)
    else:
        stat, p_value = stats.ttest_ind(sample1, sample2, alternative=alternative)

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

# Mann-Whitney U Test con opción unilateral/bilateral
@app.route('/api/mannwhitney', methods=['POST'])
def mann_whitney():
    data = request.get_json()
    sample1 = data['sample1']
    sample2 = data['sample2']
    alternative = data.get('alternative', 'two-sided')  # Configuración de unilateral/bilateral

    # Realizar la prueba Mann-Whitney
    stat, p_value = stats.mannwhitneyu(sample1, sample2, alternative=alternative)

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
        p_value_adjustment = data.get('pValueAdjustmentMethod', 'bonferroni').lower()  # Método de ajuste por defecto: Bonferroni

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
                if p_value_adjustment not in ['bonferroni', 'holm']:
                    return jsonify({'error': 'Método de ajuste no válido. Use "bonferroni" o "holm".'}), 400
                
                dunn = sp.posthoc_dunn(df, val_col='value', group_col='group', p_adjust=p_value_adjustment)

                # Procesar los resultados de Dunn
                dunn_summary = f"Comparaciones múltiples (Dunn con corrección {p_value_adjustment.capitalize()}):\n"
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


# Friedman Test con comparaciones múltiples
@app.route('/api/friedman', methods=['POST'])
def friedman_test():
    try:
        data = request.get_json()
        groups = data.get('groups', [])
        multiple_comparisons = data.get('multipleComparisons', False)

        # Validación de que haya al menos 3 grupos
        if len(groups) < 3:
            return jsonify({'error': 'Se requieren al menos 3 grupos para la prueba de Friedman.'}), 400

        # Validar que todos los grupos tengan el mismo número de observaciones
        num_observations = len(groups[0])
        if not all(len(g) == num_observations for g in groups):
            return jsonify({'error': 'Todos los grupos deben tener el mismo número de observaciones.'}), 400

        # Realizar la prueba de Friedman
        stat, p_value = stats.friedmanchisquare(*groups)

        # Determinar la significancia del valor p
        if p_value < 0.05:
            significance = "significant"
            reject_null = "Reject the null hypothesis"
        elif p_value < 0.1:
            significance = "marginally significant"
            reject_null = "Potential rejection of the null hypothesis"
        else:
            significance = "not significant"
            reject_null = "Do not reject the null hypothesis"

        # Resultados básicos de Friedman
        friedman_results = {
            'test': 'Friedman',
            'statistic': stat,
            'pValue': p_value,
            'significance': significance,
            'decision': reject_null,
            'num_groups': len(groups),
            'total_observations': num_observations * len(groups)
        }

        # Comparaciones múltiples si está habilitado
        if multiple_comparisons:
            try:
                # Preparar los datos en un DataFrame en formato de bloques
                # Cada columna será un grupo y cada fila una observación
                df = pd.DataFrame({f'Grupo {i+1}': group for i, group in enumerate(groups)})

                # Realizar la prueba de Nemenyi
                nemenyi = sp.posthoc_nemenyi_friedman(df)

                # Procesar los resultados de Nemenyi en formato de texto
                nemenyi_summary = "Comparaciones múltiples (Nemenyi):\n"
                for i in range(len(nemenyi)):
                    for j in range(i + 1, len(nemenyi)):
                        nemenyi_summary += (
                            f"----------------------------\n"
                            f"Comparación: Grupo {i+1} vs Grupo {j+1}\n"
                            f"  • p-Value ajustado: {nemenyi.iloc[i, j]:.3f}\n"
                        )

                # Agregar el resumen de Nemenyi al resultado
                friedman_results['nemenyi'] = nemenyi_summary

            except Exception as e:
                return jsonify({'error': f'Error al ejecutar la prueba de Nemenyi: {str(e)}'}), 500

        return jsonify(friedman_results)

    except Exception as e:
        return jsonify({'error': f'Error interno del servidor: {str(e)}'}), 500



# Fisher exact test
@app.route('/api/fisher', methods=['POST'])
def fisher_test():
    try:
        data = request.get_json()
        observed = data['observed']

        # Validar que sea una tabla 2x2
        if len(observed) != 2 or len(observed[0]) != 2:
            return jsonify({'error': 'Fisher’s exact test requires a 2x2 contingency table.'}), 400

        # Convertir la tabla a un array de Numpy para facilitar el manejo
        try:
            table = np.array(observed, dtype=float)  # Convertir a flotantes
        except ValueError:
            return jsonify({'error': 'All elements in "observed" must be numeric.'}), 400


        # Verificar si hay ceros
        if np.any(table == 0):
            # Aplicar suavizado aditivo sumando 1 a cada celda
            table += 1
            used_smoothing = True
        else:
            used_smoothing = False

        # Calcular oddsratio y p-value
        oddsratio, p_value = stats.fisher_exact(table)

        # Manejar valores infinitos en oddsratio
        if np.isinf(oddsratio):
            oddsratio = "Infinity"

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

        if used_smoothing:
            contiene_cero = "Se aplicó un suavizado aditivo para manejar los 0. Esto agregará 1 a todas las celdas de la tabla y permitirá realizar la prueba."
        else: 
            contiene_cero = ""
         

        response = {
            'test': 'Fisher',
            'oddsratio': oddsratio,
            'pValue': p_value,
            'significance': significance,
            'decision': reject_null,
        }

        if used_smoothing:
            response['contiene0'] = "Se aplicó un suavizado aditivo para manejar los 0. Esto agregará 1 a todas las celdas de la tabla y permitirá realizar la prueba."

        return jsonify(response)


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
        # Obtener datos de la solicitud
        data = request.get_json()
        if 'observed' not in data or not isinstance(data['observed'], list):
            return jsonify({'error': 'The "observed" field is required and must be a list.'}), 400
        
        observed = data['observed']

        # Validar que la tabla no esté vacía
        if not observed or not all(isinstance(row, list) and row for row in observed):
            return jsonify({'error': 'The "observed" data must be a non-empty list of lists.'}), 400

        # Validar que la tabla tiene al menos 3 tratamientos/condiciones
        if len(observed[0]) < 3:
            return jsonify({'error': 'Cochran’s Q test requires at least 3 treatments/conditions.'}), 400

        # Validar que la tabla solo contiene valores binarios (0 y 1)
        if any(
            not isinstance(value, (int, float)) or value not in [0, 1]
            for row in observed
            for value in row
        ):
            return jsonify({'error': 'The "observed" data must only contain binary values (0 and 1).'}), 400

        # Ejecutar la prueba de Cochran
        result = cochrans_q(observed)

        # Validar que el resultado no esté vacío
        if result is None:
            return jsonify({'error': 'The test did not return any result. Please check your input.'}), 500

        # Determinar la significancia del p-valor
        if result.pvalue < 0.05:
            significance = "significant"
            reject_null = "Reject the null hypothesis"
        elif result.pvalue < 0.1:
            significance = "marginally significant"
            reject_null = "Potential rejection of the null hypothesis"
        else:
            significance = "not significant"
            reject_null = "Do not reject the null hypothesis"

        # Devolver la respuesta con los resultados
        return jsonify({
            'test': 'Cochran\'s Q',
            'statistic': result.statistic,
            'pValue': result.pvalue,
            'significance': significance,
            'decision': reject_null
        })

    except KeyError as e:
        return jsonify({'error': f'Missing required key: {str(e)}'}), 400

    except Exception as e:
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500



# Wilcoxon Signed-Rank Test con opción unilateral/bilateral
@app.route('/api/wilcoxon', methods=['POST'])
def wilcoxon_test():
    try:
        data = request.get_json()
        sample1 = data['sample1']
        sample2 = data['sample2']
        alternative = data.get('alternative', 'two-sided')  # Configuración de unilateral/bilateral

        # Asegurarse de que ambas muestras tengan el mismo número de observaciones
        if len(sample1) != len(sample2):
            return jsonify({'error': 'Las dos muestras deben tener el mismo número de observaciones para la prueba Wilcoxon.'}), 400

        # Ejecutar la prueba de Wilcoxon
        stat, p_value = stats.wilcoxon(sample1, sample2, alternative=alternative)

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


# Ruta para la correlación de Pearson
@app.route('/api/correlation/pearson', methods=['POST'])
def pearson_correlation():
    data = request.get_json()
    sample1 = data.get('sample1', [])
    sample2 = data.get('sample2', [])
    show_plot = data.get('showPlot', False)  # Nuevo parámetro para verificar si se debe generar el gráfico

    # Validar que ambas muestras tengan datos
    if not sample1 or not sample2:
        return jsonify({'error': 'Ambas muestras deben contener datos.'}), 400

    try:
        # Convertir muestras a arrays numpy para procesamiento
        sample1 = np.array(sample1)
        sample2 = np.array(sample2)

        # Calcular la correlación de Pearson
        stat, p_value = stats.pearsonr(sample1, sample2)

        # Generar el gráfico solo si se solicita
        scatter_plot_encoded = None
        if show_plot:
            # Crear gráfico de dispersión con línea de tendencia
            plt.figure(figsize=(6, 4))
            plt.scatter(sample1, sample2, color='skyblue', edgecolor='black', label='Datos')
            
            # Ajustar la línea de tendencia
            slope, intercept = np.polyfit(sample1, sample2, 1)
            plt.plot(sample1, slope * sample1 + intercept, color='red', linewidth=2, label='Línea de tendencia')

            # Personalizar el gráfico
            plt.title('Gráfico de Dispersión con Línea de Tendencia')
            plt.xlabel('Sample 1')
            plt.ylabel('Sample 2')
            plt.legend()
            plt.tight_layout()

            # Convertir gráfico a base64
            img = io.BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            scatter_plot_encoded = base64.b64encode(img.getvalue()).decode()
            plt.close()

        # Devolver resultados en JSON
        response = {
            'test': 'Pearson Correlation',
            'statistic': stat,
            'pValue': p_value,
            'significance': 'significant' if p_value < 0.05 else 'not significant',
            'decision': 'Reject the null hypothesis' if p_value < 0.05 else 'Do not reject the null hypothesis'
        }

        # Incluir el gráfico solo si se generó
        if scatter_plot_encoded:
            response['scatter_plot'] = scatter_plot_encoded

        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Ruta para la correlación de Spearman
@app.route('/api/correlation/spearman', methods=['POST'])
def spearman_correlation():
    data = request.get_json()
    sample1 = data.get('sample1', [])
    sample2 = data.get('sample2', [])
    show_plot = data.get('showPlot', False)  # Nuevo parámetro para verificar si se debe generar el gráfico

    # Validar que ambas muestras tengan datos
    if not sample1 or not sample2:
        return jsonify({'error': 'Ambas muestras deben contener datos.'}), 400

    try:
        # Convertir muestras a arrays numpy para procesamiento
        sample1 = np.array(sample1)
        sample2 = np.array(sample2)

        # Calcular la correlación de Spearman
        stat, p_value = stats.spearmanr(sample1, sample2)

        # Generar el gráfico solo si se solicita
        scatter_plot_encoded = None
        if show_plot:
            # Crear gráfico de dispersión con línea de tendencia
            plt.figure(figsize=(6, 4))
            plt.scatter(sample1, sample2, color='blue', label='Datos')

            # Ajustar la línea de tendencia usando una regresión lineal
            m, b = np.polyfit(sample1, sample2, 1)
            plt.plot(sample1, m * sample1 + b, color='red', label='Línea de tendencia')

            plt.title('Gráfico de dispersión de Spearman')
            plt.xlabel('Muestra 1')
            plt.ylabel('Muestra 2')
            plt.legend()
            plt.tight_layout()

            # Guardar el gráfico en base64
            img = io.BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            scatter_plot_encoded = base64.b64encode(img.getvalue()).decode()
            plt.close()  # Cerrar el gráfico para liberar memoria

        # Devolver los resultados en formato JSON
        response = {
            'test': 'Spearman Correlation',
            'statistic': stat,
            'pValue': p_value,
            'significance': 'significant' if p_value < 0.05 else 'not significant',
            'decision': 'Reject the null hypothesis' if p_value < 0.05 else 'Do not reject the null hypothesis'
        }

        # Incluir el gráfico solo si se generó
        if scatter_plot_encoded:
            response['scatter_plot'] = scatter_plot_encoded

        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Ruta para la prueba t de Welch (para muestras independientes con varianzas desiguales)
@app.route('/api/welch_ttest', methods=['POST'])
def welch_t_test():
    data = request.get_json()
    sample1 = data.get('sample1', [])
    sample2 = data.get('sample2', [])
    alternative = data.get('alternative', 'two-sided')  # Configuración de unilateral/bilateral

    # Validar que ambas muestras tengan datos
    if not sample1 or not sample2:
        return jsonify({'error': 'Ambas muestras deben contener datos.'}), 400

    # Realizar la prueba t de Welch con opción de prueba unilateral/bilateral
    stat, p_value = stats.ttest_ind(sample1, sample2, equal_var=False, alternative=alternative)

    # Determinar significancia según el valor p
    if p_value < 0.05:
        significance = "significant"
        decision = "Reject the null hypothesis"
    elif p_value < 0.1:
        significance = "marginally significant"
        decision = "Potential rejection of the null hypothesis"
    else:
        significance = "not significant"
        decision = "Do not reject the null hypothesis"

    return jsonify({
        'test': 'Welch T-Test',
        'statistic': stat,
        'pValue': p_value,
        'significance': significance,
        'decision': decision
    })



# Ruta para la prueba t de una media
@app.route('/api/one_sample_ttest', methods=['POST'])
def one_sample_t_test():
    data = request.get_json()
    sample = data.get('sample', [])
    population_mean = data.get('population_mean', 0)  # Valor de referencia

    # Validar que la muestra tenga datos
    if not sample:
        return jsonify({'error': 'La muestra debe contener datos.'}), 400

    # Realizar la prueba t de una muestra
    stat, p_value = stats.ttest_1samp(sample, population_mean)

    return jsonify({
        'test': 'One-Sample T-Test',
        'statistic': stat,
        'pValue': p_value,
        'significance': 'significant' if p_value < 0.05 else 'not significant',
        'decision': 'Reject the null hypothesis' if p_value < 0.05 else 'Do not reject the null hypothesis'
    })
    
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
        show_plot = data.get('showPlot', False)  # Nuevo parámetro para verificar si se debe generar el gráfico

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
        coefficients = replace_nan_with_none(result.params[1:].tolist())  # Coeficientes sin el intercepto

        # Separar p-valores
        intercept_pvalue = replace_nan_with_none([result.pvalues[0]])[0]  # p-valor del intercepto
        p_values = replace_nan_with_none(result.pvalues[1:].tolist())  # p-valores sin el intercepto

        # Estadísticos adicionales
        t_values = replace_nan_with_none(result.tvalues[1:].tolist())
        confidence_intervals = [replace_nan_with_none(ci) for ci in result.conf_int().tolist()[1:]]  # Intervalos de confianza sin el intercepto
        r_squared = result.rsquared

        # Generar el gráfico de regresión solo si se solicita
        scatter_plot_encoded = None
        if show_plot and predictors.shape[1] == 1:  # Solo generamos el gráfico si hay un predictor
            plt.figure(figsize=(6, 4))
            plt.scatter(predictors, response, color='blue', label='Datos')
            
            # Generar la línea de regresión
            predicted_values = intercept + coefficients[0] * predictors.flatten()
            plt.plot(predictors, predicted_values, color='red', label='Línea de Regresión')

            # Personalizar el gráfico
            plt.title('Regresión Lineal')
            plt.xlabel('Predictor')
            plt.ylabel('Respuesta')
            plt.legend()
            plt.tight_layout()

            # Guardar el gráfico en base64
            img = io.BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            scatter_plot_encoded = base64.b64encode(img.getvalue()).decode()
            plt.close()

        # Construir la respuesta JSON
        response = {
            'model': 'Regresión Lineal',
            'intercept': intercept,
            'intercept_pvalue': intercept_pvalue,
            'coefficients': coefficients,
            'p_values': p_values,
            't_values': t_values,
            'confidence_intervals': confidence_intervals,
            'r_squared': r_squared
        }

        # Incluir el gráfico solo si se generó
        if scatter_plot_encoded:
            response['scatter_plot'] = scatter_plot_encoded

        return jsonify(response)
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


# Nueva función para reemplazar NaN con None, manejando estructuras anidadas
def replace_nan_with_none_cox(data):
    if isinstance(data, list):
        # Si es una lista, aplicar recursivamente usando la propia función
        return [replace_nan_with_none_cox(item) for item in data]
    elif pd.isna(data):  # Verificar si el elemento es NaN
        return None
    else:
        return data

# Endpoint de la regresión de Cox
@app.route('/api/cox_regression', methods=['POST'])
def cox_regression():
    try:
        data = request.get_json()
        
        # Convertir datos a np.array
        predictors = np.array(data['predictors'])
        time = np.array(data['time'])
        event = np.array(data['event'])

        # Asegurarse de que los predictores sean una matriz 2D
        if predictors.ndim == 1:
            predictors = predictors.reshape(-1, 1)

        # Validar que el número de observaciones sea el mismo en predictores, tiempo y evento
        if not (predictors.shape[0] == time.shape[0] == event.shape[0]):
            raise ValueError("El número de observaciones debe coincidir en predictores, tiempo y evento.")

        # Convertir a DataFrame para el modelo de Cox
        df = pd.DataFrame(predictors, columns=[f'Predictor_{i+1}' for i in range(predictors.shape[1])])
        df['time'] = time
        df['event'] = event

        # Ajustar el modelo de regresión de Cox
        cph = CoxPHFitter()
        cph.fit(df, duration_col='time', event_col='event')
        
        # Extraer los resultados usando la función de reemplazo de NaNs
        coefficients = replace_nan_with_none_cox(cph.params_.values.tolist())
        p_values = replace_nan_with_none_cox(cph.summary['p'].values.tolist())
        confidence_intervals = replace_nan_with_none_cox(cph.confidence_intervals_.values.tolist())

        # Devolver los resultados en formato JSON
        return jsonify({
            'model': 'Regresión de Cox',
            'coefficients': coefficients,
            'p_values': p_values,
            'confidence_intervals': confidence_intervals
        })
    except Exception as e:
        print(f"Error en regresión de Cox: {str(e)}")
        return jsonify({'error': str(e)}), 500



# Regresión de Poisson
@app.route('/api/poisson_regression', methods=['POST'])
def poisson_regression():
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

        # Ajustar el modelo de regresión de Poisson
        model = sm.GLM(response, X, family=sm.families.Poisson())
        result = model.fit()

        # Separar el intercepto del resto de los coeficientes
        intercept = result.params[0]  # El intercepto es el primer valor
        coefficients = result.params[1:].tolist()  # Coeficientes sin el intercepto

        # Separar p-valores
        intercept_pvalue = replace_nan_with_none([result.pvalues[0]])[0]  # p-valor del intercepto
        p_values = replace_nan_with_none(result.pvalues[1:].tolist())  # p-valores sin el intercepto

        # Estadísticos adicionales
        z_values = replace_nan_with_none(result.tvalues[1:].tolist())
        confidence_intervals = result.conf_int().tolist()[1:]  # Intervalos de confianza sin el intercepto

        # Devolver los resultados en formato JSON
        return jsonify({
            'model': 'Regresión de Poisson',
            'intercept': intercept,
            'intercept_pvalue': intercept_pvalue,
            'coefficients': coefficients,
            'p_values': p_values,
            'z_values': z_values,
            'confidence_intervals': confidence_intervals
        })
    except Exception as e:
        print(f"Error en regresión de Poisson: {str(e)}")
        return jsonify({'error': str(e)}), 500



# Ruta para llamar a Render y que no se apague
@app.route('/ping', methods=['HEAD', 'GET'])
def ping():
    return '', 200

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)











