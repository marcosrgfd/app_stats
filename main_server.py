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
from io import BytesIO

# MODELS
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from lifelines import CoxPHFitter

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import csv

import statsmodels.formula.api as smf

from flask_cors import CORS

import re  # Para limpiar los nombres de columnas

import os

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
                raise ValueError("If the effect size is not provided, mean1, mean2, std_dev_group1, and std_dev_group2 must be provided.")

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
            raise ValueError("The effect size (f²) is required for linear regression calculation.")

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
                raise ValueError("If the effect size is not provided, mean1, mean2, std_dev_group1, and std_dev_group2 must be provided.")

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
            raise ValueError("Degrees of freedom (df) are required for Chi-square calculation.")

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
            raise ValueError("The effect size (correlation coefficient r) is required for the calculation.")

        # Verificar que el coeficiente de correlación esté en el rango válido
        if not -1 <= r <= 1:
            raise ValueError("The correlation coefficient (r) must be between -1 and 1.")

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
            raise ValueError("The effect size is required for the calculation.")

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
            raise ValueError("The effect size is required for the calculation.")

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

            # Verificar que data_series1 y category_series tengan la misma longitud
            if len(data_series1) != len(category_series):
                raise ValueError("The number of elements in data1 does not match the number of categories.")

            # Calcular estadísticas descriptivas por categoría
            grouped = data_series1.groupby(category_series)
            stats_by_category = {}

            for category, group in grouped:
                # Calcular estadísticas básicas
                stat = {
                    'count': float(group.size),
                    'mean': float(group.mean()) if group.size > 0 else None,
                    'median': float(group.median()) if group.size > 0 else None,
                    'std': float(group.std()) if group.size > 1 else None,  # std no es aplicable para grupos de tamaño 1
                    'min': float(group.min()) if group.size > 0 else None,
                    'max': float(group.max()) if group.size > 0 else None,
                }
                stats_by_category[category] = stat

            results['stats_by_category'] = stats_by_category


            # Generar gráficos específicos para el análisis con categorías
            if show_boxplot:
                plt.figure(figsize=(8, 6))
                sns.boxplot(x=category_series, y=data_series1, palette="Set2", width=0.4)
                plt.title('Boxplot by categories')
                plt.xlabel('Category')
                plt.ylabel('Value')
                plt.tight_layout()
                boxplot_img = io.BytesIO()
                plt.savefig(boxplot_img, format='png', bbox_inches='tight', dpi=100)
                boxplot_img.seek(0)
                results['boxplot_by_category'] = base64.b64encode(boxplot_img.getvalue()).decode()
                plt.close()

            if show_violinplot:
                plt.figure(figsize=(8, 6))
                sns.violinplot(x=category_series, y=data_series1, palette="Set2", width=0.8)
                plt.title('Violin plot by categories')
                plt.xlabel('Category')
                plt.ylabel('Value')
                plt.tight_layout()
                violin_img = io.BytesIO()
                plt.savefig(violin_img, format='png', bbox_inches='tight', dpi=100)
                violin_img.seek(0)
                results['violin_plot_by_category'] = base64.b64encode(violin_img.getvalue()).decode()
                plt.close()

            if show_raincloudplot:
                fig, ax = plt.subplots(figsize=(8, 6))
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
                ax.set_xlabel('Value')
                ax.set_title('Raincloud plot')
                plt.tight_layout()
                raincloud_img = io.BytesIO()
                plt.savefig(raincloud_img, format='png', bbox_inches='tight', dpi=100)
                raincloud_img.seek(0)
                results['raincloud_plot'] = base64.b64encode(raincloud_img.getvalue()).decode()
                plt.close()
            return results

        # Análisis para una sola muestra sin categorías
        def analyze_single_series(data_series, title_suffix):
            stat = {
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
                plt.title(f'Histogram of data {title_suffix}')
                plt.xlabel('Value')
                plt.ylabel('Frequency')
                plt.tight_layout()
                img = io.BytesIO()
                plt.savefig(img, format='png')
                img.seek(0)
                stat['histogram'] = base64.b64encode(img.getvalue()).decode()
                plt.close()

            if show_boxplot:
                plt.figure(figsize=(6, 4))
                plt.boxplot(data_series.dropna(), vert=False, patch_artist=True, boxprops=dict(facecolor='lightblue'))
                plt.title(f'Boxplot of data {title_suffix}')
                plt.xlabel('Value')
                plt.tight_layout()
                boxplot_img = io.BytesIO()
                plt.savefig(boxplot_img, format='png')
                boxplot_img.seek(0)
                stat['boxplot'] = base64.b64encode(boxplot_img.getvalue()).decode()
                plt.close()

            if show_density:
                plt.figure(figsize=(6, 4))
                sns.kdeplot(data_series.dropna(), shade=True, color='green')
                plt.title(f'Density plot {title_suffix}')
                plt.xlabel('Value')
                plt.tight_layout()
                density_img = io.BytesIO()
                plt.savefig(density_img, format='png')
                density_img.seek(0)
                stat['density_plot'] = base64.b64encode(density_img.getvalue()).decode()
                plt.close()
            if show_violinplot:
                plt.figure(figsize=(6, 4))
                sns.violinplot(data=data_series.dropna(), color='lightcoral')
                plt.title(f'Violin plot {title_suffix}')
                plt.xlabel('Value')
                plt.tight_layout()
                violin_img = io.BytesIO()
                plt.savefig(violin_img, format='png')
                violin_img.seek(0)
                stat['violin_plot'] = base64.b64encode(violin_img.getvalue()).decode()
                plt.close()

            if show_raincloudplot:
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.violinplot(data=data_series.dropna(), ax=ax, inner=None, color='lightblue', alpha=0.5)
                sns.stripplot(data=data_series.dropna(), ax=ax, color='darkblue', size=4, jitter=True, alpha=0.6)
                ax.set_title(f'Raincloud plot {title_suffix}')
                ax.set_xlabel('Value')
                plt.tight_layout()
                raincloud_img = io.BytesIO()
                plt.savefig(raincloud_img, format='png')
                raincloud_img.seek(0)
                stat['raincloud_plot'] = base64.b64encode(raincloud_img.getvalue()).decode()
                plt.close()
            return stat

        # Si solo hay una muestra, devolver las estadísticas para una muestra
        if data_series2 is None:
            results.update(analyze_single_series(data_series1, ""))
            return results

        if data_series2 is not None:
            # Validar que ambas muestras tienen la misma longitud
            if len(data_series1) != len(data_series2):
                raise ValueError("Samples must have the same length for correlation.")

            # Calcular las medias
            mean1 = float(data_series1.mean()) if not data_series1.empty else None
            mean2 = float(data_series2.mean()) if not data_series2.empty else None

            # Calcular la correlación
            try:
                if data_series1.empty or data_series2.empty:
                    correlation = None
                else:
                    correlation, _ = stats.pearsonr(data_series1, data_series2)
            except Exception as e:
                correlation = None
                print(f"Error al calcular la correlación: {str(e)}")

            # Almacenar resultados
            results = {
                'mean1': mean1,
                'mean2': mean2,
                'correlation': correlation,
            }

            # Generar scatter plot si está solicitado
            if show_scatter:
                try:
                    plt.figure(figsize=(6, 4))
                    plt.scatter(data_series1, data_series2, color='purple', alpha=0.6)
                    plt.title('Scatter plot with trend line')
                    plt.xlabel('Sample 1')
                    plt.ylabel('Sample 2')
                    plt.tight_layout()
                    # Agregar línea de tendencia
                    m, b = np.polyfit(data_series1, data_series2, 1)
                    plt.plot(data_series1, m * data_series1 + b, color='red')
                    scatter_img = io.BytesIO()
                    plt.savefig(scatter_img, format='png')
                    scatter_img.seek(0)
                    results['scatter_plot'] = base64.b64encode(scatter_img.getvalue()).decode()
                    plt.close()
                except Exception as e:
                    print(f"Error al generar scatter plot: {str(e)}")

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
            raise ValueError("The data for the first sample is either invalid or not in the correct format.")

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
                raise ValueError("The categories are either invalid or not in the correct format.")
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
                raise ValueError("The data for the second sample is either invalid or not in the correct format.")
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

def leer_csv_automatico(content):
    """
    Lee un archivo CSV desde un contenido de texto con delimitadores ',' o ';'.
    
    Args:
        content (str): Contenido del archivo en formato de texto.
        
    Returns:
        pd.DataFrame: DataFrame con los datos del archivo CSV.
    """
    from io import StringIO
    delimitadores = [",", ";"]
    for delimitador in delimitadores:
        try:
            df = pd.read_csv(StringIO(content), delimiter=delimitador)
            # Validar si el DataFrame tiene más de una columna
            if df.shape[1] > 1:
                return df
        except pd.errors.ParserError:
            continue
    raise ValueError("The file delimiter could not be determined.")

# Método para manejar celdas vacías usando imputación básica
def imputar_valores_basico(dataframe):
    """
    Imputa valores faltantes en el DataFrame utilizando métodos básicos.

    - Columnas numéricas: Media.
    - Columnas categóricas: Moda.

    Args:
        dataframe (pd.DataFrame): DataFrame con valores faltantes.

    Returns:
        pd.DataFrame: DataFrame con valores imputados.
    """
    # Separar columnas numéricas y categóricas
    numeric_columns = dataframe.select_dtypes(include=['number']).columns.tolist()
    categorical_columns = dataframe.select_dtypes(exclude=['number']).columns.tolist()

    # Imputación para columnas numéricas: Rellenar con la media
    for column in numeric_columns:
        dataframe[column].fillna(dataframe[column].mean(), inplace=True)

    # Imputación para columnas categóricas: Rellenar con la moda
    for column in categorical_columns:
        if not dataframe[column].mode().empty:  # Verificar si hay datos para calcular la moda
            dataframe[column].fillna(dataframe[column].mode()[0], inplace=True)

    return dataframe



@app.route('/upload_csv_descriptive', methods=['POST'])
def upload_file_descriptive():
    global dataframe
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'File not found.'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'The file has no name.'}), 400

        filename = file.filename.lower()

        # Determinar el tipo de archivo y leerlo en un DataFrame de Pandas
        try:
            if filename.endswith('.csv'):
                try:
                    # Leer el contenido del archivo
                    content = file.stream.read().decode("utf-8")
                    dataframe = leer_csv_automatico(content)  # Usar la nueva función
                except UnicodeDecodeError:
                    # Intentar con ISO-8859-1 si UTF-8 falla
                    file.stream.seek(0)  # Reiniciar el stream
                    content = file.stream.read().decode("ISO-8859-1")
                    dataframe = leer_csv_automatico(content)  # Usar la nueva función


            elif filename.endswith(('.xls', '.xlsx')):
                try:
                    # Leer archivo Excel (.xls o .xlsx)
                    file_stream = io.BytesIO(file.read())
                    dataframe = pd.read_excel(file_stream, engine='openpyxl')
                except ValueError as e:
                    return jsonify({'error': f'Error reading the Excel file: {str(e)}'}), 400
                except Exception as e:
                    return jsonify({'error': f'Unknown error reading the Excel file: {str(e)}'}), 400


            elif filename.endswith('.txt'):
                try:
                    content = file.stream.read().decode("utf-8")
                    dataframe = pd.read_csv(io.StringIO(content), delimiter=r'\s+')
                except UnicodeDecodeError:
                    content = file.stream.read().decode("ISO-8859-1")
                    dataframe = pd.read_csv(io.StringIO(content), delimiter=r'\s+')
            else:
                return jsonify({'error': "Unsupported file format. Please provide a CSV, XLSX, XLS, or TXT file."}), 400

            # Check if the DataFrame was loaded correctly
            if dataframe.empty:
                return jsonify({'error': 'The file is empty or could not be processed correctly.'}), 400

            # Normalizar encabezados quitando espacios adicionales
            dataframe.columns = dataframe.columns.str.strip()

            # Manejo de celdas vacías con KNN imputación
            dataframe = imputar_valores_basico(dataframe)

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
                'message': 'File uploaded successfully',
                'numeric_columns': numeric_columns,
                'categorical_columns': categorical_columns,
                'binary_categorical_columns': binary_categorical_columns
            })

        except UnicodeDecodeError:
            return jsonify({'error': 'Encoding error. Ensure the file is in UTF-8 or ISO-8859-1 format.'}), 400
        except pd.errors.ParserError:
            return jsonify({'error': 'Error parsing the file. Check the delimiter and file format.'}), 400

        except Exception as e:
            return jsonify({'error': str(e)}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 400
    


# Ruta para el análisis descriptivo de columnas seleccionadas
@app.route('/analyze_selected_columns', methods=['POST'])
def analyze_selected_columns():
    global dataframe
    try:
        if dataframe is None:
            return jsonify({'error': 'No file has been uploaded for analysis.'}), 400

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
            return jsonify({'error': "No numerical columns were provided for analysis."}), 400

        result = {}

        # Análisis de una muestra (histograma y boxplot)
        if analysis_type == "Una muestra":
            if len(selected_columns) != 1:
                return jsonify({'error': "Select exactly one numerical column for this analysis."}), 400
            
            data_series = dataframe[selected_columns[0]].dropna()

            # Calcular estadísticas descriptivas
            result[selected_columns[0]] = calculate_descriptive_statistics_from_data(data_series)

            # Crear histograma si solicitado
            if show_histogram:
                plt.figure(figsize=(8, 6))
                sns.histplot(data_series, bins=20, kde=True, color='skyblue')
                plt.xlabel(selected_columns[0])
                plt.ylabel('Frequency')
                plt.title(f'Histogram of {selected_columns[0]}')
                histogram_img = io.BytesIO()
                plt.savefig(histogram_img, format='png', bbox_inches='tight', dpi=100)
                histogram_img.seek(0)
                result['histogram'] = base64.b64encode(histogram_img.getvalue()).decode()
                plt.close()

            # Crear boxplot si solicitado
            if show_boxplot:
                plt.figure(figsize=(8, 6))
                sns.boxplot(x=data_series, palette="Set2", width=0.4)
                plt.title(f'Boxplot of {selected_columns[0]}')
                boxplot_img = io.BytesIO()
                plt.savefig(boxplot_img, format='png', bbox_inches='tight', dpi=100)
                boxplot_img.seek(0)
                result['boxplot'] = base64.b64encode(boxplot_img.getvalue()).decode()
                plt.close()

        # Análisis de dos muestras (scatter plot y correlación)
        elif analysis_type == "Dos muestras":
            if len(selected_columns) != 2:
                return jsonify({'error': "Select exactly two numeric columns for this analysis."}), 400

            data_series1 = dataframe[selected_columns[0]].dropna()
            data_series2 = dataframe[selected_columns[1]].dropna()

            if len(data_series1) != len(data_series2):
                return jsonify({'error': "The selected columns have different amounts of valid data."}), 400

            result['mean1'] = float(data_series1.mean())
            result['mean2'] = float(data_series2.mean())
            result['correlation'], _ = stats.pearsonr(data_series1, data_series2)

            # Crear scatter plot si solicitado
            if show_scatter:
                plt.figure(figsize=(8, 6))
                sns.scatterplot(x=data_series1, y=data_series2, color='blue', alpha=0.6, s=80, edgecolor='w')
                plt.title('Scatter plot with trend line')
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
                return jsonify({'error': "Select a numeric column and a categorical column for this analysis."}), 400
            
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
                plt.title(f'Boxplot of {selected_columns[0]} by {category_column}')
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
                plt.title(f'Violin plot of {selected_columns[0]} by {category_column}')
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
                ax.set_xlabel('Values')
                ax.set_title(f'Raincloud plot of {selected_columns[0]} by {category_column}')
                raincloud_img = io.BytesIO()
                plt.savefig(raincloud_img, format='png', bbox_inches='tight', dpi=100)
                raincloud_img.seek(0)
                result['raincloud_plot'] = base64.b64encode(raincloud_img.getvalue()).decode()
                plt.close()

            result['stats_by_category'] = stats_by_category

        else:
            return jsonify({'error': "Invalid type of analysis."}), 400

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Función para calcular estadísticas descriptivas
def calculate_descriptive_statistics_from_data(data_series):
    try:
        # Ignorar NaN en lugar de reemplazarlos con 0
        count = float(data_series.dropna().size)
        mean = float(data_series.mean(skipna=True)) if not np.isnan(data_series.mean(skipna=True)) else None
        median = float(data_series.median(skipna=True)) if not np.isnan(data_series.median(skipna=True)) else None
        mode = replace_nan_with_none(data_series.mode(dropna=True).tolist())  # Usar dropna=True para la moda
        std = float(data_series.std(skipna=True)) if not np.isnan(data_series.std(skipna=True)) else None
        variance = float(data_series.var(skipna=True)) if not np.isnan(data_series.var(skipna=True)) else None
        min_value = float(data_series.min(skipna=True)) if not np.isnan(data_series.min(skipna=True)) else None
        max_value = float(data_series.max(skipna=True)) if not np.isnan(data_series.max(skipna=True)) else None
        range_value = (max_value - min_value) if min_value is not None and max_value is not None else None
        coef_var = (std / mean * 100) if mean and mean != 0 else None

        # Sanitizar skewness y kurtosis
        skewness = None if np.isnan(stats.skew(data_series.dropna())) else float(stats.skew(data_series.dropna()))
        kurtosis_value = None if np.isnan(stats.kurtosis(data_series.dropna())) else float(stats.kurtosis(data_series.dropna()))

        q1 = float(data_series.quantile(0.25, interpolation='linear')) if not np.isnan(data_series.quantile(0.25, interpolation='linear')) else None
        q3 = float(data_series.quantile(0.75, interpolation='linear')) if not np.isnan(data_series.quantile(0.75, interpolation='linear')) else None
        p10 = float(data_series.quantile(0.10, interpolation='linear')) if not np.isnan(data_series.quantile(0.10, interpolation='linear')) else None
        p90 = float(data_series.quantile(0.90, interpolation='linear')) if not np.isnan(data_series.quantile(0.90, interpolation='linear')) else None

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

def leer_csv_automatico(content):
    """
    Lee un archivo CSV desde un contenido de texto con delimitadores ',' o ';'.
    
    Args:
        content (str): Contenido del archivo en formato de texto.
        
    Returns:
        pd.DataFrame: DataFrame con los datos del archivo CSV.
    """
    from io import StringIO
    delimitadores = [",", ";"]
    for delimitador in delimitadores:
        try:
            df = pd.read_csv(StringIO(content), delimiter=delimitador)
            # Validar si el DataFrame tiene más de una columna
            if df.shape[1] > 1:
                return df
        except pd.errors.ParserError:
            continue
    raise ValueError("Could not determine the file delimiter.")

# Método para manejar celdas vacías usando imputación básica
def imputar_valores_basico(dataframe):
    """
    Imputa valores faltantes en el DataFrame utilizando métodos básicos.

    - Columnas numéricas: Media.
    - Columnas categóricas: Moda.

    Args:
        dataframe (pd.DataFrame): DataFrame con valores faltantes.

    Returns:
        pd.DataFrame: DataFrame con valores imputados.
    """
    # Separar columnas numéricas y categóricas
    numeric_columns = dataframe.select_dtypes(include=['number']).columns.tolist()
    categorical_columns = dataframe.select_dtypes(exclude=['number']).columns.tolist()

    # Imputación para columnas numéricas: Rellenar con la media
    for column in numeric_columns:
        dataframe[column].fillna(dataframe[column].mean(), inplace=True)

    # Imputación para columnas categóricas: Rellenar con la moda
    for column in categorical_columns:
        if not dataframe[column].mode().empty:  # Verificar si hay datos para calcular la moda
            dataframe[column].fillna(dataframe[column].mode()[0], inplace=True)

    return dataframe



@app.route('/upload_csv_charts', methods=['POST'])
def upload_file_charts():
    global dataframe
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'File not found.'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'The file has no name.'}), 400

        filename = file.filename.lower()

        # Determinar el tipo de archivo y leerlo en un DataFrame de Pandas
        try:
            if filename.endswith('.csv'):
                try:
                    # Leer el contenido del archivo
                    content = file.stream.read().decode("utf-8")
                    dataframe = leer_csv_automatico(content)  # Usar la nueva función
                except UnicodeDecodeError:
                    # Intentar con ISO-8859-1 si UTF-8 falla
                    file.stream.seek(0)  # Reiniciar el stream
                    content = file.stream.read().decode("ISO-8859-1")
                    dataframe = leer_csv_automatico(content)  # Usar la nueva función


            elif filename.endswith(('.xls', '.xlsx')):
                try:
                    # Leer archivo Excel (.xls o .xlsx)
                    file_stream = io.BytesIO(file.read())
                    dataframe = pd.read_excel(file_stream, engine='openpyxl')
                except ValueError as e:
                    return jsonify({'error': f'Error reading the Excel file: {str(e)}'}), 400
                except Exception as e:
                    return jsonify({'error': f'Unknown error reading the Excel file: {str(e)}'}), 400

            elif filename.endswith('.txt'):
                try:
                    content = file.stream.read().decode("utf-8")
                    dataframe = pd.read_csv(io.StringIO(content), delimiter=r'\s+')
                except UnicodeDecodeError:
                    content = file.stream.read().decode("ISO-8859-1")
                    dataframe = pd.read_csv(io.StringIO(content), delimiter=r'\s+')
            else:
                return jsonify({'error': "Unsupported file format. Please provide a CSV, XLSX, XLS, or TXT file."}), 400

            # Verificar si el DataFrame se cargó correctamente
            if dataframe.empty:
                return jsonify({'error': 'The file is empty or could not be processed correctly.'}), 400

            # Normalizar encabezados quitando espacios adicionales
            dataframe.columns = dataframe.columns.str.strip()

            # Manejo de celdas vacías con KNN imputación
            dataframe = imputar_valores_basico(dataframe)

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
                'message': 'File uploaded successfully',
                'numeric_columns': numeric_columns,
                'categorical_columns': categorical_columns,
                'binary_categorical_columns': binary_categorical_columns
            })

        except UnicodeDecodeError:
            return jsonify({'error': 'Encoding error. Ensure the file is in UTF-8 or ISO-8859-1 format.'}), 400
        except pd.errors.ParserError:
            return jsonify({'error': 'Error parsing the file. Check the delimiter and file format.'}), 400
        except Exception as e:
            return jsonify({'error': str(e)}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 400
    

# Nueva ruta para la generación de gráficos a partir de las columnas seleccionadas
@app.route('/generate_charts', methods=['POST'])
def generate_charts():
    global dataframe
    try:
        if dataframe is None:
            return jsonify({'error': 'No file has been uploaded for analysis.'}), 400

        data = request.get_json()
        x_column = data.get('x_column')
        y_column = data.get('y_column')
        chart_type = data.get('chart_type')
        categorical_column = data.get('categorical_column')
        language = data.get('language', 'en')  # Idioma por defecto: inglés

        if not x_column or not chart_type:
            return jsonify({'error': 'Please select the columns and the type of chart.'}), 400
        
        # Diccionario de traducciones
        translations = {
            'en': {
                'Scatterplot': f'Scatterplot of {x_column} vs {y_column}',
                'Histogram': f'Histogram of {x_column}',
                'Boxplot': f'Boxplot of {x_column}',
                'Boxplot_by': f'Boxplot of {x_column} by {categorical_column}',
                'Raincloud plot': f'Raincloud plot of {x_column}',
                'Raincloud plot by': f'Raincloud plot of {x_column} by {categorical_column}',
                'Frequency': 'Frequency',
                'Trend line': 'Trend line',
            },
            'es': {
                'Scatterplot': f'Diagrama de dispersión de {x_column} vs {y_column}',
                'Histogram': f'Histograma de {x_column}',
                'Boxplot': f'Diagrama de caja de {x_column}',
                'Boxplot_by': f'Diagrama de caja de {x_column} por {categorical_column}',
                'Raincloud plot': f'Gráfico de nubes de lluvia de {x_column}',
                'Raincloud plot by': f'Gráfico de nubes de lluvia de {x_column} por {categorical_column}',
                'Frequency': 'Frecuencia',
                'Trend line': 'Línea de tendencia',
            },
            'zh': {
                'Scatterplot': f'{x_column} 与 {y_column} 的散点图',
                'Histogram': f'{x_column} 的直方图',
                'Boxplot': f'{x_column} 的箱线图',
                'Boxplot_by': f'{categorical_column} 分类的 {x_column} 的箱线图',
                'Raincloud plot': f'{x_column} 的雨云图',
                'Raincloud plot by': f'{categorical_column} 分类的 {x_column} 的雨云图',
                'Frequency': '频率',
                'Trend line': '趋势线',
            }
        }

        # Limpiar los datos eliminando valores nulos
        df_clean = dataframe.dropna(subset=[x_column, y_column] if y_column else [x_column])

        # Generar el gráfico
        img = io.BytesIO()
        plt.figure(figsize=(10, 8))

        if chart_type == 'Scatterplot':
            if not y_column:
                return jsonify({'error': 'For a scatterplot, you must select two numeric variables.'}), 400
            
            add_trendline = data.get('add_trendline', False)

            if pd.api.types.is_numeric_dtype(dataframe[x_column]) and pd.api.types.is_numeric_dtype(dataframe[y_column]):
                sns.scatterplot(x=df_clean[x_column], y=df_clean[y_column], color='blue', alpha=0.6, edgecolor='w', s=80)
                plt.xlabel(x_column)
                plt.ylabel(y_column)
                plt.title(translations[language]['Scatterplot'])
                plt.grid(True)

                # Añadir línea de tendencia si se solicita
                if add_trendline:
                    m, b = np.polyfit(df_clean[x_column], df_clean[y_column], 1)
                    plt.plot(df_clean[x_column], m * df_clean[x_column] + b, color='red', linestyle='--', linewidth=2)
                    plt.legend([translations[language]['Trend line'], 'Data'])
                    
            else:
                return jsonify({'error': 'Both selected columns must be numeric for a scatterplot.'}), 400

        elif chart_type == 'Histograma':
            if pd.api.types.is_numeric_dtype(dataframe[x_column]):
                sns.histplot(df_clean[x_column], bins=20, kde=True, color='skyblue')
                plt.xlabel(x_column)
                plt.ylabel(translations[language]['Frequency'])
                plt.title(translations[language]['Histogram'])
            else:
                return jsonify({'error': 'The selected column must be numeric for a histogram.'}), 400

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
                        plt.title(translations[language]['Boxplot_by'])
                        plt.xlabel(categorical_column)  # Etiqueta correcta para el eje X
                        plt.ylabel(x_column)  # Etiqueta correcta para el eje Y
                    else:
                        return jsonify({'error': 'The selected categorical column is not valid.'}), 400
                else:
                    # Boxplot sin categorización
                    sns.boxplot(x=df_clean[x_column], color='lightgreen', width=0.4)
                    plt.title(translations[language]['Boxplot'])
                    plt.xlabel(x_column)
                    plt.ylabel(translations[language]['Frequency'])  # Etiqueta adecuada para el eje Y cuando no hay categorización
            else:
                return jsonify({'error': 'The selected column must be numeric for a boxplot.'}), 400


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
                    ax.set_title(translations[language]['Raincloud plot by'])
                    plt.grid(True)
                else:
                    # Raincloud Plot sin categorización
                    sns.violinplot(x=df_clean[x_column], inner=None, color='skyblue', alpha=0.5)
                    sns.boxplot(x=df_clean[x_column], color='lightgreen', width=0.2)
                    y_jitter = np.random.uniform(-0.05, 0.05, size=len(df_clean[x_column]))
                    plt.scatter(df_clean[x_column], y_jitter, s=10, color='blue', alpha=0.6)
                    plt.xlabel(x_column)
                    plt.title(translations[language]['Raincloud plot'])

            else:
                return jsonify({'error': 'The selected column must be numeric for a raincloud plot.'}), 400

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

def leer_csv_automatico(content):
    """
    Lee un archivo CSV desde un contenido de texto con delimitadores ',' o ';'.
    
    Args:
        content (str): Contenido del archivo en formato de texto.
        
    Returns:
        pd.DataFrame: DataFrame con los datos del archivo CSV.
    """
    from io import StringIO
    delimitadores = [",", ";"]
    for delimitador in delimitadores:
        try:
            df = pd.read_csv(StringIO(content), delimiter=delimitador)
            # Validar si el DataFrame tiene más de una columna
            if df.shape[1] > 1:
                return df
        except pd.errors.ParserError:
            continue
    raise ValueError("The file delimiter could not be determined.")

# Método para manejar celdas vacías usando imputación básica
def imputar_valores_basico(dataframe):
    """
    Imputa valores faltantes en el DataFrame utilizando métodos básicos.

    - Columnas numéricas: Media.
    - Columnas categóricas: Moda.

    Args:
        dataframe (pd.DataFrame): DataFrame con valores faltantes.

    Returns:
        pd.DataFrame: DataFrame con valores imputados.
    """
    # Separar columnas numéricas y categóricas
    numeric_columns = dataframe.select_dtypes(include=['number']).columns.tolist()
    categorical_columns = dataframe.select_dtypes(exclude=['number']).columns.tolist()

    # Imputación para columnas numéricas: Rellenar con la media
    for column in numeric_columns:
        dataframe[column].fillna(dataframe[column].mean(), inplace=True)

    # Imputación para columnas categóricas: Rellenar con la moda
    for column in categorical_columns:
        if not dataframe[column].mode().empty:  # Verificar si hay datos para calcular la moda
            dataframe[column].fillna(dataframe[column].mode()[0], inplace=True)

    return dataframe



@app.route('/upload_csv_stat', methods=['POST'])
def upload_file_stat():
    global dataframe
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'The file was not found.'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'The file has no name.'}), 400

        filename = file.filename.lower()

        # Determinar el tipo de archivo y leerlo en un DataFrame de Pandas
        try:
            if filename.endswith('.csv'):
                try:
                    # Leer el contenido del archivo
                    content = file.stream.read().decode("utf-8")
                    dataframe = leer_csv_automatico(content)  # Usar la nueva función
                except UnicodeDecodeError:
                    # Intentar con ISO-8859-1 si UTF-8 falla
                    file.stream.seek(0)  # Reiniciar el stream
                    content = file.stream.read().decode("ISO-8859-1")
                    dataframe = leer_csv_automatico(content)  # Usar la nueva función


            elif filename.endswith(('.xls', '.xlsx')):
                try:
                    # Leer archivo Excel (.xls o .xlsx)
                    file_stream = io.BytesIO(file.read())
                    dataframe = pd.read_excel(file_stream, engine='openpyxl')
                except ValueError as e:
                    return jsonify({'error': f'Error reading the Excel file: {str(e)}'}), 400
                except Exception as e:
                    return jsonify({'error': f'Unknown error reading the Excel file: {str(e)}'}), 400

            elif filename.endswith('.txt'):
                try:
                    content = file.stream.read().decode("utf-8")
                    dataframe = pd.read_csv(io.StringIO(content), delimiter=r'\s+')
                except UnicodeDecodeError:
                    content = file.stream.read().decode("ISO-8859-1")
                    dataframe = pd.read_csv(io.StringIO(content), delimiter=r'\s+')
            else:
                return jsonify({'error': "Unsupported file format. Please provide a CSV, XLSX, XLS, or TXT file."}), 400

            # Verificar si el DataFrame se cargó correctamente
            if dataframe.empty:
                return jsonify({'error': 'The file is empty or could not be processed correctly.'}), 400

            # Normalizar encabezados quitando espacios adicionales
            dataframe.columns = dataframe.columns.str.strip()

            # Manejo de celdas vacías con KNN imputación
            dataframe = imputar_valores_basico(dataframe)

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
                'message': 'File uploaded successfully',
                'numeric_columns': numeric_columns,
                'categorical_columns': categorical_columns,
                'binary_categorical_columns': binary_categorical_columns
            })

        except UnicodeDecodeError:
            return jsonify({'error': 'Encoding error. Ensure the file is in UTF-8 or ISO-8859-1 format.'}), 400
        except pd.errors.ParserError:
            return jsonify({'error': 'Error parsing the file. Check the delimiter and file format.'}), 400
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

# Función para generar imágenes base64
def generate_base64_image(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    base64_image = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)  # Cerrar la figura para liberar memoria
    return base64_image

# REGRESIÓN SIMPLE
@app.route('/run_regression', methods=['POST'])
def run_regression():
    global dataframe
    try:
        if dataframe is None:
            return jsonify({'error': 'No file has been uploaded for analysis'}), 400

        # Limpia los nombres de las columnas antes de procesar
        name_mapping = clean_column_names(dataframe)

        data = request.get_json()
        response_variable = data.get('response_variable')
        covariates = data.get('covariates')
        analyze_residuals = data.get('analyze_residuals', False)  # Nuevo parámetro

        if not response_variable or not covariates:
            return jsonify({'error': 'Insufficient variables for regression.'}), 400
        
        # Normaliza los nombres de las variables
        response_variable_clean = re.sub(r'[^a-zA-Z0-9_]', '_', response_variable).lower()
        covariates_clean = [re.sub(r'[^a-zA-Z0-9_]', '_', cov).lower() for cov in covariates]

        # Verificar que la variable de respuesta y las covariables existen en el DataFrame
        if response_variable_clean not in dataframe.columns:
            return jsonify({'error': f'The response variable {response_variable} does not exist in the data.'}), 400

        for cov_clean, cov_original in zip(covariates_clean, covariates):
            if cov_clean not in dataframe.columns:
                return jsonify({'error': f'The covariate {cov_original} does not exist in the data.'}), 400

        # Preparar el DataFrame para el modelo
        df = dataframe[covariates_clean + [response_variable_clean]].copy()
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)

        # Verificar si el DataFrame resultante tiene suficientes datos
        if df.empty or len(df) < 2:
            return jsonify({'error': 'There is not enough valid data after preprocessing.'}), 400

        # Validar que las covariables tengan variación
        for cov in covariates_clean:
            if df[cov].nunique() < 2:
                return jsonify({'error': f'The covariate {cov} does not have sufficient variation.'}), 400

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

        # Calcular análisis adicional de residuos si el usuario lo solicita
        residuals_data = None
        if analyze_residuals:
            residuals = model_full.resid
            residuals = residuals.replace([np.inf, -np.inf], np.nan).dropna()

            # Verificar los valores de residuos
            if residuals.empty:
                return jsonify({'error': 'Valid residuals could not be calculated.'}), 400

            # Histograma de residuos
            fig_hist, ax_hist = plt.subplots(figsize=(8, 6))  # Ajustar el tamaño
            sns.histplot(residuals, bins=10, kde=True, color='blue', ax=ax_hist)
            ax_hist.set_title("Histogram of residuals")
            ax_hist.set_xlabel("Residuals")
            ax_hist.set_ylabel("Density")

            # Q-Q Plot
            fig_qq = plt.figure(figsize=(8, 6))  # Ajustar el tamaño
            sm.qqplot(residuals, line='45', fit=True, ax=fig_qq.add_subplot(111))

            # Histograma de residuos
            hist_image = generate_base64_image(fig_hist)
            plt.close(fig_hist)

            # Q-Q Plot
            qq_image = generate_base64_image(fig_qq)
            plt.close(fig_qq)


            # Determinar el test de normalidad a usar según el tamaño de muestra
            n_residuals = len(residuals)
            if n_residuals < 50:
                # Usar Shapiro-Wilk
                normality_stat, normality_p_value = stats.shapiro(residuals)
                normality_test_name = "Shapiro-Wilk"
            else:
                # Usar Kolmogorov-Smirnov
                normality_stat, normality_p_value = stats.kstest(residuals, 'norm')
                normality_test_name = "Kolmogorov-Smirnov"

            # Prueba de Breusch-Pagan (heterocedasticidad)
            exog = model_full.model.exog
            exog = exog[~np.isnan(residuals)]  # Mantén solo las filas válidas para residuos
            bp_stat, bp_p_value, _, _ = sm.stats.diagnostic.het_breuschpagan(residuals, exog)

            # Prueba de Durbin-Watson (autocorrelación)
            durbin_watson_stat = sm.stats.stattools.durbin_watson(residuals)

            # Preparar los datos para el análisis de residuos
            residuals_data = {
                "histogram": hist_image,
                "qq_plot": qq_image,
                "normality_test": {
                    "test_name": normality_test_name,
                    "statistic": normality_stat,
                    "p_value": normality_p_value
                },
                "breusch_pagan_test": {
                    "statistic": bp_stat,
                    "p_value": bp_p_value
                },
                "durbin_watson": durbin_watson_stat
            }

        return jsonify({
            'regression_result': regression_result,
            'residuals': residuals_data,
            'warnings': []
        })

    except Exception as e:
        if "invalid syntax" in str(e).lower():
            error_message = (
                "Error in variable names. Ensure they do not include special characters "
                "(accents, spaces, symbols)."
            )

        else:
            error_message = str(e)
        return jsonify({'error': error_message}), 400

# REGRESIÓN LOGÍSTICA
@app.route('/run_logistic_regression', methods=['POST'])
def run_logistic_regression():
    global dataframe
    try:
        if dataframe is None:
            return jsonify({'error': 'No file has been uploaded for analysis.'}), 400

        # Limpia los nombres de las columnas antes de procesar
        name_mapping = clean_column_names(dataframe)

        # Obtener los datos de la solicitud
        data = request.get_json()
        response_variable = data.get('response_variable')  # Variable dependiente
        covariates = data.get('covariates')  # Lista de covariables

        if not response_variable or not covariates:
            return jsonify({'error': 'Insufficient variables for logistic regression.'}), 400

        # Normaliza los nombres de las variables
        response_variable_clean = re.sub(r'[^a-zA-Z0-9_]', '_', response_variable).lower()
        covariates_clean = [re.sub(r'[^a-zA-Z0-9_]', '_', cov).lower() for cov in covariates]

        # Verificar que la variable de respuesta y covariables existen en el DataFrame
        if response_variable_clean not in dataframe.columns:
            return jsonify({'error': f'The response variable {response_variable} does not exist in the data.'}), 400

        for cov_clean, cov_original in zip(covariates_clean, covariates):
            if cov_clean not in dataframe.columns:
                return jsonify({'error': f'The covariate {cov_original} does not exist in the data.'}), 400

        # Preparar el DataFrame para el modelo
        df = dataframe[covariates_clean + [response_variable_clean]].copy()
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)

        # Verificar si el DataFrame tiene suficientes datos
        if df.empty or len(df) < 10:
            return jsonify({'error': 'There is not enough valid data after preprocessing.'}), 400

        # Validar y convertir la variable dependiente a binaria (0 y 1)
        if df[response_variable_clean].nunique() == 2:
            # Si es categórica, convertir a 0 y 1
            if not pd.api.types.is_numeric_dtype(df[response_variable_clean]):
                df[response_variable_clean] = pd.Categorical(df[response_variable_clean]).codes
        else:
            return jsonify({'error': 'The response variable is not binary (it must have exactly two levels).'}), 400

        # Validar que las covariables tengan variación
        for cov in covariates_clean:
            if df[cov].nunique() < 2:
                return jsonify({'error': f'The covariate {cov} does not have sufficient variation.'}), 400

        # Definir la fórmula para el modelo logístico
        formula = f"{response_variable_clean} ~ " + " + ".join(covariates_clean)

        # Ajustar el modelo logístico usando statsmodels
        model = smf.logit(formula=formula, data=df).fit(disp=False)

        # Extraer coeficientes y p-valores
        coefficients = model.params.to_dict()
        p_values = model.pvalues.to_dict()
        intercept = coefficients.pop('Intercept', None)

        # Calcular el pseudo R-cuadrado
        pseudo_r_squared = model.prsquared

        # Generar predicciones
        predictions = model.predict(df)
        predicted_classes = (predictions >= 0.5).astype(int)

        # Calcular precisión y matriz de confusión
        actual_classes = df[response_variable_clean]
        accuracy = np.mean(predicted_classes == actual_classes)
        confusion_matrix = pd.crosstab(actual_classes, predicted_classes,
                                       rownames=['Actual'], colnames=['Predicted'])

        # Calcular el odds ratio
        odds_ratios = {key: np.exp(value) for key, value in coefficients.items()}

        # Preparar resultados
        regression_result = {
            "coefficients": coefficients,
            "odds_ratios": odds_ratios,
            "intercept": intercept,
            "pseudo_r_squared": pseudo_r_squared,
            "accuracy": accuracy,
            "p_values": p_values,
            "confusion_matrix": confusion_matrix.to_dict()
        }

        return jsonify({
            'regression_result': regression_result,
            'warnings': []
        })

    except Exception as e:
        error_message = str(e)
        return jsonify({'error': error_message}), 400


    
# 3. T-Test
@app.route('/run_ttest', methods=['POST'])
def run_ttest():
    global dataframe
    try:
        data = request.get_json()
        comparison_type = data.get('comparison_type')
        paired = data.get('paired', False)
        alternative = data.get('alternative', 'two-sided')

        if comparison_type == 'categorical_vs_numeric':
            numeric_column = data.get('numeric_column')
            categorical_column = data.get('categorical_column')

            if numeric_column not in dataframe.columns or categorical_column not in dataframe.columns:
                return jsonify({'error': 'The specified columns do not exist in the uploaded data. Please check the names and try again.'}), 400

            # Agrupar los datos por la columna categórica
            groups = dataframe.groupby(categorical_column)[numeric_column].apply(list)

            # Verificar que haya al menos dos grupos
            if len(groups) < 2:
                return jsonify({'error': 'Insufficient data to perform the T-Test. At least two different categories are required.'}), 400

            category_names = groups.index.tolist()

            # Calcular estadísticas descriptivas
            group_stats = {
                category: {
                    'mean': np.mean(data),
                    'std': np.std(data),
                    'count': len(data)
                } for category, data in groups.items()
            }

            # Verificar longitudes para pruebas pareadas
            if paired and len(groups.iloc[0]) != len(groups.iloc[1]):
                return jsonify({
                    'error': (
                        f'The paired T-Test requires the two groups to have the same number of elements. '
                        f'Group "{category_names[0]}" has {len(groups.iloc[0])} elements, while '
                        f'group "{category_names[1]}" has {len(groups.iloc[1])} elements.'
                    )
                }), 400

            # Realizar la prueba T con el tipo de prueba especificado
            if paired:
                t_stat, p_value = stats.ttest_rel(groups.iloc[0], groups.iloc[1], alternative=alternative)
            else:
                t_stat, p_value = stats.ttest_ind(groups.iloc[0], groups.iloc[1], alternative=alternative, equal_var=False)

            result = {
                'test': 'T-Test' + (' pareado' if paired else ''),
                't_statistic': t_stat,
                'p_value': p_value,
                'significance': "significant" if p_value < 0.05 else "not significant",
                'decision': "Reject the null hypothesis" if p_value < 0.05 else "Do not reject the null hypothesis",
                'category1': category_names[0],
                'category2': category_names[1],
                'alternative': alternative,
                'group_statistics': group_stats  # Agregar estadísticas descriptivas
            }

        elif comparison_type == 'numeric_vs_numeric':
            numeric_column1 = data.get('numeric_column1')
            numeric_column2 = data.get('numeric_column2')

            if numeric_column1 not in dataframe.columns or numeric_column2 not in dataframe.columns:
                return jsonify({'error': 'The specified columns do not exist in the uploaded data. Please check the names and try again.'}), 400

            # Extraer las columnas
            col1_data = dataframe[numeric_column1].dropna()
            col2_data = dataframe[numeric_column2].dropna()

            # Calcular estadísticas descriptivas
            column_stats = {
                numeric_column1: {
                    'mean': np.mean(col1_data),
                    'std': np.std(col1_data),
                    'count': len(col1_data)
                },
                numeric_column2: {
                    'mean': np.mean(col2_data),
                    'std': np.std(col2_data),
                    'count': len(col2_data)
                }
            }

            # Verificar longitudes para pruebas pareadas
            if paired and len(col1_data) != len(col2_data):
                return jsonify({
                    'error': (
                        f'The paired T-Test requires the two variables to have the same number of elements. '
                        f'Column "{numeric_column1}" has {len(col1_data)} elements, while '
                        f'column "{numeric_column2}" has {len(col2_data)} elements.'
                    )
                }), 400

            # Realizar la prueba T con el tipo de prueba especificado
            if paired:
                t_stat, p_value = stats.ttest_rel(col1_data, col2_data, alternative=alternative)
            else:
                t_stat, p_value = stats.ttest_ind(col1_data, col2_data, alternative=alternative, equal_var=False)

            result = {
                'test': 'T-Test' + (' pareado' if paired else ''),
                't_statistic': t_stat,
                'p_value': p_value,
                'significance': "significant" if p_value < 0.05 else "not significant",
                'decision': "Reject the null hypothesis" if p_value < 0.05 else "Do not reject the null hypothesis",
                'column1': numeric_column1,
                'column2': numeric_column2,
                'alternative': alternative,
                'column_statistics': column_stats  # Agregar estadísticas descriptivas
            }

        else:
            return jsonify({'error': 'Invalid comparison type specified. Please check the input data.'}), 400

        # Limpiar los resultados antes de enviarlos
        result = clean_results(result)
        return jsonify({'result': result})

    except Exception as e:
        return jsonify({
            'error': 'An unexpected error occurred during the T-Test calculation. Please check the input data and try again.',
            'details': str(e)
        }), 400




# DEVOLVER A LA APP EL NOMBRE DE LAS CATEGORÍAS
@app.route('/get_category_names', methods=['POST'])
def get_category_names():
    global dataframe
    try:
        data = request.get_json()
        categorical_column = data.get('categorical_column')

        if categorical_column not in dataframe.columns:
            return jsonify({'error': 'The specified categorical column was not found.'}), 400

        # Obtener los nombres de las categorías
        category_names = list(dataframe[categorical_column].dropna().unique())

        if len(category_names) < 2:
            return jsonify({'error': 'Insufficient data to retrieve category names.'}), 400

        return jsonify({'category_names': category_names[:2]})

    except Exception as e:
        return jsonify({'error': str(e)}), 400
    
# WELCH T-Test
@app.route('/run_welch_ttest', methods=['POST'])
def run_welch_ttest():
    global dataframe
    try:
        # Obtener los datos de la solicitud
        data = request.get_json()
        comparison_type = data.get('comparison_type')
        alternative = data.get('alternative', 'two-sided')

        if comparison_type == 'categorical_vs_numeric':
            # Variables para comparación categórica vs. numérica
            numeric_column = data.get('numeric_column')
            categorical_column = data.get('categorical_column')

            # Validar las columnas especificadas
            if numeric_column not in dataframe.columns or categorical_column not in dataframe.columns:
                return jsonify({'error': 'The specified columns were not found in the data.'}), 400

            # Agrupar los datos por la columna categórica
            groups = dataframe.groupby(categorical_column)[numeric_column].apply(list)

            # Verificar que haya al menos dos grupos
            if len(groups) < 2:
                return jsonify({'error': 'Insufficient data to perform the Welch T-Test. At least two categories are required.'}), 400

            category_names = groups.index.tolist()

            # Calcular estadísticas descriptivas para cada grupo
            group_stats = {
                category: {
                    'mean': np.mean(data),
                    'std': np.std(data, ddof=1),  # ddof=1 para muestras
                    'count': len(data)
                } for category, data in groups.items()
            }

            # Realizar la prueba T de Welch
            t_stat, p_value = stats.ttest_ind(
                groups.iloc[0], 
                groups.iloc[1], 
                equal_var=False,  # Welch T-Test: no se asume igualdad de varianzas
                alternative=alternative
            )

            result = {
                'test': 'Welch T-Test',
                't_statistic': t_stat,
                'p_value': p_value,
                'significance': "significant" if p_value < 0.05 else "not significant",
                'decision': "Reject the null hypothesis" if p_value < 0.05 else "Do not reject the null hypothesis",
                'category1': category_names[0],
                'category2': category_names[1],
                'alternative': alternative,
                'group_statistics': group_stats  # Agregar estadísticas descriptivas
            }

        elif comparison_type == 'numeric_vs_numeric':
            # Variables para comparación numérica vs. numérica
            numeric_column1 = data.get('numeric_column1')
            numeric_column2 = data.get('numeric_column2')

            # Validar las columnas especificadas
            if numeric_column1 not in dataframe.columns or numeric_column2 not in dataframe.columns:
                return jsonify({'error': 'The specified columns were not found in the data.'}), 400

            # Extraer los datos de las columnas
            col1_data = dataframe[numeric_column1].dropna()
            col2_data = dataframe[numeric_column2].dropna()

            # Calcular estadísticas descriptivas para cada columna
            column_stats = {
                numeric_column1: {
                    'mean': np.mean(col1_data),
                    'std': np.std(col1_data, ddof=1),  # ddof=1 para muestras
                    'count': len(col1_data)
                },
                numeric_column2: {
                    'mean': np.mean(col2_data),
                    'std': np.std(col2_data, ddof=1),  # ddof=1 para muestras
                    'count': len(col2_data)
                }
            }

            # Realizar la prueba T de Welch
            t_stat, p_value = stats.ttest_ind(
                col1_data,
                col2_data,
                equal_var=False,  # Welch T-Test: no se asume igualdad de varianzas
                alternative=alternative
            )

            result = {
                'test': 'Welch T-Test',
                't_statistic': t_stat,
                'p_value': p_value,
                'significance': "significant" if p_value < 0.05 else "not significant",
                'decision': "Reject the null hypothesis" if p_value < 0.05 else "Do not reject the null hypothesis",
                'column1': numeric_column1,
                'column2': numeric_column2,
                'alternative': alternative,
                'column_statistics': column_stats  # Agregar estadísticas descriptivas
            }

        else:
            return jsonify({'error': 'Invalid comparison type specified. Please check the input data.'}), 400

        return jsonify({'result': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 400


# 1. Prueba de Levene
@app.route('/run_levene', methods=['POST'])
def run_levene():
    global dataframe
    try:
        # Recibir datos del cliente
        data = request.get_json()
        numeric_column = data.get('numeric_column')
        categorical_column = data.get('categorical_column')

        # Validar columnas
        if numeric_column not in dataframe.columns or categorical_column not in dataframe.columns:
            return jsonify({'error': 'The specified columns were not found in the data.'}), 400

        # Group the data by the categorical column
        groups = dataframe.groupby(categorical_column)[numeric_column].apply(list)

        # Ensure there are at least two groups
        if len(groups) < 2:
            return jsonify({'error': 'At least two categories are required to perform Levene’s test.'}), 400

        # Convert the groups to a list of lists
        group_values = [group for group in groups]

        # Perform Levene’s test
        stat, p_value = stats.levene(*group_values)

        # Evaluate significance based on the p-value
        significance = "significant" if p_value < 0.05 else "not significant"
        decision = "Reject the null hypothesis: the variances are not equal" if p_value < 0.05 else "Do not reject the null hypothesis: the variances are equal"


        # Preparar los resultados
        result = {
            'test': 'Prueba de Levene',
            'statistic': stat,
            'p_value': p_value,
            'significance': significance,
            'decision': decision,
            'groups': groups.index.tolist()
        }

        return jsonify({'result': result})

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
            return jsonify({'error': 'The specified columns were not found in the data.'}), 400

        # Group the data by the categorical column
        groups = dataframe.groupby(categorical_column)[numeric_column].apply(list)

        # Check if there is sufficient data for each group
        if any(len(values) < 2 for values in groups):
            return jsonify({'error': 'There are groups with insufficient data to perform ANOVA.'}), 400

        # Realizar ANOVA
        f_statistic, p_value = f_oneway(*groups)

        # Crear respuesta inicial
        result = {
            'f_statistic': f_statistic,
            'p_value': p_value,
            'num_groups': len(groups),
            'total_observations': sum(len(group) for group in groups),
        }

        # If multiple comparisons are enabled, perform Tukey HSD
        if multiple_comparisons:
            if len(groups) < 3:
                return jsonify({'error': 'At least three groups are required to perform multiple comparisons (Tukey HSD).'}), 400

            try:
                # Prepare the data for Tukey HSD
                all_data = []
                labels = []
                for i, group in enumerate(groups):
                    # Check that the group has at least one value
                    if len(group) == 0:
                        return jsonify({'error': f'Group {groups.index[i]} does not contain enough data.'}), 400

                    all_data.extend(group)
                    labels.extend([groups.index[i]] * len(group))

                # Crear DataFrame para Tukey HSD
                df = pd.DataFrame({'value': all_data, 'group': labels})

                # Realizar Tukey HSD
                tukey = mc.pairwise_tukeyhsd(df['value'], df['group'], alpha=0.05)

                # Formatear los resultados de Tukey HSD
                tukey_summary = []
                for res in tukey.summary().data[1:]:  # Ignorar la cabecera
                    # Determinar el nivel de significancia para los asteriscos
                    if res[3] < 0.001:
                        significance = "***"
                    elif res[3] < 0.01:
                        significance = "**"
                    elif res[3] < 0.05:
                        significance = "*"
                    else:
                        significance = ""

                    # Formatear cada resultado y redondear
                    comparison = f"{res[0]} vs {res[1]}"
                    mean_diff = f"{res[2]:.4f}"
                    p_adj = f"{res[3]:.4f} {significance}"
                    ci_lower = f"{res[4]:.4f}"
                    ci_upper = f"{res[5]:.4f}"
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
                return jsonify({'error': f'Error executing Tukey HSD: {str(e)}'}), 500

        # Limpiar los resultados antes de enviarlos
        result = clean_results(result)

        return jsonify({'result': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 400
    

# FRIEDMAN
@app.route('/run_friedman', methods=['POST'])
def run_friedman():
    global dataframe
    try:
        # Obtener los datos de la solicitud
        data = request.get_json()
        numeric_column = data.get('numeric_column')  # Columna numérica
        group_column = data.get('group_column')      # Variable categórica
        subject_column = data.get('subject_column')  # Variable de identificación de sujetos
        include_posthoc = data.get('multipleComparisons', False)

        # Validar que las columnas existan
        if not all(col in dataframe.columns for col in [numeric_column, group_column, subject_column]):
            return jsonify({'error': 'The specified columns were not found in the data.'}), 400

        # Filter valid data
        filtered_df = dataframe[[subject_column, group_column, numeric_column]].dropna(subset=[subject_column, group_column, numeric_column])

        if filtered_df.empty:
            return jsonify({'error': 'There is not enough data after removing rows with null values.'}), 400

        # Reshape data for Friedman test
        grouped_data = filtered_df.pivot(index=subject_column, columns=group_column, values=numeric_column)

        initial_subjects = dataframe[subject_column].nunique()
        remaining_subjects = grouped_data.shape[0]
        omitted_subjects = initial_subjects - remaining_subjects

        if remaining_subjects < 2:
            return jsonify({'error': 'There are not enough subjects with complete data to perform the test.'}), 400

        # Perform Friedman test
        try:
            friedman_stat, p_value = stats.friedmanchisquare(*[grouped_data[group].dropna() for group in grouped_data.columns])
        except Exception as e:
            return jsonify({'error': f'Error in Friedman test: {str(e)}'}), 500


        result = {
            'test': 'Friedman Test',
            'friedman_statistic': round(friedman_stat, 4),
            'p_value': round(p_value, 4),
            'num_groups': grouped_data.shape[1],
            'num_subjects': remaining_subjects,
            'significance': "significant" if p_value < 0.05 else "not significant",
            'decision': "Reject the null hypothesis" if p_value < 0.05 else "Do not reject the null hypothesis"
        }

        if omitted_subjects > 0:
            result['warnings'] = [f"{omitted_subjects} subjects were omitted due to incomplete data."]

        # Multiple comparisons (Post-hoc Nemenyi)
        if include_posthoc:
            try:
                # Ensure complete data for multiple comparisons
                posthoc_data = grouped_data.dropna()

                if posthoc_data.shape[1] < 3:
                    return jsonify({'error': 'At least three groups are required to perform multiple comparisons.'}), 400

                import scikit_posthocs as sp
                posthoc_results = sp.posthoc_nemenyi_friedman(posthoc_data.to_numpy())

                # Convertir los resultados a un formato legible
                posthoc_summary = []
                group_labels = posthoc_data.columns.tolist()

                for i, group1 in enumerate(group_labels):
                    for j, group2 in enumerate(group_labels):
                        if i < j:  # Solo pares únicos
                            p_value_adj = posthoc_results.iloc[i, j]  # Usar iloc para acceder a DataFrame
                            posthoc_summary.append({
                                'comparison': f"{group1} vs {group2}",
                                'p_value_adjusted': round(p_value_adj, 4),
                                'reject_h0': "Yes" if p_value_adj < 0.05 else "No"
                            })

                result['posthoc_comparisons'] = posthoc_summary

            except Exception as e:
                return jsonify({'error': f'Error in multiple comparisons: {str(e)}'}), 500

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
            return jsonify({'error': 'The specified columns were not found in the data.'}), 400

        # Create the contingency table
        contingency_table = pd.crosstab(dataframe[categorical_column1], dataframe[categorical_column2])

        # Perform the Chi-Square test
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)

        # Evaluate significance
        significance = "significant" if p_value < 0.05 else "not significant"
        decision = "Reject the null hypothesis" if p_value < 0.05 else "Do not reject the null hypothesis"


        # Preparar el resultado
        result = {
            'test': 'Chi-Square test',
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
    
    
# 5. Fisher's Exact Test
@app.route('/run_fisher', methods=['POST'])
def run_fisher():
    global dataframe
    try:
        data = request.get_json()
        categorical_column1 = data.get('categorical_column1')
        categorical_column2 = data.get('categorical_column2')
        show_contingency_table = data.get('show_contingency_table', False)

        # Validate that the columns exist in the DataFrame
        if categorical_column1 not in dataframe.columns or categorical_column2 not in dataframe.columns:
            return jsonify({'error': 'The specified columns were not found in the data.'}), 400

        # Create the contingency table
        contingency_table = pd.crosstab(dataframe[categorical_column1], dataframe[categorical_column2])

        # Check the size of the table
        is_not_2x2 = contingency_table.shape != (2, 2)
        warning_message = None

        # If the table is not 2x2, set a warning
        if is_not_2x2:
            warning_message = "The contingency table is not 2x2. The calculation will proceed, but the results may not be valid under Fisher's Test assumptions."

        # Attempt to perform Fisher's Test
        try:
            _, p_value = fisher_exact(contingency_table)
        except ValueError as e:
            return jsonify({
                'error': f'Error calculating Fisher\'s Test: {str(e)}'
            }), 400

        # Evaluate significance
        significance = "significant" if p_value < 0.05 else "not significant"
        decision = "Reject the null hypothesis" if p_value < 0.05 else "Do not reject the null hypothesis"

        # Preparar el resultado
        result = {
            'test': "Fisher's Exact Test",
            'p_value': p_value,
            'significance': significance,
            'decision': decision
        }

        # Incluir la tabla de contingencia si se solicitó
        if show_contingency_table:
            result['contingency_table'] = contingency_table.to_dict()

        # Agregar advertencia si la tabla no es 2x2
        return jsonify({
            'result': result,
            'show_warning': is_not_2x2,
            'warning_message': warning_message
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# MC NEMAR
@app.route('/run_mcnemar', methods=['POST'])
def run_mcnemar():
    global dataframe
    try:
        # Obtener los datos de la solicitud
        data = request.get_json()
        categorical_column1 = data.get('categorical_column1')
        categorical_column2 = data.get('categorical_column2')
        show_contingency_table = data.get('show_contingency_table', False)

        # Validate that the columns exist
        if categorical_column1 not in dataframe.columns or categorical_column2 not in dataframe.columns:
            return jsonify({'error': 'The specified columns were not found in the data.'}), 400

        # Create the contingency table
        contingency_table = pd.crosstab(dataframe[categorical_column1], dataframe[categorical_column2])

        # Check if the table is 2x2
        if contingency_table.shape != (2, 2):
            return jsonify({
                'error': 'The contingency table is not 2x2. McNemar’s Test requires a table of this type.'
            }), 400

        # Perform McNemar's Test
        try:
            result = mcnemar(contingency_table, exact=True)
            p_value = result.pvalue
            statistic = result.statistic
        except ValueError as e:
            return jsonify({'error': f'Error calculating McNemar’s Test: {str(e)}'}), 400

        # Evaluate significance
        significance = "significant" if p_value < 0.05 else "not significant"
        decision = "Reject the null hypothesis" if p_value < 0.05 else "Do not reject the null hypothesis"

        # Preparar el resultado
        response = {
            'test': "McNemar's Test",
            'statistic': statistic,
            'p_value': p_value,
            'significance': significance,
            'decision': decision
        }

        # Incluir la tabla de contingencia si se solicitó
        if show_contingency_table:
            response['contingency_table'] = contingency_table.to_dict()

        return jsonify({'result': response})

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

        # Validate that the numeric column exists
        if numeric_column not in dataframe.columns:
            return jsonify({'error': 'The specified numeric column was not found in the data.'}), 400

        # If a group column is specified, perform Shapiro-Wilk by group
        if group_column and group_column in dataframe.columns:
            groups = dataframe.groupby(group_column)[numeric_column].apply(list)
            shapiro_results = {}

            for group, values in groups.items():
                if len(values) < 3:
                    shapiro_results[group] = {'w_statistic': None, 'p_value': 'Insufficient data'}
                else:
                    w_stat, p_value = shapiro(values)
                    # Result by groups
                    shapiro_results[group] = clean_results({'w_statistic': w_stat, 'p_value': p_value})

            return jsonify({'type': 'grouped', 'result': shapiro_results})

        # If no group is specified, perform a global Shapiro-Wilk test
        values = dataframe[numeric_column].dropna()
        if len(values) < 3:
            return jsonify({'error': 'Insufficient data to perform the Shapiro-Wilk test.'}), 400

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

        # Validate that the numeric column exists
        if numeric_column not in dataframe.columns:
            return jsonify({'error': 'The specified numeric column was not found in the data.'}), 400

        # If a group column is specified, perform Kolmogorov-Smirnov by group
        if group_column and group_column in dataframe.columns:
            groups = dataframe.groupby(group_column)[numeric_column].apply(list)
            ks_results = {}

            for group, values in groups.items():
                if len(values) < 3:
                    ks_results[group] = {'ks_statistic': None, 'p_value': 'Insufficient data'}
                else:
                    # Perform Kolmogorov-Smirnov test assuming normal distribution
                    ks_stat, p_value = stats.kstest(values, 'norm', args=(np.mean(values), np.std(values)))
                    # Result by groups
                    ks_results[group] = clean_results({'ks_statistic': ks_stat, 'p_value': p_value})

            return jsonify({'type': 'grouped', 'result': ks_results})

        # If no group is specified, perform a global Kolmogorov-Smirnov test
        values = dataframe[numeric_column].dropna()
        if len(values) < 3:
            return jsonify({'error': 'Insufficient data to perform the Kolmogorov-Smirnov test.'}), 400

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
        comparison_type = data.get('comparison_type')  # Tipo de comparación
        alternative = data.get('alternative', 'two-sided')

        if comparison_type == 'categorical_vs_numeric':
            # Variables para comparación categórica vs. numérica
            numeric_column = data.get('numeric_column')
            categorical_column = data.get('categorical_column')

            # Validar las columnas especificadas
            if numeric_column not in dataframe.columns or categorical_column not in dataframe.columns:
                return jsonify({'error': 'The specified columns were not found in the data.'}), 400

            # Agrupar los datos por la columna categórica
            groups = dataframe.groupby(categorical_column)[numeric_column].apply(list)

            # Verificar que haya al menos dos grupos
            if len(groups) < 2:
                return jsonify({'error': 'Insufficient data to perform the Mann-Whitney U Test. At least two categories are required.'}), 400

            category_names = groups.index.tolist()

            # Calcular estadísticas descriptivas para cada grupo
            group_stats = {
                category: {
                    'mean': np.mean(data),
                    'std': np.std(data, ddof=1),  # ddof=1 para muestras
                    'count': len(data)
                } for category, data in groups.items()
            }

            # Realizar el Mann-Whitney U Test
            u_stat, p_value = stats.mannwhitneyu(groups.iloc[0], groups.iloc[1], alternative=alternative)

            result = {
                'test': 'Mann-Whitney U Test',
                'u_statistic': u_stat,
                'p_value': p_value,
                'significance': "significant" if p_value < 0.05 else "not significant",
                'decision': "Reject the null hypothesis" if p_value < 0.05 else "Do not reject the null hypothesis",
                'category1': category_names[0],
                'category2': category_names[1],
                'alternative': alternative,
                'group_statistics': group_stats  # Agregar estadísticas descriptivas
            }

        elif comparison_type == 'numeric_vs_numeric':
            # Variables para comparación numérica vs. numérica
            numeric_column1 = data.get('numeric_column1')
            numeric_column2 = data.get('numeric_column2')

            # Validar las columnas especificadas
            if numeric_column1 not in dataframe.columns or numeric_column2 not in dataframe.columns:
                return jsonify({'error': 'The specified columns were not found in the data.'}), 400

            # Extraer los datos de las columnas
            col1_data = dataframe[numeric_column1].dropna()
            col2_data = dataframe[numeric_column2].dropna()

            # Calcular estadísticas descriptivas para cada columna
            column_stats = {
                numeric_column1: {
                    'mean': np.mean(col1_data),
                    'std': np.std(col1_data, ddof=1),  # ddof=1 para muestras
                    'count': len(col1_data)
                },
                numeric_column2: {
                    'mean': np.mean(col2_data),
                    'std': np.std(col2_data, ddof=1),  # ddof=1 para muestras
                    'count': len(col2_data)
                }
            }

            # Realizar el Mann-Whitney U Test
            u_stat, p_value = stats.mannwhitneyu(col1_data, col2_data, alternative=alternative)

            result = {
                'test': 'Mann-Whitney U Test',
                'u_statistic': u_stat,
                'p_value': p_value,
                'significance': "significant" if p_value < 0.05 else "not significant",
                'decision': "Reject the null hypothesis" if p_value < 0.05 else "Do not reject the null hypothesis",
                'column1': numeric_column1,
                'column2': numeric_column2,
                'alternative': alternative,
                'column_statistics': column_stats  # Agregar estadísticas descriptivas
            }

        else:
            return jsonify({'error': 'Invalid comparison type specified. Please check the input data.'}), 400

        # Limpiar los resultados antes de enviarlos
        result = clean_results(result)
        return jsonify({'result': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# 9. Wilcoxon Signed-Rank Test
@app.route('/run_wilcoxon', methods=['POST'])
def run_wilcoxon():
    global dataframe
    try:
        # Validar si el dataframe está cargado
        if dataframe is None:
            return jsonify({'error': 'No data has been uploaded. Please upload your dataset first.'}), 400

        # Obtener los datos de la solicitud
        data = request.get_json()
        comparison_type = data.get('comparison_type')  # Tipo de comparación
        alternative = data.get('alternative', 'two-sided')

        if comparison_type == 'categorical_vs_numeric':
            # Variables para comparación categórica vs. numérica
            numeric_column = data.get('numeric_column')
            categorical_column = data.get('categorical_column')

            # Validar las columnas especificadas
            if numeric_column not in dataframe.columns or categorical_column not in dataframe.columns:
                return jsonify({'error': 'The specified columns were not found in the data.'}), 400

            # Agrupar los datos por la columna categórica
            groups = dataframe.groupby(categorical_column)[numeric_column].apply(list)

            # Verificar que haya exactamente dos grupos
            if len(groups) != 2:
                return jsonify({'error': 'The categorical variable must have exactly two categories.'}), 400

            category_names = groups.index.tolist()

            # Calcular estadísticas descriptivas para cada grupo
            group_stats = {
                category: {
                    'mean': np.mean(data),
                    'std': np.std(data, ddof=1),  # ddof=1 para muestras
                    'count': len(data)
                } for category, data in groups.items()
            }

            # Ajustar los tamaños de las muestras si son diferentes
            min_size = min(len(groups.iloc[0]), len(groups.iloc[1]))
            group1 = groups.iloc[0][:min_size]
            group2 = groups.iloc[1][:min_size]

            # Realizar el Wilcoxon Signed-Rank Test
            w_stat, p_value = stats.wilcoxon(group1, group2, alternative=alternative)

            result = {
                'test': 'Wilcoxon Signed-Rank Test',
                'w_statistic': w_stat,
                'p_value': p_value,
                'significance': "significant" if p_value < 0.05 else "not significant",
                'decision': "Reject the null hypothesis" if p_value < 0.05 else "Do not reject the null hypothesis",
                'category1': category_names[0],
                'category2': category_names[1],
                'alternative': alternative,
                'group_statistics': group_stats  # Agregar estadísticas descriptivas
            }

        elif comparison_type == 'numeric_vs_numeric':
            # Variables para comparación numérica vs. numérica
            numeric_column1 = data.get('numeric_column1')
            numeric_column2 = data.get('numeric_column2')

            # Validar las columnas especificadas
            if numeric_column1 not in dataframe.columns or numeric_column2 not in dataframe.columns:
                return jsonify({'error': 'The specified columns were not found in the data.'}), 400

            # Extraer los datos de las columnas
            col1_data = dataframe[numeric_column1].dropna()
            col2_data = dataframe[numeric_column2].dropna()

            # Calcular estadísticas descriptivas para cada columna
            column_stats = {
                numeric_column1: {
                    'mean': np.mean(col1_data),
                    'std': np.std(col1_data, ddof=1),  # ddof=1 para muestras
                    'count': len(col1_data)
                },
                numeric_column2: {
                    'mean': np.mean(col2_data),
                    'std': np.std(col2_data, ddof=1),  # ddof=1 para muestras
                    'count': len(col2_data)
                }
            }

            # Ajustar los tamaños de las muestras si son diferentes
            min_size = min(len(col1_data), len(col2_data))
            col1_data = col1_data[:min_size]
            col2_data = col2_data[:min_size]

            # Realizar el Wilcoxon Signed-Rank Test
            w_stat, p_value = stats.wilcoxon(col1_data, col2_data, alternative=alternative)

            result = {
                'test': 'Wilcoxon Signed-Rank Test',
                'w_statistic': w_stat,
                'p_value': p_value,
                'significance': "significant" if p_value < 0.05 else "not significant",
                'decision': "Reject the null hypothesis" if p_value < 0.05 else "Do not reject the null hypothesis",
                'column1': numeric_column1,
                'column2': numeric_column2,
                'alternative': alternative,
                'column_statistics': column_stats  # Agregar estadísticas descriptivas
            }

        else:
            return jsonify({'error': 'Invalid comparison type specified. Please check the input data.'}), 400

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

        # Verify that the columns exist
        if numeric_column not in dataframe.columns or categorical_column not in dataframe.columns:
            return jsonify({'error': 'The specified columns were not found in the data.'}), 400

        # Group the data by the categorical column
        groups = dataframe.groupby(categorical_column)[numeric_column].apply(list)

        # Check if there is sufficient data for each group
        if any(len(values) < 2 for values in groups):
            return jsonify({'error': 'There are groups with insufficient data to perform Kruskal-Wallis.'}), 400

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
                            # Determinar el nivel de significancia para los asteriscos
                            if p_val < 0.001:
                                significance = "***"
                            elif p_val < 0.01:
                                significance = "**"
                            elif p_val < 0.05:
                                significance = "*"
                            else:
                                significance = ""

                            # Formatear cada resultado y redondear
                            comparison = f"{i} vs {j}"
                            p_adj = f"{round(p_val, 4)} {significance}"
                            reject_h0 = "Sí" if p_val < 0.05 else "No"

                            dunn_summary.append({
                                'comparison': comparison,
                                'p_value_adjusted': p_adj,
                                'reject_h0': reject_h0
                            })


                # Agregar el resumen del test de Dunn al resultado
                result['dunn'] = dunn_summary

            except Exception as e:
                return jsonify({'error': f'Error executing Dunn\'s test: {str(e)}'}), 500

        return jsonify({'result': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 400
    
# PEARSON
@app.route('/run_pearson', methods=['POST'])
def run_pearson():
    global dataframe
    try:
        data = request.get_json()
        numeric_column1 = data.get('numeric_column1')
        numeric_column2 = data.get('numeric_column2')
        categorical_column = data.get('categorical_column')  # Variable categórica
        correlation_by_categories = data.get('correlation_by_categories', False)

        if not numeric_column1 or not numeric_column2:
            return jsonify({'error': 'You must specify two valid numeric columns.'}), 400

        # Case: General correlation without categories
        if not correlation_by_categories:
            corr, p_value = stats.pearsonr(dataframe[numeric_column1], dataframe[numeric_column2])
            
            # Replace NaN or infinity with None
            corr = None if not math.isfinite(corr) else round(corr, 4)
            p_value = None if not math.isfinite(p_value) else round(p_value, 4)

            result = {
                'correlation': corr,
                'p_value': p_value,
                'significance': "significant" if p_value and p_value < 0.05 else "not significant"
            }
            return jsonify({'result': result})

        # Case: Correlation by categories
        if not categorical_column:
            return jsonify({'error': 'You must specify a categorical column.'}), 400

        grouped = dataframe.groupby(categorical_column)
        correlation_results = []

        for category, group in grouped:
            # Asegurarse de que existan suficientes datos
            if len(group) > 1 and group[[numeric_column1, numeric_column2]].notnull().all(axis=None):
                corr, p_value = stats.pearsonr(group[numeric_column1], group[numeric_column2])
                
                # Reemplazar NaN o infinito con None
                corr = None if not math.isfinite(corr) else round(corr, 4)
                p_value = None if not math.isfinite(p_value) else round(p_value, 4)

                correlation_results.append({
                'category': category,
                'correlation': corr,
                'p_value': p_value,
                'significance': "significant" if p_value and p_value < 0.05 else "not significant"
            })
            else:
                correlation_results.append({
                    'category': category,
                    'error': 'Insufficient data to calculate the correlation.'
                })


        # Return debe ir FUERA del bucle
        return jsonify({
            'result': {
                'categories': correlation_results
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
# SPEARMAN
@app.route('/run_spearman', methods=['POST'])
def run_spearman():
    global dataframe
    try:
        data = request.get_json()
        numeric_column1 = data.get('numeric_column1')
        numeric_column2 = data.get('numeric_column2')
        categorical_column = data.get('categorical_column')  # Variable categórica
        correlation_by_categories = data.get('correlation_by_categories', False)

        if not numeric_column1 or not numeric_column2:
            return jsonify({'error': 'You must specify two valid numeric columns.'}), 400

        # Case: General correlation without categories
        if not correlation_by_categories:
            corr, p_value = stats.spearmanr(dataframe[numeric_column1], dataframe[numeric_column2])
            
            # Replace NaN or infinity with None
            corr = None if not math.isfinite(corr) else round(corr, 4)
            p_value = None if not math.isfinite(p_value) else round(p_value, 4)

            result = {
                'correlation': corr,
                'p_value': p_value,
                'significance': "significant" if p_value and p_value < 0.05 else "not significant"
            }
            return jsonify({'result': result})

        # Case: Correlation by categories
        if not categorical_column:
            return jsonify({'error': 'You must specify a categorical column.'}), 400

        grouped = dataframe.groupby(categorical_column)
        correlation_results = []

        for category, group in grouped:
            # Asegurarse de que existan suficientes datos
            if len(group) > 1 and group[[numeric_column1, numeric_column2]].notnull().all(axis=None):
                corr, p_value = stats.spearmanr(group[numeric_column1], group[numeric_column2])
                
                # Reemplazar NaN o infinito con None
                corr = None if not math.isfinite(corr) else round(corr, 4)
                p_value = None if not math.isfinite(p_value) else round(p_value, 4)

                correlation_results.append({
                'category': category,
                'correlation': corr,
                'p_value': p_value,
                'significance': "significant" if p_value and p_value < 0.05 else "not significant"
            })
            else:
                correlation_results.append({
                    'category': category,
                    'error': 'Insufficient data to calculate the correlation.'
            })


        # Return debe ir FUERA del bucle
        return jsonify({
            'result': {
                'categories': correlation_results
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500



##########################################################################################
####################################### STAT TESTS #######################################
##########################################################################################

# Ruta para la prueba Shapiro-Wilk
@app.route('/api/shapiro', methods=['POST'])
def shapiro_test():
    try:
        data = request.get_json()
        sample = data.get('sample')

        # Validar que la muestra no esté vacía
        if not sample or not isinstance(sample, list) or len(sample) == 0:
            return jsonify({
                'error': 'Invalid input',
                'message': 'The sample must be a non-empty list of numbers.'
            }), 400

        # Validar tamaño mínimo de muestra para Shapiro-Wilk
        if len(sample) < 3:
            return jsonify({
                'error': 'Sample size too small',
                'message': 'The sample size must be at least 3 for the Shapiro-Wilk test.'
            }), 400

        # Ejecutar la prueba Shapiro-Wilk
        stat, p_value = stats.shapiro(sample)

        # Determinar la significancia del p-valor
        if p_value < 0.05:
            significance = "significant"
            reject_null = "Reject the null hypothesis"
        elif p_value < 0.1:
            significance = "marginally significant"
            reject_null = "Potential rejection of the null hypothesis"
        else:
            significance = "not significant"
            reject_null = "Do not reject the null hypothesis"

        # Respuesta exitosa
        return jsonify({
            'test': 'Shapiro-Wilk',
            'statistic': stat,
            'pValue': p_value,
            'significance': significance,
            'decision': reject_null
        })

    except Exception as e:
        # Manejar errores inesperados del servidor
        return jsonify({
            'error': 'Server error',
            'message': f'An unexpected error occurred: {str(e)}'
        }), 500


# Ruta para la prueba Kolmogorov-Smirnov
@app.route('/api/kolmogorov', methods=['POST'])
def kolmogorov_test():
    try:
        data = request.get_json()
        sample = data.get('sample')

        # Validar que la muestra no esté vacía
        if not sample or not isinstance(sample, list) or len(sample) == 0:
            return jsonify({
                'error': 'Invalid input',
                'message': 'The sample must be a non-empty list of numbers.'
            }), 400

        # Validar tamaño mínimo de muestra
        if len(sample) < 3:
            return jsonify({
                'error': 'Sample size too small',
                'message': 'The sample size must be at least 3 for the Kolmogorov-Smirnov test.'
            }), 400

        # Calcular estadísticas y realizar la prueba Kolmogorov-Smirnov
        sample_mean = np.mean(sample)
        sample_std = np.std(sample, ddof=1)
        stat, p_value = stats.kstest(sample, 'norm', args=(sample_mean, sample_std))

        # Determinar la significancia del p-valor
        if p_value < 0.05:
            significance = "significant"
            reject_null = "Reject the null hypothesis"
        elif p_value < 0.1:
            significance = "marginally significant"
            reject_null = "Potential rejection of the null hypothesis"
        else:
            significance = "not significant"
            reject_null = "Do not reject the null hypothesis"

        # Respuesta exitosa
        return jsonify({
            'test': 'Kolmogorov-Smirnov',
            'statistic': stat,
            'pValue': p_value,
            'significance': significance,
            'decision': reject_null
        })

    except Exception as e:
        # Manejar errores inesperados del servidor
        return jsonify({
            'error': 'Server error',
            'message': f'An unexpected error occurred: {str(e)}'
        }), 500


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
            return jsonify({'error': "At least two groups with data are required to perform Levene's test."}), 400

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
        # Log the error on the server for debugging
        print(f'Error executing Levene\'s test: {str(e)}')
        return jsonify({'error': 'An error occurred while performing Levene\'s test. Please check your input data and try again.'}), 500


# Ruta para la prueba t de Student con opción pareado y unilateral/bilateral
@app.route('/api/ttest', methods=['POST'])
def t_test():
    try:
        data = request.get_json()
        sample1 = data['sample1']
        sample2 = data['sample2']
        paired = data.get('paired', False)
        alternative = data.get('alternative', 'two-sided')  # Configuración de unilateral/bilateral

        # Calcular las estadísticas descriptivas para cada muestra
        stats_group1 = {
            'mean': np.mean(sample1),
            'std': np.std(sample1, ddof=1),  # Desviación estándar muestral
            'count': len(sample1)
        }
        stats_group2 = {
            'mean': np.mean(sample2),
            'std': np.std(sample2, ddof=1),  # Desviación estándar muestral
            'count': len(sample2)
        }

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
            'decision': reject_null,
            'group1_statistics': stats_group1,  # Estadísticas del grupo 1
            'group2_statistics': stats_group2   # Estadísticas del grupo 2
        })

    except Exception as e:
        # Registrar el error en el servidor para depuración
        print(f'Error executing T-Test: {str(e)}')
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500


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
        return jsonify({'error': 'The provided data must be numeric.'}), 400

    # Validate that observed and expected have the same length if expected is provided
    if expected and len(observed) != len(expected):
        return jsonify({'error': 'Observed and expected frequencies must have the same size.'}), 400

    try:
        # Perform the Chi-Square test
        if expected:
            stat, p_value = chisquare(f_obs=observed, f_exp=expected)
        else:
            stat, p_value = chisquare(f_obs=observed)
        
        # Return the results
        return jsonify({'test': 'Chi-Square (Goodness of Fit)', 'statistic': stat, 'pValue': p_value})
    except Exception as e:
        # Handle errors during test execution
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500




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
            return jsonify({'error': 'anova_test_error'}), 400

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
            if len(groups) < 3:
                return jsonify({'error': 'anova_test_error'}), 400

            # Prepare data for Tukey HSD
            all_data = []
            labels = []
            for i, group in enumerate(groups):
                if len(group) == 0:
                    return jsonify({'error': 'anova_test_error'}), 400

                all_data.extend(group)
                labels.extend([f'Group {i+1}'] * len(group))

            df = pd.DataFrame({'value': all_data, 'group': labels})

            tukey = mc.pairwise_tukeyhsd(df['value'], df['group'], alpha=0.05)

            # Procesar resultados de Tukey HSD
            tukey_summary = "Multiple comparisons (Tukey HSD):\n"
            for result in tukey.summary().data[1:]:
                significance = (
                    "***" if result[3] < 0.001 else
                    "**" if result[3] < 0.01 else
                    "*" if result[3] < 0.05 else ""
                )
                tukey_summary += (
                    f"----------------------------\n"
                    f"Comparison: {result[0]} vs {result[1]}\n"
                    f"  • Mean difference: {result[2]:.4f}\n"
                    f"  • Adjusted p-Value: {result[3]:.4f} {significance}\n"
                    f"  • Lower IC: {result[4]:.4f}, Upper IC: {result[5]:.4f}\n"
                    f"  • Reject H0: {'Yes' if result[6] else 'No'}\n"
                )

            anova_results['tukey'] = tukey_summary

        return jsonify(anova_results)

    except Exception:
        # Mensaje de error general
        return jsonify({'error': 'anova_test_error'}), 500




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
            return jsonify({'error': 'The vectors factor1, factor2, and values must have the same length.'}), 400

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
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

# Mann-Whitney U Test con opción unilateral/bilateral
@app.route('/api/mannwhitney', methods=['POST'])
def mann_whitney():
    try:
        data = request.get_json()
        sample1 = data['sample1']
        sample2 = data['sample2']
        alternative = data.get('alternative', 'two-sided')  # Configuración de unilateral/bilateral

        # Calcular las estadísticas descriptivas para cada muestra
        stats_group1 = {
            'mean': np.mean(sample1),
            'std': np.std(sample1, ddof=1),  # Desviación estándar muestral
            'count': len(sample1)
        }
        stats_group2 = {
            'mean': np.mean(sample2),
            'std': np.std(sample2, ddof=1),  # Desviación estándar muestral
            'count': len(sample2)
        }

        # Realizar la prueba Mann-Whitney
        stat, p_value = stats.mannwhitneyu(sample1, sample2, alternative=alternative)

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

        # Devolver la respuesta con estadísticas descriptivas y resultados de la prueba Mann-Whitney
        return jsonify({
            'test': 'Mann-Whitney U',
            'statistic': stat,
            'pValue': p_value,
            'significance': significance,
            'decision': reject_null,
            'group1_statistics': stats_group1,  # Estadísticas del grupo 1
            'group2_statistics': stats_group2   # Estadísticas del grupo 2
        })
    
    except Exception as e:
        # Registrar el error en el servidor para depuración
        print(f'Error executing Mann-Whitney: {str(e)}')
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500




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
            return jsonify({'error': 'At least two groups with data are required to perform the Kruskal-Wallis test.'}), 400

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
                return jsonify({'error': 'At least three groups are required to perform multiple comparisons.'}), 400

            try:
                # Preparar los datos para las comparaciones múltiples
                all_data = []
                labels = []
                for i, group in enumerate(groups):
                    all_data.extend(group)
                    labels.extend([f'Group {i+1}'] * len(group))

                # Convertir a DataFrame para usar en las pruebas
                df = pd.DataFrame({'value': all_data, 'group': labels})

                # Realizar la prueba de Dunn
                if p_value_adjustment not in ['bonferroni', 'holm']:
                    return jsonify({'error': "Invalid adjustment method. Use 'bonferroni' or 'holm'."}), 400
                
                dunn = sp.posthoc_dunn(df, val_col='value', group_col='group', p_adjust=p_value_adjustment)

                # Procesar los resultados de Dunn
                dunn_summary = f"Multiple comparisons (Dunn with correction {p_value_adjustment.capitalize()}):\n"
                for i in range(len(dunn)):
                    for j in range(i+1, len(dunn)):
                        # Determinar el número de asteriscos en función del valor p ajustado
                        p_val = dunn.iloc[i, j]
                        if p_val < 0.001:
                            significance = "***"
                        elif p_val < 0.01:
                            significance = "**"
                        elif p_val < 0.05:
                            significance = "*"
                        else:
                            significance = ""

                        # Generar el texto para cada comparación
                        dunn_summary += (
                            f"----------------------------\n"
                            f"Compararison: Group {i + 1} vs Group {j + 1}\n"
                            f"  • Adjusted p-Value: {p_val:.4f} {significance}\n"
                        )

                # Agregar el resumen de Dunn al resultado
                kw_results['dunn'] = dunn_summary

            except Exception as e:
                return jsonify({'error': f'Error executing Dunn\'s test: {str(e)}'}), 500

        return jsonify(kw_results)

    except Exception as e:
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500


# Friedman Test con comparaciones múltiples
@app.route('/api/friedman', methods=['POST'])
def friedman_test():
    try:
        data = request.get_json()
        groups = data.get('groups', [])
        multiple_comparisons = data.get('multipleComparisons', False)

        # Validation to ensure there are at least 3 groups
        if len(groups) < 3:
            return jsonify({'error': 'At least 3 groups are required for the Friedman test.'}), 400

        # Validate that all groups have the same number of observations
        num_observations = len(groups[0])
        if not all(len(g) == num_observations for g in groups):
            return jsonify({'error': 'All groups must have the same number of observations.'}), 400


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
                df = pd.DataFrame({f'Group {i+1}': group for i, group in enumerate(groups)})

                # Realizar la prueba de Nemenyi
                nemenyi = sp.posthoc_nemenyi_friedman(df)

                # Procesar los resultados de Nemenyi en formato de texto
                nemenyi_summary = "Multiple comparisons (Nemenyi):\n"
                for i in range(len(nemenyi)):
                    for j in range(i + 1, len(nemenyi)):
                        # Determinar el número de asteriscos en función del valor p ajustado
                        p_val = nemenyi.iloc[i, j]
                        if p_val < 0.001:
                            significance = "***"
                        elif p_val < 0.01:
                            significance = "**"
                        elif p_val < 0.05:
                            significance = "*"
                        else:
                            significance = ""

                        # Generar el texto para cada comparación
                        nemenyi_summary += (
                            f"----------------------------\n"
                            f"Comparison: Group {i + 1} vs Group {j + 1}\n"
                            f"  • Adjusted p-Value: {p_val:.3f} {significance}\n"
                        )

                # Agregar el resumen de Nemenyi al resultado
                friedman_results['nemenyi'] = nemenyi_summary

            except Exception as e:
                return jsonify({'error': f'Error executing the Nemenyi test: {str(e)}'}), 500

        return jsonify(friedman_results)

    except Exception as e:
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500



# Fisher exact test
@app.route('/api/fisher', methods=['POST'])
def fisher_test():
    try:
        data = request.get_json()
        observed = data['observed']

        # Validar que sea una tabla 2x2
        if len(observed) != 2 or len(observed[0]) != 2:
            return jsonify({'error': 'Fisher\'s exact test requires a 2x2 contingency table.'}), 400

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
         

        response = {
            'test': 'Fisher',
            'oddsratio': oddsratio,
            'pValue': p_value,
            'significance': significance,
            'decision': reject_null,
        }

        if used_smoothing:
            response['contiene0'] = "An additive smoothing was applied to handle the 0s. This will add 1 to all cells in the table and allow the test to be performed."

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
            return jsonify({'error': 'McNemar\'s test requires a 2x2 contingency table.'}), 400

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
            return jsonify({'error': 'Cochran\'s Q test requires at least 3 treatments/conditions.'}), 400

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
            return jsonify({'error': 'The two samples must have the same number of observations for the Wilcoxon test.'}), 400

        # Calcular las estadísticas descriptivas para cada muestra
        stats_group1 = {
            'mean': np.mean(sample1),
            'std': np.std(sample1, ddof=1),  # Desviación estándar muestral
            'count': len(sample1)
        }
        stats_group2 = {
            'mean': np.mean(sample2),
            'std': np.std(sample2, ddof=1),  # Desviación estándar muestral
            'count': len(sample2)
        }

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

        # Respuesta con los resultados y estadísticas descriptivas
        return jsonify({
            'test': 'Wilcoxon Signed-Rank',
            'statistic': stat,
            'pValue': p_value,
            'significance': significance,
            'decision': reject_null,
            'group1_statistics': stats_group1,  # Estadísticas del grupo 1
            'group2_statistics': stats_group2   # Estadísticas del grupo 2
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
        return jsonify({'error': 'Both samples must contain data.'}), 400

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
            plt.title('Scatter plot with trend line')
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
        return jsonify({'error': 'Both samples must contain data.'}), 400

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
            plt.plot(sample1, m * sample1 + b, color='red', label='Trend line')

            plt.title('Scatter plot for Spearman')
            plt.xlabel('Sample 1')
            plt.ylabel('Sample 2')
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
    try: 
        data = request.get_json()
        sample1 = data.get('sample1', [])
        sample2 = data.get('sample2', [])
        alternative = data.get('alternative', 'two-sided')  # Configuración de unilateral/bilateral

        # Validar que ambas muestras tengan datos
        if not sample1 or not sample2:
            return jsonify({'error': 'Both samples must contain data.'}), 400

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

    except Exception as e:
        # Registrar el error en el servidor para depuración
        print(f'Error executing Welch T-Test: {str(e)}')
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500



# Ruta para la prueba t de una media
@app.route('/api/one_sample_ttest', methods=['POST'])
def one_sample_t_test():
    data = request.get_json()
    sample = data.get('sample', [])
    population_mean = data.get('population_mean', 0)  # Valor de referencia

    # Validar que la muestra tenga datos
    if not sample:
        return jsonify({'error': 'The sample must contain data.'}), 400

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
            raise ValueError("The number of predictors and responses must match.")

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
            plt.scatter(predictors, response, color='blue', label='Data')
            
            # Generate the regression line
            predicted_values = intercept + coefficients[0] * predictors.flatten()
            plt.plot(predictors, predicted_values, color='red', label='Regression Line')

            # Customize the plot
            plt.title('Linear Regression')
            plt.xlabel('Predictor')
            plt.ylabel('Response')

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
            'model': 'Linear Regression',
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
        print(f"Error in linear regression: {str(e)}")
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
            raise ValueError("The number of predictors and responses must match.")

        # Verificar que la respuesta sea binaria (0 o 1)
        if not np.array_equal(np.unique(response), [0, 1]):
            raise ValueError("The response variable must be binary (0 or 1).")

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
            'model': 'Logistic Regression',
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
        print(f"Error in logistic regression: {str(e)}")
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
            raise ValueError("The number of observations must match in predictors, time, and event.")

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
            'model': 'Cox Regression',
            'coefficients': coefficients,
            'p_values': p_values,
            'confidence_intervals': confidence_intervals
        })
    except Exception as e:
        print(f"Error in Cox regression: {str(e)}")
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
            raise ValueError("The number of predictors and responses must match.")

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
            'model': 'Poisson Regression',
            'intercept': intercept,
            'intercept_pvalue': intercept_pvalue,
            'coefficients': coefficients,
            'p_values': p_values,
            'z_values': z_values,
            'confidence_intervals': confidence_intervals
        })
    except Exception as e:
        print(f"Error in Poisson regression: {str(e)}")
        return jsonify({'error': str(e)}), 500



# Ruta para llamar a Render y que no se apague
@app.route('/ping', methods=['HEAD', 'GET'])
def ping():
    return '', 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))  # Por defecto, usa 8080
    app.run(debug=False, host='0.0.0.0', port=port)










