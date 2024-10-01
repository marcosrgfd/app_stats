from flask import Flask, request, jsonify
from statsmodels.stats.power import TTestIndPower, NormalIndPower  # Importaciones corregidas
import numpy as np
import math

app = Flask(__name__)

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

        return jsonify({'sample_size': round(sample_size)})

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

        return jsonify({'sample_size': round(sample_size)})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
