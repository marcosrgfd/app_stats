from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

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


# Nueva ruta para el análisis descriptivo básico a partir de un archivo CSV cargado
@app.route('/upload_csv_basic_analysis', methods=['POST'])
def upload_csv_basic_analysis():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No se encontró el archivo.'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'El archivo no tiene nombre.'}), 400

        # Leer el archivo CSV
        df = pd.read_csv(io.StringIO(file.stream.read().decode("utf-8")))
        
        # Realizar el análisis descriptivo
        result = {}
        for column in df.columns:
            if pd.api.types.is_numeric_dtype(df[column]):
                stats = df[column].describe()
                result[column] = {
                    'mean': stats['mean'],
                    'median': df[column].median(),
                    'mode': df[column].mode().tolist(),
                    'std': stats['std'],
                    'variance': df[column].var(),
                    'min': stats['min'],
                    'max': stats['max'],
                    'range': stats['max'] - stats['min'],
                    'coef_var': (stats['std'] / stats['mean']) * 100 if stats['mean'] != 0 else None
                }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 400


# Función para calcular estadísticas descriptivas comunes
def calculate_descriptive_statistics(data_series):
    # Calcular estadísticas descriptivas
    mean = data_series.mean()
    median = data_series.median()
    mode = data_series.mode().tolist()  # Puede tener más de un valor
    std = data_series.std()
    variance = data_series.var()
    min_value = data_series.min()
    max_value = data_series.max()
    range_value = max_value - min_value
    coef_var = (std / mean) * 100 if mean != 0 else None

    # Crear un histograma y convertirlo a base64
    plt.figure(figsize=(6, 4))
    plt.hist(data_series.dropna(), bins=10, color='skyblue', edgecolor='black')
    plt.title('Histograma de Datos')
    plt.xlabel('Valor')
    plt.ylabel('Frecuencia')
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    encoded_img = base64.b64encode(img.getvalue()).decode()

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
        'histogram': encoded_img  # Imagen codificada en base64
    }

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)

