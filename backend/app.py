# backend/app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from analysis import (
    perform_correlation_analysis,
    perform_linear_regression,
    perform_chi_square_test,
    perform_time_series_analysis,
    perform_descriptive_statistics,
)
import logging

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Parse the incoming JSON data
        data = request.get_json()
        logging.debug(f"Received data: {data}")

        # Validate input data
        if not data or 'dataset' not in data or 'selected_analyses' not in data:
            logging.error("Invalid input data")
            return jsonify({"error": "Invalid input data"}), 400

        # Extract dataset and analysis details from the request
        dataset = pd.DataFrame(data['dataset'])
        selected_analyses = data['selected_analyses']
        logging.info(f"Number of analyses requested: {len(selected_analyses)}")

        results = []

        for analysis in selected_analyses:
            analysis_type = analysis['analysis_name']
            fields = analysis['required_fields']
            logging.info(f"Performing {analysis_type} with fields {fields}")

            # Run the requested analysis
            if analysis_type == 'Correlation Analysis':
                result = perform_correlation_analysis(dataset, fields['numerical_columns'])
            elif analysis_type == 'Linear Regression':
                result = perform_linear_regression(
                    dataset,
                    fields['numerical_columns'][0],
                    fields['numerical_columns'][1]
                )
            elif analysis_type == 'Chi-Square Test':
                result = perform_chi_square_test(dataset, fields['categorical_columns'])
            elif analysis_type == 'Time Series Analysis':
                result = perform_time_series_analysis(
                    dataset,
                    fields['date_columns'][0],
                    fields['numerical_columns'][0]
                )
            elif analysis_type == 'Descriptive Statistics':
                result = perform_descriptive_statistics(dataset, fields)
            else:
                result = {"error": f"Unsupported analysis type: {analysis_type}"}
                logging.warning(f"Unsupported analysis type requested: {analysis_type}")

            # Add analysis name to result
            result['analysis_name'] = analysis_type
            results.append(result)

        logging.debug(f"Analysis results: {results}")
        return jsonify({"results": results}), 200
    except Exception as e:
        logging.error(f"Error during analysis: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
