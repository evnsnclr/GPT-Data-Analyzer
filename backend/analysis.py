# backend/analysis.py

import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import logging
from scipy.stats import chi2_contingency

# Function to perform correlation analysis
def perform_correlation_analysis(df, numerical_columns):
    try:
        logging.debug(f"Computing correlation matrix for columns: {numerical_columns}")
        # Calculate correlation matrix
        correlation_matrix = df[numerical_columns].corr()

        # Convert the matrix to a dictionary to return it as JSON
        correlation_result = correlation_matrix.to_dict()

        return {
            "analysis_type": "Correlation Analysis",
            "correlation_matrix": correlation_result
        }
    except Exception as e:
        logging.error(f"Error in perform_correlation_analysis: {str(e)}", exc_info=True)
        return {"error": str(e)}

# Function to perform linear regression
def perform_linear_regression(df, dependent_var, independent_var):
    try:
        logging.debug(f"Performing linear regression with dependent_var: {dependent_var}, independent_var: {independent_var}")
        # Reshape data for sklearn
        X = df[[independent_var]].values.reshape(-1, 1)
        y = df[dependent_var].values

        # Fit the linear regression model
        model = LinearRegression()
        model.fit(X, y)

        # Get regression coefficients and intercept
        slope = model.coef_[0]
        intercept = model.intercept_

        # Optionally, predict values
        predictions = model.predict(X)

        return {
            "analysis_type": "Linear Regression",
            "slope": slope,
            "intercept": intercept,
            "predictions": predictions.tolist()
        }
    except Exception as e:
        logging.error(f"Error in perform_linear_regression: {str(e)}", exc_info=True)
        return {"error": str(e)}

# Function to perform Chi-Square Test
def perform_chi_square_test(df, categorical_columns):
    try:
        logging.debug(f"Performing Chi-Square Test on columns: {categorical_columns}")
        # Create a contingency table
        contingency_table = pd.crosstab(df[categorical_columns[0]], df[categorical_columns[1]])
        logging.debug(f"Contingency table:\n{contingency_table}")

        # Perform Chi-Square Test
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        logging.debug(f"Chi2: {chi2}, p-value: {p}, dof: {dof}")

        return {
            "analysis_type": "Chi-Square Test",
            "chi2_statistic": chi2,
            "p_value": p,
            "degrees_of_freedom": dof,
            "expected_frequencies": expected.tolist()
        }
    except Exception as e:
        logging.error(f"Error in perform_chi_square_test: {str(e)}", exc_info=True)
        return {"error": str(e)}

# Function to perform Time Series Analysis
def perform_time_series_analysis(df, date_column, numerical_column):
    try:
        logging.debug(f"Performing Time Series Analysis on date_column: {date_column}, numerical_column: {numerical_column}")
        # Ensure the date column is in datetime format
        df[date_column] = pd.to_datetime(df[date_column])

        # Set the date column as the index
        df.set_index(date_column, inplace=True)

        # Resample data (e.g., monthly averages)
        time_series = df[numerical_column].resample('M').mean()
        logging.debug(f"Time series data:\n{time_series}")

        return {
            "analysis_type": "Time Series Analysis",
            "time_series": time_series.to_dict()
        }
    except Exception as e:
        logging.error(f"Error in perform_time_series_analysis: {str(e)}", exc_info=True)
        return {"error": str(e)}

# Function to perform Descriptive Statistics
def perform_descriptive_statistics(df, fields):
    try:
        logging.debug(f"Computing descriptive statistics for fields: {fields}")
        stats = df.describe(include='all').to_dict()
        return {
            "analysis_type": "Descriptive Statistics",
            "statistics": stats
        }
    except Exception as e:
        logging.error(f"Error in perform_descriptive_statistics: {str(e)}", exc_info=True)
        return {"error": str(e)}
