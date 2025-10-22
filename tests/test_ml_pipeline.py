"""
Testes para pipeline de Machine Learning.
"""
import tempfile
from unittest.mock import MagicMock, Mock, patch

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


class TestDataProcessing:
    """Testes para processamento de dados."""

    def test_data_loading(self, sample_flight_data, temp_directory):
        """Testa carregamento de dados."""
        # Salva dados de teste
        test_file = temp_directory / 'test_data.csv'
        sample_flight_data.to_csv(test_file, index=False)

        # Testa se consegue carregar dados
        loaded_data = pd.read_csv(test_file)

        assert isinstance(loaded_data, pd.DataFrame)
        assert len(loaded_data) == len(sample_flight_data)
        assert list(loaded_data.columns) == list(sample_flight_data.columns)

    def test_data_preprocessing(self, sample_flight_data):
        """Testa preprocessamento de dados."""
        # Verifica estrutura básica dos dados
        assert 'is_cancelled' in sample_flight_data.columns
        assert 'flight_id' in sample_flight_data.columns

        # Verifica tipos de dados
        assert sample_flight_data['is_cancelled'].dtype in [
            'int64',
            'int32',
            'bool',
        ]
        assert sample_flight_data['flight_id'].dtype == 'object'

    def test_feature_engineering(self, sample_flight_data):
        """Testa engenharia de features."""
        # Cria features temporais simples
        data_with_features = sample_flight_data.copy()

        # Extrai hora da partida
        data_with_features['departure_hour'] = pd.to_datetime(
            data_with_features['scheduled_departure']
        ).dt.hour

        assert 'departure_hour' in data_with_features.columns
        assert data_with_features['departure_hour'].min() >= 0
        assert data_with_features['departure_hour'].max() <= 23

    def test_categorical_encoding(self, sample_flight_data):
        """Testa codificação de variáveis categóricas."""
        from sklearn.preprocessing import LabelEncoder

        # Codifica variável categórica
        le = LabelEncoder()
        encoded_airline = le.fit_transform(sample_flight_data['airline'])

        assert len(encoded_airline) == len(sample_flight_data)
        assert encoded_airline.dtype in ['int32', 'int64']
        assert encoded_airline.min() >= 0


class TestModelTraining:
    """Testes para treinamento de modelos."""

    def test_train_test_split(self, sample_processed_features):
        """Testa divisão de dados para treinamento."""
        X = sample_processed_features.drop('is_cancelled', axis=1)
        y = sample_processed_features['is_cancelled']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        assert len(X_train) + len(X_test) == len(X)
        assert len(y_train) + len(y_test) == len(y)
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)

    def test_model_training_random_forest(self, sample_processed_features):
        """Testa treinamento com Random Forest."""
        from sklearn.ensemble import RandomForestClassifier

        X = sample_processed_features.drop('is_cancelled', axis=1)
        y = sample_processed_features['is_cancelled']

        # Treina modelo
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        # Verifica se modelo foi treinado
        assert hasattr(model, 'feature_importances_')
        assert len(model.feature_importances_) == len(X.columns)
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')

    def test_model_training_logistic_regression(
        self, sample_processed_features
    ):
        """Testa treinamento com Regressão Logística."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        X = sample_processed_features.drop('is_cancelled', axis=1)
        y = sample_processed_features['is_cancelled']

        # Normaliza features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Treina modelo
        model = LogisticRegression(random_state=42)
        model.fit(X_scaled, y)

        # Verifica se modelo foi treinado
        assert hasattr(model, 'coef_')
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')

    def test_model_training_gradient_boosting(self, sample_processed_features):
        """Testa treinamento com Gradient Boosting."""
        from sklearn.ensemble import GradientBoostingClassifier

        X = sample_processed_features.drop('is_cancelled', axis=1)
        y = sample_processed_features['is_cancelled']

        # Treina modelo
        model = GradientBoostingClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        # Verifica se modelo foi treinado
        assert hasattr(model, 'feature_importances_')
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')


class TestModelEvaluation:
    """Testes para avaliação de modelos."""

    def test_model_predictions(self, mock_model, sample_processed_features):
        """Testa predições do modelo."""
        X = sample_processed_features.drop('is_cancelled', axis=1)

        # Configura mock
        mock_model.predict.return_value = np.array([0, 0, 1, 0, 1])

        predictions = mock_model.predict(X)

        assert len(predictions) == len(X)
        assert all(pred in [0, 1] for pred in predictions)

    def test_model_probabilities(self, mock_model, sample_processed_features):
        """Testa probabilidades do modelo."""
        X = sample_processed_features.drop('is_cancelled', axis=1)

        # Configura mock
        mock_model.predict_proba.return_value = np.array(
            [[0.8, 0.2], [0.9, 0.1], [0.3, 0.7], [0.7, 0.3], [0.2, 0.8]]
        )

        probabilities = mock_model.predict_proba(X)

        assert probabilities.shape == (len(X), 2)
        assert np.all((probabilities >= 0) & (probabilities <= 1))
        assert np.allclose(probabilities.sum(axis=1), 1.0)

    def test_evaluation_metrics(self, sample_processed_features):
        """Testa cálculo de métricas de avaliação."""
        from sklearn.ensemble import RandomForestClassifier

        X = sample_processed_features.drop('is_cancelled', axis=1)
        y = sample_processed_features['is_cancelled']

        # Treina modelo
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        # Predições
        predictions = model.predict(X)

        # Calcula métricas
        accuracy = accuracy_score(y, predictions)
        precision = precision_score(
            y, predictions, average='weighted', zero_division=0
        )
        recall = recall_score(
            y, predictions, average='weighted', zero_division=0
        )
        f1 = f1_score(y, predictions, average='weighted', zero_division=0)

        # Verifica se métricas estão no intervalo correto
        assert 0 <= accuracy <= 1
        assert 0 <= precision <= 1
        assert 0 <= recall <= 1
        assert 0 <= f1 <= 1

    def test_cross_validation(self, sample_processed_features):
        """Testa validação cruzada."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score

        X = sample_processed_features.drop('is_cancelled', axis=1)
        y = sample_processed_features['is_cancelled']

        # Modelo
        model = RandomForestClassifier(n_estimators=10, random_state=42)

        # Validação cruzada com 2 folds (dados pequenos)
        scores = cross_val_score(model, X, y, cv=2, scoring='accuracy')

        assert len(scores) == 2
        assert all(0 <= score <= 1 for score in scores)


class TestModelPersistence:
    """Testes para persistência de modelos."""

    def test_save_model(self, temp_directory, sample_processed_features):
        """Testa salvamento de modelo."""
        from sklearn.ensemble import RandomForestClassifier

        X = sample_processed_features.drop('is_cancelled', axis=1)
        y = sample_processed_features['is_cancelled']

        # Treina modelo
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        # Salva modelo
        model_path = temp_directory / 'test_model.pkl'
        joblib.dump(model, model_path)

        # Verifica se arquivo foi criado
        assert model_path.exists()
        assert model_path.stat().st_size > 0

    def test_load_model(self, temp_directory, sample_processed_features):
        """Testa carregamento de modelo."""
        from sklearn.ensemble import RandomForestClassifier

        X = sample_processed_features.drop('is_cancelled', axis=1)
        y = sample_processed_features['is_cancelled']

        # Treina e salva modelo
        original_model = RandomForestClassifier(
            n_estimators=10, random_state=42
        )
        original_model.fit(X, y)

        model_path = temp_directory / 'test_model.pkl'
        joblib.dump(original_model, model_path)

        # Carrega modelo
        loaded_model = joblib.load(model_path)

        # Verifica se modelo carregado funciona
        assert hasattr(loaded_model, 'predict')

        # Testa predições
        original_predictions = original_model.predict(X)
        loaded_predictions = loaded_model.predict(X)

        np.testing.assert_array_equal(original_predictions, loaded_predictions)

    def test_model_versioning(self, temp_directory):
        """Testa versionamento de modelos."""
        from sklearn.ensemble import RandomForestClassifier

        # Cria modelos com versões diferentes
        versions = ['v1.0.0', 'v1.1.0', 'v2.0.0']

        for version in versions:
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            model_path = temp_directory / f'model_{version}.pkl'
            joblib.dump(model, model_path)

            assert model_path.exists()

        # Verifica se todos os modelos foram salvos
        model_files = list(temp_directory.glob('model_*.pkl'))
        assert len(model_files) == len(versions)


class TestFeatureImportance:
    """Testes para importância de features."""

    def test_feature_importance_extraction(self, sample_processed_features):
        """Testa extração de importância das features."""
        from sklearn.ensemble import RandomForestClassifier

        X = sample_processed_features.drop('is_cancelled', axis=1)
        y = sample_processed_features['is_cancelled']

        # Treina modelo
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        # Extrai importância
        feature_importance = model.feature_importances_

        assert len(feature_importance) == len(X.columns)
        assert np.all(feature_importance >= 0)
        assert np.isclose(feature_importance.sum(), 1.0)

    def test_feature_importance_ranking(self, sample_processed_features):
        """Testa ranking de importância das features."""
        from sklearn.ensemble import RandomForestClassifier

        X = sample_processed_features.drop('is_cancelled', axis=1)
        y = sample_processed_features['is_cancelled']

        # Treina modelo
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        # Cria DataFrame com importância
        feature_importance_df = pd.DataFrame(
            {'feature': X.columns, 'importance': model.feature_importances_}
        ).sort_values('importance', ascending=False)

        assert len(feature_importance_df) == len(X.columns)
        assert (
            feature_importance_df['importance'].iloc[0]
            >= feature_importance_df['importance'].iloc[-1]
        )


class TestModelPrediction:
    """Testes para predições do modelo."""

    def test_single_prediction(self, sample_processed_features):
        """Testa predição para uma única amostra."""
        from sklearn.ensemble import RandomForestClassifier

        X = sample_processed_features.drop('is_cancelled', axis=1)
        y = sample_processed_features['is_cancelled']

        # Treina modelo
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        # Predição para primeira linha
        single_sample = X.iloc[[0]]
        prediction = model.predict(single_sample)
        probability = model.predict_proba(single_sample)

        assert len(prediction) == 1
        assert prediction[0] in [0, 1]
        assert probability.shape == (1, 2)
        assert 0 <= probability[0, 0] <= 1
        assert 0 <= probability[0, 1] <= 1

    def test_batch_prediction(self, sample_processed_features):
        """Testa predição em lote."""
        from sklearn.ensemble import RandomForestClassifier

        X = sample_processed_features.drop('is_cancelled', axis=1)
        y = sample_processed_features['is_cancelled']

        # Treina modelo
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        # Predições em lote
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)

        assert len(predictions) == len(X)
        assert all(pred in [0, 1] for pred in predictions)
        assert probabilities.shape == (len(X), 2)

    def test_prediction_consistency(self, sample_processed_features):
        """Testa consistência das predições."""
        from sklearn.ensemble import RandomForestClassifier

        X = sample_processed_features.drop('is_cancelled', axis=1)
        y = sample_processed_features['is_cancelled']

        # Treina modelo
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        # Múltiplas predições da mesma amostra
        sample = X.iloc[[0]]

        pred1 = model.predict(sample)
        pred2 = model.predict(sample)

        # Deve ser consistente
        np.testing.assert_array_equal(pred1, pred2)
