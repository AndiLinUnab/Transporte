# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Configurar la página
st.set_page_config(
    page_title="Clasificador de Fallas - Metro",
    page_icon="🚇",
    layout="wide",
    initial_sidebar_state="expanded"
)

class MetroStreamlitApp:
    def __init__(self):
        self.models = {}
        self.preprocessor = None
        self.feature_names = []
        self.metadata = {}
        
    def load_artifacts(self):
        """Cargar modelos y preprocesadores"""
        try:
            # Intentar cargar desde la nueva estructura
            self.preprocessor = joblib.load('model_artifacts/preprocessor.pkl')
            self.feature_names = joblib.load('model_artifacts/feature_names.pkl')
            self.metadata = joblib.load('model_artifacts/metadata.pkl')
            
            # Cargar modelos individuales
            model_files = {
                'Random Forest': 'random_forest_model.pkl',
                'XGBoost': 'xgboost_model.pkl', 
                'SVM': 'svm_model.pkl',
                'KNN': 'knn_model.pkl',
                'Logistic Regression': 'logistic_regression_model.pkl'
            }
            
            for model_name, filename in model_files.items():
                try:
                    self.models[model_name] = joblib.load(f'model_artifacts/{filename}')
                except FileNotFoundError:
                    st.warning(f"Modelo {model_name} no encontrado")
            
            # También cargar modelos legacy si existen
            try:
                self.models['SVC Legacy'] = joblib.load('svc_model.jb')
                st.sidebar.success("✅ Modelo SVC legacy cargado")
            except:
                pass
                
            try:
                self.legacy_scaler = joblib.load('scaler.jb')
                st.sidebar.success("✅ Scaler legacy cargado")
            except:
                pass
            
            st.sidebar.success("🎉 Todos los artefactos cargados exitosamente!")
            return True
            
        except Exception as e:
            st.error(f"❌ Error cargando artefactos: {e}")
            st.info("💡 Asegúrate de que los archivos .pkl estén en la carpeta model_artifacts/")
            return False

    def show_header(self):
        """Mostrar encabezado de la aplicación"""
        st.title("🚇 Sistema de Clasificación de Fallas en Metro")
        st.markdown("""
        Esta aplicación utiliza modelos de Machine Learning para predecir tipos de fallas 
        en sistemas de transporte público basándose en datos de sensores.
        """)
        
    def show_model_comparison(self):
        """Mostrar comparación de modelos entrenados"""
        st.header("📊 Modelos Entrenados")
        
        if not self.metadata.get('results'):
            st.warning("No hay información de modelos disponible")
            return
        
        # Crear tabla de comparación
        comparison_data = []
        for model_name, result in self.metadata['results'].items():
            comparison_data.append({
                'Modelo': model_name,
                'Accuracy': result['accuracy'],
                'F1-Score': result['f1_score'],
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        
        # Mostrar en dos columnas
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Métricas de Modelos")
            st.dataframe(df_comparison, use_container_width=True)
        
        with col2:
            st.subheader("Comparación Visual")
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                name='Accuracy',
                x=df_comparison['Modelo'],
                y=df_comparison['Accuracy'],
                marker_color='#1f77b4'
            ))
            
            fig.add_trace(go.Bar(
                name='F1-Score',
                x=df_comparison['Modelo'],
                y=df_comparison['F1-Score'],
                marker_color='#ff7f0e'
            ))
            
            fig.update_layout(
                title="Comparación de Modelos Entrenados",
                xaxis_title="Modelos",
                yaxis_title="Puntuación",
                barmode='group',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Mostrar mejor modelo
        best_model_name = max(self.metadata['results'].items(), key=lambda x: x[1]['f1_score'])[0]
        best_result = self.metadata['results'][best_model_name]
        
        st.success(f"🏆 **Mejor modelo**: {best_model_name} - F1-Score: {best_result['f1_score']:.4f}")

    def prediction_interface(self):
        """Interfaz de predicción en tiempo real"""
        st.header("🔮 Predicción en Tiempo Real")
        
        if not self.models:
            st.error("No hay modelos cargados para hacer predicciones")
            return
        
        st.subheader("Ingresa los valores de los sensores:")
        
        # Crear inputs basados en las características
        feature_inputs = {}
        
        # Organizar en columnas
        cols_per_row = 3
        features = self.feature_names[:12]  # Mostrar máximo 12 características
        
        for i, feature_name in enumerate(features):
            col_idx = i % cols_per_row
            if col_idx == 0:
                # Crear nuevas columnas cada 3 inputs
                cols = st.columns(cols_per_row)
            
            with cols[col_idx]:
                feature_inputs[feature_name] = st.number_input(
                    f"{feature_name}",
                    value=0.0,
                    step=0.1,
                    key=f"input_{i}"
                )
        
        # Completar características faltantes con valores por defecto
        for feature_name in self.feature_names[12:]:
            feature_inputs[feature_name] = 0.0
        
        # Selector de modelo
        col1, col2 = st.columns([1, 2])
        
        with col1:
            selected_model = st.selectbox(
                "Selecciona el modelo para predecir:",
                list(self.models.keys())
            )
        
        with col2:
            st.markdown("###")
            if st.button("🎯 Predecir Tipo de Falla", type="primary", use_container_width=True):
                self.make_prediction(feature_inputs, selected_model)

    def make_prediction(self, feature_inputs, model_name):
        """Realizar predicción con los inputs"""
        try:
            # Convertir a array en el orden correcto
            input_values = [feature_inputs[feature] for feature in self.feature_names]
            input_array = np.array(input_values).reshape(1, -1)
            
            # Preprocesar input
            input_processed = self.preprocessor['scaler'].transform(input_array)
            
            # Hacer predicción
            model = self.models[model_name]
            prediction = model.predict(input_processed)[0]
            probabilities = model.predict_proba(input_processed)[0]
            
            # Decodificar clase
            class_name = self.preprocessor['label_encoder'].inverse_transform([prediction])[0]
            confidence = np.max(probabilities)
            
            # Mostrar resultados
            st.success("### 📊 Resultados de la Predicción")
            
            results_col1, results_col2, results_col3 = st.columns(3)
            
            with results_col1:
                st.metric(
                    "🎯 Predicción", 
                    f"Clase {class_name}",
                    delta=f"Confianza: {confidence:.2%}"
                )
            
            with results_col2:
                st.metric("🤖 Modelo Usado", model_name)
            
            with results_col3:
                st.metric("📈 Clase", f"{prediction}")
            
            # Mostrar probabilidades por clase
            st.subheader("📋 Probabilidades por Clase")
            
            prob_df = pd.DataFrame({
                'Clase': [f'Clase {i}' for i in range(len(probabilities))],
                'Probabilidad': probabilities
            }).sort_values('Probabilidad', ascending=False)
            
            # Mostrar en dos columnas
            prob_col1, prob_col2 = st.columns([2, 1])
            
            with prob_col1:
                fig = px.bar(
                    prob_df.head(8),
                    x='Probabilidad',
                    y='Clase',
                    orientation='h',
                    title="Distribución de Probabilidades",
                    color='Probabilidad',
                    color_continuous_scale='reds'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with prob_col2:
                st.dataframe(
                    prob_df.style.format({'Probabilidad': '{:.2%}'}),
                    use_container_width=True
                )
            
        except Exception as e:
            st.error(f"❌ Error en la predicción: {str(e)}")

    def show_dataset_info(self):
        """Mostrar información del dataset"""
        st.header("📁 Información del Dataset")
        
        if not self.metadata.get('dataset_info'):
            return
        
        info = self.metadata['dataset_info']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Muestras", f"{info['n_samples']:,}")
        
        with col2:
            st.metric("Características", info['n_features'])
        
        with col3:
            st.metric("Variable Target", info['target_column'])
        
        with col4:
            st.metric("Modelos Entrenados", len(self.models))
        
        # Mostrar características
        with st.expander("📋 Lista de Características Usadas"):
            features_df = pd.DataFrame({
                'Característica': self.feature_names,
                'Índice': range(len(self.feature_names))
            })
            st.dataframe(features_df, use_container_width=True)

    def run(self):
        """Ejecutar la aplicación completa"""
        # Sidebar
        st.sidebar.title("⚙️ Configuración")
        
        if st.sidebar.button("🔄 Cargar Modelos", use_container_width=True):
            with st.spinner("Cargando modelos pre-entrenados..."):
                self.load_artifacts()
        
        st.sidebar.markdown("---")
        st.sidebar.info("""
        **Instrucciones:**
        1. Click en 'Cargar Modelos'
        2. Ve a la pestaña 'Predecir'
        3. Ingresa valores de sensores
        4. Click en 'Predecir'
        """)
        
        # Contenido principal
        self.show_header()
        
        # Tabs principales
        tab1, tab2, tab3 = st.tabs(["📊 Modelos", "🔮 Predecir", "📁 Info Dataset"])
        
        with tab1:
            self.show_model_comparison()
        
        with tab2:
            self.prediction_interface()
        
        with tab3:
            self.show_dataset_info()

# Ejecutar la aplicación
if __name__ == "__main__":
    app = MetroStreamlitApp()
    app.run()
