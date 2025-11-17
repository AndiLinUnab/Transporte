# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys

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
        
    def check_files_exist(self):
        """Verificar que los archivos necesarios existen"""
        st.sidebar.subheader("🔍 Verificación de Archivos")
        
        required_files = [
            'model_artifacts/preprocessor.pkl',
            'model_artifacts/feature_names.pkl', 
            'model_artifacts/metadata.pkl',
            'model_artifacts/random_forest_model.pkl',
            'model_artifacts/xgboost_model.pkl',
            'model_artifacts/svm_model.pkl',
            'model_artifacts/knn_model.pkl',
            'model_artifacts/logistic_regression_model.pkl'
        ]
        
        missing_files = []
        for file_path in required_files:
            if os.path.exists(file_path):
                st.sidebar.success(f"✅ {file_path}")
            else:
                st.sidebar.error(f"❌ {file_path}")
                missing_files.append(file_path)
        
        return len(missing_files) == 0

    def load_artifacts(self):
        """Cargar modelos y preprocesadores con mejor manejo de errores"""
        try:
            st.sidebar.info("🔄 Iniciando carga de modelos...")
            
            # Verificar que los archivos existen primero
            if not self.check_files_exist():
                st.error("❌ Faltan archivos necesarios. Verifica la estructura.")
                return False
            
            # 1. Cargar preprocesador
            st.sidebar.info("📦 Cargando preprocesador...")
            self.preprocessor = joblib.load('model_artifacts/preprocessor.pkl')
            st.sidebar.success("✅ Preprocesador cargado")
            
            # 2. Cargar nombres de características
            st.sidebar.info("📋 Cargando nombres de características...")
            self.feature_names = joblib.load('model_artifacts/feature_names.pkl')
            st.sidebar.success(f"✅ {len(self.feature_names)} características cargadas")
            
            # 3. Cargar metadatos
            st.sidebar.info("📊 Cargando metadatos...")
            self.metadata = joblib.load('model_artifacts/metadata.pkl')
            st.sidebar.success("✅ Metadatos cargados")
            
            # 4. Cargar modelos individuales
            st.sidebar.info("🤖 Cargando modelos de ML...")
            model_files = {
                'Random Forest': 'random_forest_model.pkl',
                'XGBoost': 'xgboost_model.pkl', 
                'SVM': 'svm_model.pkl',
                'KNN': 'knn_model.pkl',
                'Logistic Regression': 'logistic_regression_model.pkl'
            }
            
            successful_models = 0
            for model_name, filename in model_files.items():
                try:
                    model_path = f'model_artifacts/{filename}'
                    self.models[model_name] = joblib.load(model_path)
                    successful_models += 1
                    st.sidebar.success(f"✅ {model_name}")
                except Exception as e:
                    st.sidebar.warning(f"⚠️ {model_name}: {str(e)}")
            
            st.sidebar.success(f"🎉 {successful_models}/5 modelos cargados exitosamente!")
            
            # Guardar estado en session state
            st.session_state.models_loaded = True
            st.session_state.feature_names = self.feature_names
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error crítico cargando artefactos: {str(e)}")
            st.info("""
            **Posibles soluciones:**
            1. Verifica que la carpeta `model_artifacts/` esté en la raíz del repositorio
            2. Asegúrate de que todos los archivos .pkl existan
            3. Los modelos deben ser entrenados en la misma versión de scikit-learn
            """)
            return False

    def show_model_comparison(self):
        """Mostrar comparación de modelos entrenados"""
        st.header("📊 Modelos Entrenados")
        
        if not self.metadata.get('results'):
            st.warning("""
            **No hay información de modelos disponible**
            
            Esto significa que:
            - Los modelos no se cargaron correctamente, O
            - El archivo `metadata.pkl` no contiene los resultados del entrenamiento
            
            **Solución:** Haz click en 'Cargar Modelos' en el sidebar y verifica que no haya errores.
            """)
            return
        
        # Crear tabla de comparación
        comparison_data = []
        for model_name, result in self.metadata['results'].items():
            comparison_data.append({
                'Modelo': model_name,
                'Accuracy': f"{result['accuracy']:.4f}",
                'F1-Score': f"{result['f1_score']:.4f}",
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        
        # Mostrar en dos columnas
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Métricas de Modelos")
            st.dataframe(df_comparison, use_container_width=True)
        
        with col2:
            st.subheader("Comparación Visual")
            
            # Convertir a float para plotting
            df_comparison['Accuracy_num'] = df_comparison['Accuracy'].astype(float)
            df_comparison['F1-Score_num'] = df_comparison['F1-Score'].astype(float)
            
            import plotly.graph_objects as go
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                name='Accuracy',
                x=df_comparison['Modelo'],
                y=df_comparison['Accuracy_num'],
                marker_color='#1f77b4'
            ))
            
            fig.add_trace(go.Bar(
                name='F1-Score',
                x=df_comparison['Modelo'],
                y=df_comparison['F1-Score_num'],
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
        
        st.success(f"🏆 **Mejor modelo**: {best_model_name} - Accuracy: {best_result['accuracy']:.4f}, F1-Score: {best_result['f1_score']:.4f}")

    def prediction_interface(self):
        """Interfaz de predicción en tiempo real"""
        st.header("🔮 Predicción en Tiempo Real")
        
        # Verificar si los modelos están cargados
        if not hasattr(st.session_state, 'models_loaded') or not st.session_state.models_loaded:
            st.error("""
            **❌ Los modelos no están cargados**
            
            Para usar la predicción:
            1. Haz click en **'Cargar Modelos'** en el sidebar
            2. Espera a que todos los modelos se carguen (deben aparecer checkmarks verdes)
            3. Vuelve a esta pestaña
            """)
            return
        
        if not self.models:
            st.error("No hay modelos disponibles para hacer predicciones")
            return
        
        st.success("✅ Modelos cargados correctamente. Puedes hacer predicciones.")
        
        st.subheader("Ingresa los valores de los sensores:")
        
        # Usar feature_names de session state
        feature_names = st.session_state.get('feature_names', [f'Feature_{i}' for i in range(10)])
        
        # Crear inputs basados en las características
        feature_inputs = {}
        
        # Organizar en columnas
        features_to_show = feature_names[:12]  # Mostrar máximo 12 características
        cols_per_row = 3
        
        for i, feature_name in enumerate(features_to_show):
            col_idx = i % cols_per_row
            if col_idx == 0:
                cols = st.columns(cols_per_row)
            
            with cols[col_idx]:
                feature_inputs[feature_name] = st.number_input(
                    f"{feature_name}",
                    value=0.0,
                    step=0.1,
                    key=f"input_{i}"
                )
        
        # Completar características faltantes con valores por defecto
        for feature_name in feature_names[12:]:
            feature_inputs[feature_name] = 0.0
        
        # Selector de modelo y botón de predicción
        col1, col2 = st.columns([1, 2])
        
        with col1:
            selected_model = st.selectbox(
                "Selecciona el modelo para predecir:",
                list(self.models.keys())
            )
        
        with col2:
            st.markdown("###")
            if st.button("🎯 Predecir Tipo de Falla", type="primary", use_container_width=True):
                self.make_prediction(feature_inputs, selected_model, feature_names)

    def make_prediction(self, feature_inputs, model_name, feature_names):
        """Realizar predicción con los inputs"""
        try:
            # Convertir a array en el orden correcto
            input_values = [feature_inputs.get(feature, 0.0) for feature in feature_names]
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
                st.metric("📈 Clase Numérica", f"{prediction}")
            
            # Mostrar probabilidades por clase
            st.subheader("📋 Probabilidades por Clase")
            
            prob_df = pd.DataFrame({
                'Clase': [f'Clase {i}' for i in range(len(probabilities))],
                'Probabilidad': probabilities
            }).sort_values('Probabilidad', ascending=False)
            
            # Mostrar en dos columnas
            prob_col1, prob_col2 = st.columns([2, 1])
            
            with prob_col1:
                import plotly.express as px
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
            st.info("""
            **Posibles causas:**
            - Los modelos no se cargaron correctamente
            - El preprocesador no coincide con los modelos
            - Error en la transformación de datos
            """)

    def show_dataset_info(self):
        """Mostrar información del dataset"""
        st.header("📁 Información del Dataset")
        
        if not self.metadata.get('dataset_info'):
            st.info("Carga los modelos para ver la información del dataset")
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
        # Inicializar session state
        if 'models_loaded' not in st.session_state:
            st.session_state.models_loaded = False
        
        # Sidebar
        st.sidebar.title("⚙️ Configuración")
        
        if st.sidebar.button("🔄 Cargar Modelos", use_container_width=True):
            with st.spinner("Cargando modelos pre-entrenados..."):
                success = self.load_artifacts()
                if success:
                    st.sidebar.success("✅ Modelos cargados correctamente")
                else:
                    st.sidebar.error("❌ Error cargando modelos")
        
        st.sidebar.markdown("---")
        st.sidebar.info("""
        **Instrucciones:**
        1. Click en 'Cargar Modelos'
        2. **Verifica** que aparezcan checkmarks verdes
        3. Ve a la pestaña 'Predecir'
        4. Ingresa valores de sensores
        5. Click en 'Predecir'
        """)
        
        # Contenido principal
        st.title("🚇 Sistema de Clasificación de Fallas en Metro")
        st.markdown("Esta aplicación utiliza modelos de ML para predecir fallas basándose en datos de sensores.")
        
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
