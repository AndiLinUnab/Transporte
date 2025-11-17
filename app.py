# app.py (VERSIÓN CON MÚLTIPLES MÉTODOS DE CARGA)
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# Configurar la página
st.set_page_config(
    page_title="Clasificador de Fallas - Metro",
    page_icon="🚇",
    layout="wide"
)

class MetroStreamlitApp:
    def __init__(self):
        self.models = {}
        self.preprocessor = None
        self.feature_names = []
        self.metadata = {}
        
    def try_multiple_load_methods(self, filepath):
        """Intentar múltiples métodos para cargar archivos .pkl"""
        methods = [
            self.load_with_joblib,
            self.load_with_pickle,
            self.load_with_cloudpickle
        ]
        
        for method in methods:
            try:
                result = method(filepath)
                if result is not None:
                    st.sidebar.success(f"✅ Cargado con {method.__name__}")
                    return result
            except Exception as e:
                st.sidebar.warning(f"⚠️ {method.__name__} falló: {str(e)[:30]}")
        
        return None

    def load_with_joblib(self, filepath):
        """Cargar con joblib (método principal)"""
        import joblib
        return joblib.load(filepath)

    def load_with_pickle(self, filepath):
        """Cargar con pickle nativo"""
        import pickle
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def load_with_cloudpickle(self, filepath):
        """Cargar con cloudpickle (más compatible)"""
        import cloudpickle
        with open(filepath, 'rb') as f:
            return cloudpickle.load(f)

    def load_artifacts_robust(self):
        """Carga robusta con múltiples métodos"""
        st.sidebar.info("🔄 Carga robusta iniciada...")
        
        try:
            # 1. Cargar preprocesador
            st.sidebar.write("📦 Cargando preprocesador...")
            self.preprocessor = self.try_multiple_load_methods('model_artifacts/preprocessor.pkl')
            if self.preprocessor is None:
                st.error("❌ No se pudo cargar el preprocesador con ningún método")
                return False
            
            # Verificar estructura del preprocesador
            if not isinstance(self.preprocessor, dict):
                st.error(f"❌ Preprocesador no es un diccionario: {type(self.preprocessor)}")
                return False
            
            required_keys = ['scaler', 'imputer', 'label_encoder']
            missing_keys = [key for key in required_keys if key not in self.preprocessor]
            if missing_keys:
                st.error(f"❌ Faltan keys en preprocesador: {missing_keys}")
                st.info(f"Keys disponibles: {list(self.preprocessor.keys())}")
                return False
            
            st.sidebar.success("✅ Preprocesador válido")
            
            # 2. Cargar feature_names
            st.sidebar.write("📋 Cargando características...")
            self.feature_names = self.try_multiple_load_methods('model_artifacts/feature_names.pkl')
            if self.feature_names is None:
                st.error("❌ No se pudo cargar feature_names")
                return False
            
            if not isinstance(self.feature_names, list) or len(self.feature_names) == 0:
                st.error(f"❌ feature_names no es una lista válida: {type(self.feature_names)}")
                return False
            
            st.sidebar.success(f"✅ {len(self.feature_names)} características cargadas")
            
            # 3. Cargar metadatos
            st.sidebar.write("📊 Cargando metadatos...")
            self.metadata = self.try_multiple_load_methods('model_artifacts/metadata.pkl')
            if self.metadata is None:
                st.warning("⚠️ No se pudieron cargar metadatos, continuando sin ellos...")
                self.metadata = {}  # Metadata opcional
            else:
                st.sidebar.success("✅ Metadatos cargados")
            
            # 4. Cargar modelos (intentar al menos uno)
            st.sidebar.write("🤖 Cargando modelos...")
            model_files = [
                'random_forest_model.pkl',
                'xgboost_model.pkl',
                'svm_model.pkl', 
                'knn_model.pkl',
                'logistic_regression_model.pkl'
            ]
            
            loaded_models = 0
            for model_file in model_files:
                model_path = f'model_artifacts/{model_file}'
                if os.path.exists(model_path):
                    model = self.try_multiple_load_methods(model_path)
                    if model is not None:
                        model_name = model_file.replace('_model.pkl', '').replace('_', ' ').title()
                        self.models[model_name] = model
                        loaded_models += 1
                        st.sidebar.success(f"✅ {model_name}")
                    else:
                        st.sidebar.warning(f"⚠️ No se pudo cargar {model_file}")
                else:
                    st.sidebar.warning(f"⚠️ {model_file} no existe")
            
            if loaded_models == 0:
                st.error("❌ No se pudo cargar ningún modelo")
                return False
            
            st.sidebar.success(f"🎉 {loaded_models} modelos cargados")
            
            # Guardar en session state
            st.session_state.models_loaded = True
            st.session_state.feature_names = self.feature_names
            st.session_state.preprocessor = self.preprocessor
            st.session_state.models = self.models
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error en carga robusta: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            return False

    def show_success_interface(self):
        """Mostrar interfaz cuando la carga es exitosa"""
        st.header("🎉 ¡Modelos Cargados Correctamente!")
        
        # Resumen de lo cargado
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Características", len(self.feature_names))
        
        with col2:
            st.metric("Modelos", len(self.models))
        
        with col3:
            st.metric("Preprocesador", "✅")
        
        with col4:
            st.metric("Metadatos", "✅" if self.metadata else "⚠️")
        
        # Mostrar características
        with st.expander("📋 Ver características cargadas"):
            st.write(f"Total: {len(self.feature_names)} características")
            for i, feature in enumerate(self.feature_names[:10]):  # Mostrar primeras 10
                st.write(f"{i+1}. {feature}")
            if len(self.feature_names) > 10:
                st.write(f"... y {len(self.feature_names) - 10} más")
        
        # Mostrar modelos cargados
        with st.expander("🤖 Ver modelos cargados"):
            for model_name in self.models.keys():
                st.success(f"✅ {model_name}")
        
        # Interfaz de predicción
        st.header("🔮 Haz una Predicción")
        
        st.info("Ingresa valores para las características y haz una predicción:")
        
        # Crear inputs dinámicos
        feature_inputs = {}
        features_to_show = self.feature_names[:8]  # Mostrar primeras 8 para no saturar
        
        cols = st.columns(2)
        for i, feature_name in enumerate(features_to_show):
            col_idx = i % 2
            with cols[col_idx]:
                feature_inputs[feature_name] = st.number_input(
                    f"{feature_name}",
                    value=0.0,
                    step=0.1,
                    key=f"pred_{i}"
                )
        
        # Valores por defecto para características no mostradas
        for feature_name in self.feature_names[8:]:
            feature_inputs[feature_name] = 0.0
        
        # Selector de modelo
        selected_model = st.selectbox(
            "Selecciona el modelo:",
            list(self.models.keys())
        )
        
        if st.button("🎯 Predecir", type="primary"):
            self.make_prediction(feature_inputs, selected_model)

    def make_prediction(self, feature_inputs, model_name):
        """Hacer predicción con los inputs"""
        try:
            # Preparar datos de entrada
            input_values = [feature_inputs[feature] for feature in self.feature_names]
            input_array = np.array(input_values).reshape(1, -1)
            
            # Preprocesar
            input_processed = self.preprocessor['scaler'].transform(input_array)
            
            # Predecir
            model = self.models[model_name]
            prediction = model.predict(input_processed)[0]
            probabilities = model.predict_proba(input_processed)[0]
            
            # Decodificar clase
            class_name = self.preprocessor['label_encoder'].inverse_transform([prediction])[0]
            confidence = np.max(probabilities)
            
            # Mostrar resultados
            st.success("### 📊 Resultado de la Predicción")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "🎯 Predicción", 
                    f"Clase {class_name}",
                    delta=f"{confidence:.2%} confianza"
                )
            
            with col2:
                st.metric("🤖 Modelo", model_name)
            
            with col3:
                st.metric("🔢 Clase", f"{prediction}")
            
            # Mostrar probabilidades
            st.subheader("📈 Probabilidades por Clase")
            
            prob_df = pd.DataFrame({
                'Clase': [f'Clase {i}' for i in range(len(probabilities))],
                'Probabilidad': probabilities
            }).sort_values('Probabilidad', ascending=False)
            
            # Gráfico de barras
            import plotly.express as px
            fig = px.bar(
                prob_df,
                x='Probabilidad',
                y='Clase',
                orientation='h',
                title="Distribución de Probabilidades",
                color='Probabilidad',
                color_continuous_scale='reds'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Tabla de probabilidades
            st.dataframe(
                prob_df.style.format({'Probabilidad': '{:.2%}'}),
                use_container_width=True
            )
            
        except Exception as e:
            st.error(f"❌ Error en predicción: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

    def show_debug_info(self):
        """Mostrar información de debug"""
        st.header("🐛 Información de Debug")
        
        st.subheader("Contenido de los archivos:")
        
        # Preprocesador
        if self.preprocessor:
            st.success("✅ Preprocesador cargado")
            st.write("Keys:", list(self.preprocessor.keys()))
            for key, value in self.preprocessor.items():
                st.write(f"  {key}: {type(value)}")
        else:
            st.error("❌ Preprocesador NO cargado")
        
        # Feature names
        if self.feature_names:
            st.success(f"✅ Feature names: {len(self.feature_names)} elementos")
            st.write("Ejemplo:", self.feature_names[:3])
        else:
            st.error("❌ Feature names NO cargados")
        
        # Models
        if self.models:
            st.success(f"✅ Modelos: {len(self.models)} cargados")
            for name, model in self.models.items():
                st.write(f"  {name}: {type(model)}")
        else:
            st.error("❌ Modelos NO cargados")

    def run(self):
        """Ejecutar aplicación"""
        st.sidebar.title("⚙️ Configuración")
        
        # Botón de carga robusta
        if st.sidebar.button("🔄 Carga Robusta de Modelos", type="primary", use_container_width=True):
            with st.spinner("Cargando con múltiples métodos..."):
                success = self.load_artifacts_robust()
                if success:
                    st.sidebar.success("✅ ¡Carga exitosa!")
                    st.session_state.load_success = True
                else:
                    st.sidebar.error("❌ Falló la carga")
                    st.session_state.load_success = False
        
        st.sidebar.markdown("---")
        st.sidebar.info("""
        **Este método prueba:**
        - joblib (principal)
        - pickle nativo  
        - cloudpickle (backup)
        """)
        
        # Contenido principal
        st.title("🚇 Clasificador - CARGA ROBUSTA")
        
        # Verificar si la carga fue exitosa
        if hasattr(st.session_state, 'load_success') and st.session_state.load_success:
            self.show_success_interface()
        else:
            st.warning("👆 Haz click en 'Carga Robusta de Modelos' para cargar los modelos")
            
            # Mostrar debug si ya se intentó cargar
            if hasattr(st.session_state, 'load_success') and not st.session_state.load_success:
                self.show_debug_info()

if __name__ == "__main__":
    # Instalar cloudpickle si no está disponible
    try:
        import cloudpickle
    except ImportError:
        st.warning("📦 Instalando cloudpickle para mejor compatibilidad...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "cloudpickle"])
        import cloudpickle
    
    app = MetroStreamlitApp()
    app.run()
