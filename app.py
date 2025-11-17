# app.py (VERSIÓN CON DIAGNÓSTICO)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

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
        
    def diagnostic(self):
        """Función de diagnóstico para ver qué archivos existen"""
        st.sidebar.subheader("🔍 Diagnóstico de Archivos")
        
        # Verificar estructura de archivos
        base_path = "."
        model_artifacts_path = "./model_artifacts"
        
        st.sidebar.write("**Estructura del repositorio:**")
        
        if os.path.exists(model_artifacts_path):
            st.sidebar.success("✅ Carpeta model_artifacts/ existe")
            
            # Listar archivos en model_artifacts
            files = os.listdir(model_artifacts_path)
            st.sidebar.write(f"**Archivos encontrados ({len(files)}):**")
            for file in files:
                file_path = os.path.join(model_artifacts_path, file)
                file_size = os.path.getsize(file_path) if os.path.isfile(file_path) else 0
                st.sidebar.write(f"   📄 {file} ({file_size} bytes)")
                
            # Verificar archivos esenciales
            essential_files = [
                'preprocessor.pkl',
                'feature_names.pkl', 
                'metadata.pkl',
                'random_forest_model.pkl'
            ]
            
            st.sidebar.write("**Archivos esenciales:**")
            missing_essential = []
            for file in essential_files:
                if file in files:
                    st.sidebar.success(f"   ✅ {file}")
                else:
                    st.sidebar.error(f"   ❌ {file}")
                    missing_essential.append(file)
            
            return len(missing_essential) == 0
        else:
            st.sidebar.error("❌ Carpeta model_artifacts/ NO existe")
            return False

    def load_artifacts(self):
        """Cargar modelos y preprocesadores"""
        try:
            # Primero hacer diagnóstico
            if not self.diagnostic():
                st.error("❌ Faltan archivos esenciales. No se pueden cargar los modelos.")
                return False
            
            st.sidebar.info("🔄 Cargando modelos...")
            
            # 1. Cargar preprocesador
            self.preprocessor = joblib.load('model_artifacts/preprocessor.pkl')
            st.sidebar.success("✅ Preprocesador cargado")
            
            # 2. Cargar nombres de características
            self.feature_names = joblib.load('model_artifacts/feature_names.pkl')
            st.sidebar.success(f"✅ {len(self.feature_names)} características")
            
            # 3. Cargar metadatos
            self.metadata = joblib.load('model_artifacts/metadata.pkl')
            st.sidebar.success("✅ Metadatos cargados")
            
            # 4. Cargar modelos
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
                    self.models[model_name] = joblib.load(f'model_artifacts/{filename}')
                    successful_models += 1
                    st.sidebar.success(f"✅ {model_name}")
                except Exception as e:
                    st.sidebar.warning(f"⚠️ {model_name}: {str(e)[:50]}...")
            
            st.sidebar.success(f"🎉 {successful_models}/5 modelos cargados")
            
            # Guardar estado
            st.session_state.models_loaded = True
            return True
            
        except Exception as e:
            st.error(f"❌ Error cargando modelos: {str(e)}")
            return False

    def show_diagnostic_info(self):
        """Mostrar información de diagnóstico"""
        st.header("🔍 Información de Diagnóstico")
        
        # Verificar si los modelos están cargados
        if hasattr(st.session_state, 'models_loaded') and st.session_state.models_loaded:
            st.success("✅ Modelos cargados en session_state")
        else:
            st.warning("⚠️ Modelos NO cargados en session_state")
        
        # Mostrar información de metadatos si existe
        if self.metadata:
            st.subheader("📊 Metadatos Cargados")
            st.json(self.metadata)
        else:
            st.error("❌ No hay metadatos cargados")
        
        # Mostrar información de características
        if self.feature_names:
            st.subheader("📋 Características Cargadas")
            st.write(f"Número de características: {len(self.feature_names)}")
            st.write("Primeras 10 características:")
            st.write(self.feature_names[:10])
        else:
            st.error("❌ No hay características cargadas")

    def prediction_interface(self):
        """Interfaz de predicción"""
        st.header("🔮 Predicción en Tiempo Real")
        
        # Verificar estado
        if not hasattr(st.session_state, 'models_loaded') or not st.session_state.models_loaded:
            st.error("""
            **Los modelos no están cargados.**
            
            **Solución:**
            1. Haz click en **'Cargar Modelos'** en el sidebar
            2. **VERIFICA** que aparezcan checkmarks verdes en el diagnóstico
            3. Si hay errores, revisa que los archivos .pkl existan en model_artifacts/
            """)
            return
        
        if not self.models:
            st.error("No hay modelos disponibles")
            return
            
        st.success("✅ Listo para predecir!")
        
        # Inputs simples para prueba
        st.subheader("Ingresa valores de prueba:")
        value = st.slider("Valor de prueba", -10.0, 10.0, 0.0, key="test_slider")
        
        if st.button("🎯 Probar Predicción"):
            try:
                # Crear array de prueba
                input_array = np.array([value] * len(self.feature_names)).reshape(1, -1)
                
                # Preprocesar
                input_processed = self.preprocessor['scaler'].transform(input_array)
                
                # Predecir con primer modelo
                model_name = list(self.models.keys())[0]
                model = self.models[model_name]
                prediction = model.predict(input_processed)[0]
                probabilities = model.predict_proba(input_processed)[0]
                
                # Mostrar resultados
                class_name = self.preprocessor['label_encoder'].inverse_transform([prediction])[0]
                
                st.success(f"**Predicción exitosa!**")
                st.metric("Clase Predicha", f"Clase {class_name}")
                st.metric("Modelo Usado", model_name)
                
            except Exception as e:
                st.error(f"❌ Error en predicción: {str(e)}")

    def run(self):
        """Ejecutar aplicación"""
        # Sidebar
        st.sidebar.title("⚙️ Configuración")
        
        # Botón de diagnóstico siempre visible
        if st.sidebar.button("🔍 Ejecutar Diagnóstico", use_container_width=True):
            self.diagnostic()
        
        if st.sidebar.button("🔄 Cargar Modelos", use_container_width=True):
            with st.spinner("Cargando..."):
                self.load_artifacts()
        
        st.sidebar.markdown("---")
        st.sidebar.info("""
        **Si hay errores:**
        1. Verifica que model_artifacts/ esté en GitHub
        2. Los archivos .pkl deben estar en esa carpeta
        3. Ejecuta el diagnóstico primero
        """)
        
        # Contenido principal
        st.title("🚇 Clasificador de Fallas - MODO DIAGNÓSTICO")
        
        tab1, tab2 = st.tabs(["🔍 Diagnóstico", "🔮 Predecir"])
        
        with tab1:
            self.show_diagnostic_info()
        
        with tab2:
            self.prediction_interface()

if __name__ == "__main__":
    app = MetroStreamlitApp()
    app.run()
