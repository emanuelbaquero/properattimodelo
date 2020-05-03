import streamlit as st
import pandas as pd
import numpy as np
import Util as utl
from sklearn.cross_validation import cross_val_score
from sklearn import metrics
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

modeloMatriz= pd.read_csv('modeloMatriz.csv',sep='|')
modeloMatriz = modeloMatriz.iloc[:,1:]




st.title('Calcular Precio M2 (CABA)')
st.title('')

var_superficie = st.text_input("Ingrese la superficie", 0)



if st.checkbox('Jardin'):
	JARDIN = '1'   
else:
	JARDIN = '0'   


if st.checkbox('Terraza'):
	TERRAZA = '1'
else:
	TERRAZA = '0'


df = pd.DataFrame({
  'Propiedad': ['Casa', 'Departamento', 'PH'],
  'second column': [10, 20, 30]
})


diccionar_tipos = {'Casa':'CASA','Departamento':'DTO','PH':'PH'}

var_tipo = st.selectbox(
    '¿Tipo de Propiedad?',
     df['Propiedad'])


#st.title(diccionar_tipos[var_tipo])


df = pd.DataFrame({
  'barrios': ['mataderos', 'liniers', 'belgrano', 'palermo soho', 'palermo',
       'flores', 'boedo', 'las canitas', 'puerto madero', 'balvanera',
       'caballito', 'nunez', 'san telmo', 'almagro', 'capital federal',
       'colegiales', 'barrio norte', 'barracas', 'recoleta',
       'villa crespo', 'constitucion', 'villa urquiza',
       'palermo hollywood', 'saavedra', 'pompeya', 'parque chas',
       'paternal', 'agronomia', 'villa pueyrredon', 'coghlan',
       'parque centenario', 'monserrat', 'palermo chico', 'floresta',
       'villa luro', 'villa devoto', 'boca', 'parque avellaneda',
       'san cristobal', 'velez sarsfield', 'abasto', 'versalles',
       'villa del parque', 'monte castro', 'retiro', 'parque patricios',
       'san nicolas', 'villa santa rita', 'chacarita', 'congreso',
       'centro / microcentro', 'once', 'tribunales', 'parque chacabuco',
       'catalinas', 'villa general mitre', 'palermo viejo',
       'villa lugano', 'villa ortuzar', 'villa soldati', 'villa real',
       'villa riachuelo']
})


diccionar_tipos = {'Casa':'CASA','Departamento':'DTO','PH':'PH'}

var_barrio = st.selectbox(
    'Elegi el barrio...',
     df['barrios'])


#st.title(var_superficie)
#st.title(diccionar_tipos[var_tipo])
#st.title(var_barrio)



SUPERFICIE_TOTAL = var_superficie           
CANTIDAD_DE_AMBIENTES = '1'       
TIPO_DE_PROPIEDAD = diccionar_tipos[var_tipo]
BARRIO = var_barrio



if st.button('Predecir Precio'):
	modelo = utl.modelo_lasso_cross_validation(modeloMatriz)
	nuevos_Feactures = utl.nuevosDatos(modeloMatriz, SUPERFICIE_TOTAL, JARDIN, TERRAZA, CANTIDAD_DE_AMBIENTES, TIPO_DE_PROPIEDAD, BARRIO)
	y_predict = modelo.predict(nuevos_Feactures)
	st.title('El precio por M2 es de $'+str(y_predict[0].round(-1).astype(int)))













