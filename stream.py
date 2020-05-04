import streamlit as st
import pandas as pd
import numpy as np
#import unidecode as uni
#import Util as utl
from sklearn.cross_validation import cross_val_score
from sklearn import metrics
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt





def nuevosDatos (p_modeloMatriz, superficie_total, jardin, terraza, ambientes, tipo, barrio):

                            
    modeloMatriz = p_modeloMatriz

    ##SUPERFICIE TOTAL
    df0 = pd.DataFrame({'superficie_total':pd.Series(superficie_total)})

    ##BARRIOS
    barrios = pd.Series(modeloMatriz.iloc[:,7:].columns)
    barrios = (barrios.str.replace('_',' '))
    df1 = barrios.apply(lambda x: 1 if x==barrio else 0)
    df2 = pd.DataFrame(columns=barrios)
    df2 = df2.append({ 'flores' : 0 } , ignore_index=True)
    df2 = df2.fillna(0).astype(int)
    df2.iloc[:,barrios[barrios.str.contains(barrio+'$',regex=True)].index] = '1'


    ##AMBIENTES Y TIPOS


    
    if jardin=='1':
        var_jardin = 'jardin'
    else:
        var_jardin = ''
    if terraza=='1':
        var_terraza = 'terraza'
    else:
        var_terraza = ''
    if (jardin == '1') & (terraza == '1'):
        var_jardinTerraza = 'jardinTerraza'
    else:
        var_jardinTerraza = ''

    
    df4 = pd.DataFrame({'jardin':pd.Series(0),'jardinTerraza':pd.Series(0),'CASA':pd.Series(0),'PH':pd.Series(0),'DTO':pd.Series(0)})
    indices = df4.columns
    indices = pd.Series(indices).astype(str)
    indices_bool = (indices.apply(lambda x: x=='CASA')) | (indices.apply(lambda x: x==var_jardin)) | (indices.apply(lambda x: x==var_terraza)) | (indices.apply(lambda x: x==var_jardinTerraza))  
    serie_df4 = indices_bool.apply(lambda x : 1 if x else 0)

    df4_proc = pd.DataFrame({

    'jardin':pd.Series(serie_df4[0]), 
    'jardinTerraza':pd.Series(serie_df4[1]),
    'CASA':pd.Series(serie_df4[2]),
    'PH':pd.Series(serie_df4[3]),
    'DTO':pd.Series(serie_df4[4])
    })

    
    predecir_data = pd.concat([df0,df4_proc],axis=1)
    predecir_data = pd.concat([predecir_data, df2],axis=1)
    #predecir_data.superficie_total_2 = predecir_data.superficie_total**2

    return predecir_data


@st.cache
def modelo_lasso_cross_validation(p_modeloMatriz):
    
    modeloMatriz = p_modeloMatriz

    xs = modeloMatriz.iloc[:,1:]
    y = modeloMatriz.iloc[:,0]
    xs = np.array(xs)  
    y = np.array(y) 
    lassocv = linear_model.LassoCV(alphas=np.linspace(0.01,100, 1000), cv=5, normalize=True)
    x_train, x_test, y_train, y_test = train_test_split(xs, y, test_size=0.4)
    lassocv.fit(x_train, y_train)
    alpha_lasso = lassocv.alpha_

    lasso = linear_model.Lasso(alpha=alpha_lasso, normalize=True)
    x_train, x_test, y_train, y_test = train_test_split(xs, y, test_size=0.4)
    lasso_model =lasso.fit(x_train, y_train)
    scores = cross_val_score(lasso_model, x_train, y_train, cv=5)
    y_predict = lasso_model.predict(x_test)

    #plt.scatter(x_test[:,0], y_test, color='blue')
    #plt.scatter(x_test[:,0], y_predict, color='red')

    #print('LASSO REGRESSION')
    #print('CROSS VALIDATION:', scores[0], scores[1], scores[2], scores[3],scores[4])
    #print ('MAE LASSO:', metrics.mean_absolute_error(y_test, y_predict))
    #print ('MSE LASSO:', metrics.mean_squared_error(y_test, y_predict))
    #print ('RMSE LASSO:', np.sqrt(metrics.mean_squared_error(y_test, y_predict)))
    #print ("LASSO -> R2 TRAIN: ", lasso_model.score(x_train, y_train))
    #print ("LASSO -> R2 TEST: ", lasso_model.score(x_test, y_test))

    return lasso_model




modeloMatriz= pd.read_csv('modeloMatriz.csv',sep='|')
modeloMatriz = modeloMatriz.iloc[:,1:]


st.write(
      '<h1 class="titulo">Calcular Precio M2</h1>',
      unsafe_allow_html=True
  )
#st.title('')

st.write(
      '<h3 class="titulo_secundario">en Capital Federal</h1>',
      unsafe_allow_html=True
  )


st.markdown('<style>h1.titulo{color:#9f9f9f;padding:0;margin-botton:0;}.titulo:hover{color:#ff5454;}</style>', unsafe_allow_html=True)

st.markdown('<style>h3.titulo_secundario{color:#9f9f9f;padding:0;margin-botton:0;}.titulo:hover{color:#ff5454;}</style>', unsafe_allow_html=True)




st.markdown('<style>.reportview-container .main .block-container{border-radius:5%;padding:5%;margin:5%;background:#fff8f8;text-align:center;}</style>', unsafe_allow_html=True)


st.markdown('<style>.block-container{text-align:center;}</style>', unsafe_allow_html=True)


st.markdown('<style>html{padding:5%;background:#eeeeee;}</style>', unsafe_allow_html=True)

st.write(
      '<h3 class="sup_total">Ingrese la Superficie Total...</h3>',
      unsafe_allow_html=True
  )

st.markdown('<style>h3.sup_total{margin:-2%;padding:0;} .sup_total{color:#9f9f9f;}.sup_total:hover{color:#ff5454;}</style>', unsafe_allow_html=True)


var_superficie = st.text_input('')

st.title('')

if st.checkbox('Jardin'):
	JARDIN = '1'   
else:
	JARDIN = '0'   


if st.checkbox('Terraza' ):
	TERRAZA = '1'
else:
	TERRAZA = '0'


st.markdown('<style>.st-bp.st-c3.st-ai.st-ae.st-af.st-ag.st-c4{color:#9f9f9f;}.st-ai{color:#9f9f9f}</style>', unsafe_allow_html=True)



st.markdown('<style>div.Widget.row-widget.stCheckbox{margin-left:45%;}</style>', unsafe_allow_html=True)



#.st-bb {
#    -webkit-box-align: start;
#    align-items: flex-start;
#}

df = pd.DataFrame({
  'Propiedad': ['Casa', 'Departamento', 'PH'],
  'second column': [10, 20, 30]
})


diccionar_tipos = {'Casa':'CASA','Departamento':'DTO','PH':'PH'}


st.write(
      '<h3 class="tipo_propiedad">Seleccione el Tipo de Propiedad...</h3>',
      unsafe_allow_html=True
  )


st.markdown('<style>h3.tipo_propiedad{margin:-3%;padding:0;} .tipo_propiedad{color:#9f9f9f}.tipo_propiedad:hover{color:#ff5454;}</style>', unsafe_allow_html=True)



var_tipo = st.selectbox(
    '',
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



st.write(
      '<h3 class="barrio">Seleccione el barrio...</h3>',
      unsafe_allow_html=True
  )




st.markdown('<style>.barrio{color:#9f9f9f;}.barrio:hover{color:#ff5454;}</style>', unsafe_allow_html=True)


st.markdown('<style>h3.barrio{padding-top:5%;padding-botton:0;margin:-5%}</style>', unsafe_allow_html=True)


var_barrio = st.selectbox(
    '',
     df['barrios'])


#st.title(var_superficie)
#st.title(diccionar_tipos[var_tipo])
#st.title(var_barrio)



SUPERFICIE_TOTAL = var_superficie           
CANTIDAD_DE_AMBIENTES = '1'       
TIPO_DE_PROPIEDAD = diccionar_tipos[var_tipo]
BARRIO = var_barrio


st.title('')

 


if st.button('Predecir Precio'):
  if SUPERFICIE_TOTAL.isnumeric():
    modelo = modelo_lasso_cross_validation(modeloMatriz)
    nuevos_Feactures = nuevosDatos(modeloMatriz, SUPERFICIE_TOTAL, JARDIN, TERRAZA, CANTIDAD_DE_AMBIENTES, TIPO_DE_PROPIEDAD, BARRIO)
    y_predict = modelo.predict(nuevos_Feactures)
    st.title('El precio por M2 es de U$D'+str(y_predict[0].round(-1).astype(int)))
  else:
    st.title('Debe Ingresar un valor correcto de Superficie Total')


st.markdown('<style>.st-dd{background:#ff5454;}button.st-ae{color:white;}</style>', unsafe_allow_html=True)


st.markdown('<style>html{background: #fffff}</style>', unsafe_allow_html=True)





st.title('')









