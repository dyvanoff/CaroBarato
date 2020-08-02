# Importación de las librerías necesarias
import numpy as np
import pandas as pd
from io import StringIO
# Puede que nos sirvan también
import matplotlib as mpl
mpl.get_cachedir()
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import sys
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
#nltk.download('punkt')
import re
from collections import Counter

#if 'swifter' not in sys.modules:
#        !pip install -q swifter
        
import swifter
import chardet
import requests

pd.set_option('display.max_columns', 150)
pd.set_option('display.max_rows', 150)
pd.set_option('max_colwidth', 151)
pd.set_option('display.float_format', lambda x: '%.2f' % x)


def rename(newname):
    def decorator(f):
        f.__name__ = newname
        return f
    return decorator

#Definimos una función para pasar una o varias columnas minuscula
def column_lower(df, columns=[]):
    columns = list(columns)
    if len(columns) !=0:
        for i in range(len(columns)):
            if columns[i] in df.columns:
                df[columns[i]]=df[columns[i]].str.lower()
    else:
        print("Seleccione una columna o la columna no se encuentra en el DataFrame")

#Definimos una función para pasar sacar los acentos.
from unidecode import unidecode

def remove_accents(df, columns=[]):
    columns = list(columns)
    if len(columns) !=0:
        for i in range(len(columns)):
            if columns[i] in df.columns:
                df[columns[i]]=df[columns[i]].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')                
    else:
        print("Seleccione una columna o la columna no se encuentra en el DataFrame")

#Definimos una funcion para sobreescribir la um_depurada y cantidad_depurada en funcion de las reglas que definimos
#Procederemos a actualizar la um_depurada y la cantidad_depurada en función de las siguientes reglas que definimos:
#1- Cuando hay 1kg en nombre y 1 unidad en presentacion entonces dejamos 1 kg en la presentacion
#2- Cuando hay gr en mombre y unidad en presentacion entonces dejamos la cantidad de gramos por cantidad de unidades en presentacion
#3- Cuando hay mililitro en nombre y litro en presentacion entonces dejamos ml en presentacion. 
#Cabe destacar que existe un vino cuyo producto_id=7791540049415 que tiene 750 lt en la presentacion en vez de 750 ml
def update_var_depurada(row,data):
    for i in data.index:
        if (row[row.index==i].presentacion_en_nombre==True).any():
            if (((data[data.index==i].um_depurada=='kg') & (row[row.index==i].um_depurada=='un') & (row[row.index==i].cantidad_depurada=='1')).any() |
                ((data[data.index==i].um_depurada=='ml') & (row[row.index==i].um_depurada=='lt')).any()):
                row.loc[i,'um_depurada']=data.loc[i,'um_depurada']
                row.loc[i,'cantidad_depurada']=data.loc[i,'cantidad_depurada']
                if (i==59):
                    print('llego',data[data.index==i].um_depurada,data[data.index==i].cantidad_depurada)
            elif ((data[data.index==i].um_depurada=='gr') & (row[row.index==i].um_depurada=='un')).any():
                row.loc[i,'um_depurada']=data.loc[i,'um_depurada']
                row.loc[i,'cantidad_depurada']=pd.to_numeric(data.loc[i,'cantidad_depurada']) * pd.to_numeric(row.loc[i,'cantidad_depurada'])
    return row

# Realizamos la conversión de las UM a UM homogeneas.
# estandar= ('kg', 'un','lt', 'mt' ,'pack')
def homogenea (cantidad,unidad):
    if unidad in ('un', 'mt', 'kg', 'lt'):
        return round (1/cantidad, 3), unidad
    elif unidad in ('pack'):
        return 1, unidad
    elif unidad in ('gr'):
        return round (1000/cantidad, 3), 'kg' 
    elif unidad in ('cc', 'ml'):
        return round (1000/cantidad, 3), 'lt'

# Definimos una función para contar palabras
word_count = {}
def count_word(list_word):
    for item in list_word:
        if item in word_count:
            word_count[item] = word_count.get(item) + 1
    else:
        word_count[item] = 1      

#Creamos la función que crea dummies
def valor_dummie (valor):
    if valor == True:
        dummies = 1
    else:
        dummies = 0
    return dummies

def carga_precios():
	# Por un lado, cargamos los precios y los unimos en un único dataframe
	precios_20200412_20200413 = pd.read_csv('https://raw.githubusercontent.com/solujan/mentoria_2020/master/raw_dataset/precios_20200412_20200413.csv')
	precios_20200419_20200419 = pd.read_csv('https://raw.githubusercontent.com/solujan/mentoria_2020/master/raw_dataset/precios_20200419_20200419.csv')
	precios_20200426_20200426 = pd.read_csv('https://raw.githubusercontent.com/solujan/mentoria_2020/master/raw_dataset/precios_20200426_20200426.csv')
	precios_20200502_20200503 = pd.read_csv('https://raw.githubusercontent.com/solujan/mentoria_2020/master/raw_dataset/precios_20200502_20200503.csv')
	precios_20200518_20200518 = pd.read_csv('https://raw.githubusercontent.com/solujan/mentoria_2020/master/raw_dataset/precios_20200518_20200518.csv')

	lista_df_px = [precios_20200412_20200413, precios_20200419_20200419, precios_20200426_20200426,
	 precios_20200502_20200503, precios_20200518_20200518]
	fecha_px = ['20200412', '20200419', '20200426', '20200502', '20200518']

	fprecios = pd.DataFrame()
	for df, fecha in zip(lista_df_px, fecha_px):
	    df['fecha'] = fecha
	    fprecios = pd.concat([fprecios,df])

	return fprecios
 
def carga_productos():
	# Cargamos los txt de productos para tener la información de cada campo
	producto_url = 'https://raw.githubusercontent.com/solujan/mentoria_2020/master/raw_dataset/productos.csv'
	fproductos = pd.read_csv(producto_url)

	fproductos.set_index('id')
	fproductos[fproductos.index.duplicated()]

	#se saca la marca del nombre
	fproductos["nombre"] = fproductos.apply(lambda x: replace_substring(x), axis=1)

	#Convertimos a minusculas las letras que estan en presentacion y generamos una nueva columna presentacion_depurada
	fproductos['presentacion_depurada'] = fproductos['presentacion'].str.lower()

	#Generamos la columna um_depurada con las unidades de medidas presentes en presentacion_depurada
	fproductos['um_depurada'] = fproductos['presentacion_depurada'].str[-2:]

	#Generamos una nueva columna cantidad_depurada con las cantidades presentes en la presentacion_depurada
	fproductos['cantidad_depurada'] = fproductos['presentacion_depurada'].str[0:-3]

	#Creamos una nueva columna de nombre llamada nombre_depurado para trabajar sobre su depuracion
	fproductos['nombre_depurado'] = fproductos['nombre']
	
	#Pasamos marca y nombre_depurado a minusculas y le removemos los acentos
	column_lower(fproductos,['marca','nombre_depurado'])
	remove_accents(fproductos,['marca','nombre_depurado'])	

	#Verificamos la presencia de alguna de las unidades de medidas en el nombre del producto
	search_um='\\b'+'\\b|\\b'.join(map(str, fproductos['um_depurada'].unique())) + '\\b'
	fproductos['presentacion_en_nombre']=fproductos['nombre_depurado'].str.contains(search_um, na=False)	

	#Obtenemos aquellos productos cuya unidad de medida en el nombre no coincide con la unidad de medida de la presentación y la unidad de medida del nombre es una unidad de presentación válida
	aux_df=fproductos[(fproductos.presentacion_en_nombre==True) & (fproductos.nombre_depurado.str[-2:] != fproductos.um_depurada) & (fproductos.nombre_depurado.str[-2:].str.contains(search_um, na=False))]

	#Generamos un dataset df_last_words especificamente con las unidades de medidas que tiene el nombre
	df_last_words=aux_df.nombre_depurado.str.extract('(\w+) (\w+)$',expand=False)
	df_last_words.columns=['cantidad_depurada','um_depurada']

	#Actualizo las um_depurada y cantidad_depurada de acuerdo a la reglas que definimos
	update_var_depurada(fproductos,df_last_words)	

	#Asginamos la unidad de medida pack a las promociones y a los que en el nombre tiene las palabras pack, Paga o Lleva
	fproductos.loc[fproductos['nombre'].str.contains('\\b3x2\\b|\\b2x1\\b|\\b4x3\\b|\\bPaga\\b|\\bLleva\\b|\\bpack\\b', na=False),'um_depurada']='pack'

	# Creamos dos nuevas columnas denominadas um_homogenea y factor_homogenea
	fproductos[['factor_homogenea','um_homogenea']] = fproductos.swifter.apply(lambda x: homogenea(float(x['cantidad_depurada']),x['um_depurada']),axis=1, result_type='expand')

	# Creamos variables Dummies que identifiquen el tipo de Unidad de Medida
	fproductos=pd.get_dummies(fproductos, columns=['um_homogenea'])

	# Le quitamos las cantidades, unidad de medida, caracteres especiales y preposiciones
	delete_words='\\d|\\bde\\b|\\ben\\b|\\bla\\b|\\bque\\b|\\b\\de\\b\\|\\b\\a\\b\\|\\b\\sobre\\b\\|\\b\\sin\\b\\|\\by\\b|\\bun\\b|\\bkg\\b|\\bgr\\b|\\bml\\b|\\bcc\\b|\\blt\\b|\\bmt\\b|\\bante\\b|\\bbajo\\b|\\bcabe\\b|\\bcon\\b|\\bcontra\\b|\\bdesde\\b|\\bdurante\\b|\\bentre\\b|\\bhacia\\b|\\bhasta\\b|\\bmediante\\b|\\bpara\\b|\\bsegún\\b|\\bsegun\\b|\\bsi\\b|\\bso\\b|\\bsobre\\b|\\btras\\b|\\bversus\\b|\\bvía\\b|\\bvia\\b|[^\w ]'
	fproductos['nombre_depurado']=fproductos.nombre_depurado.str.replace(repl='',pat=(delete_words))	

	word_count = {}
	fproductos['nombre_depurado'].apply(word_tokenize).apply(lambda x: count_word(x))

	df=pd.DataFrame(data=[word_count.keys(), word_count.values()], )
	df=df.T
	df.columns=['token', 'count']
	df=df.sort_values('count', ascending=False).reset_index(drop=True)

	df['cum_sum']=df['count'].cumsum()
	indice=[]
	for i in range(len(df)):
	    if df.loc[i,'cum_sum']<df['count'].sum()*0.2:
	        pass
	    else:
	        indice=i; break	

	lista_frecuente = df.token[0:indice]
	lista_dummies=[None]*len(lista_frecuente)
	for i in range(len(lista_frecuente)):
    	lista_dummies[i]='dummy_'+ lista_frecuente[i];

	# Creamos las dummies para todos los productos frecuentes.
	for i in lista_frecuente:
	    fproductos['dummy_'+i]=fproductos['nombre_depurado'].str.contains('\\b'+ i + '\\b',regex=True).apply(valor_dummie)    	

	fproductos['palabras_nombre']=fproductos['nombre_depurado'].apply(word_tokenize).apply(len)

	fproductos['total_dummies']=fproductos.filter(regex='dummy',axis=1).iloc[:, -(colum+1):-2].sum(axis=1)

	fproductos['otras_palabras']=np.where(fproductos['palabras_nombre'] > fproductos['total_dummies'], 1, 0)	

	#Generamos un diccionario con las 30 marcas con mas frecuencia 
	data = fproductos.marca.value_counts()
	dict_marcas_frecuentas = dict(data[:30])
	dict_marcas_frecuentas	

	#Genero la columna de marcas mas frecuentes 
	fproductos['marcas_frecuentes'] = fproductos.marca.apply(lambda x: x if x in dict_marcas_frecuentas.keys() else 'otras')

	#Genero las variables dummy y las joineo con la de productos por index
	dummy = pd.get_dummies(fproductos.marcas_frecuentes)
	fproductos = fproductos.join(dummy)

	#Correjimos las columnas 
	fproductos.columns = fproductos.columns.str.replace(' ','_',regex=True)

	return fproductos

def carga_sucursales():
	# Cargamos los txt de sucursales para tener la información de cada campo
	sucursal_url='https://raw.githubusercontent.com/solujan/mentoria_2020/master/raw_dataset/sucursales.csv'
	fsucursales = pd.read_csv(sucursal_url)

	# A las sucursales, le agregamos la descripción de la provincia y la región
	provincia_txt = """
	provincia	nom_provincia	region
	AR-A	Salta	Norte Grande
	AR-B	Provincia de Buenos Aires	Centro
	AR-C	Ciudad Autónoma de Buenos Aires	Centro
	AR-D	San Luis	Cuyo
	AR-E	Entre Ríos	Centro
	AR-F	La Rioja	Cuyo
	AR-G	Santiago del Estero	Norte Grande
	AR-H	Chaco	Norte Grande
	AR-J	San Juan	Cuyo
	AR-K	Catamarca	Norte Grande
	AR-L	La Pampa	Centro
	AR-M	Mendoza	Cuyo
	AR-N	Misiones	Norte Grande
	AR-P	Formosa	Norte Grande
	AR-Q	Neuquén	Patagonia
	AR-R	Río Negro	Patagonia
	AR-S	Santa Fe	Centro
	AR-T	Tucumán	Norte Grande
	AR-U	Chubut	Patagonia
	AR-V	Tierra del Fuego	Patagonia
	AR-W	Corrientes	Norte Grande
	AR-X	Córdoba	Centro
	AR-Y	Jujuy	Norte Grande
	AR-Z	Santa Cruz	Patagonia
	"""
	
	provincia_csv = StringIO(provincia_txt)
	entidad_provincia = pd.read_csv(provincia_csv, sep=('\t'))
	fsucursales = fsucursales.merge(entidad_provincia, on = 'provincia')

	fsucursales.set_index('id')
	fsucursales[fsucursales.index.duplicated()]

	# Se agregan variables dummies por provicia
	sucursales_d=pd.get_dummies(fsucursales, columns=['nom_provincia','sucursalTipo'])
	sucursales_new = pd.merge(fsucursales,sucursales_d)

	return fsucursales

def drop_nan_precios(p_df_precios):
	p_df_precios=p_df_precios.dropna()
	return p_df_precios

def drop_nan_productos(p_df_productos):
	p_df_productos=p_df_productos.dropna(subset=['marca'])
	return p_df_productos

def consistenciaDatos(p_df_precios,p_df_productos,p_df_sucursales):
	df_precios = p_df_precios.set_index('producto_id')
	df_productos = p_df_productos.set_index('id')
#	p_df_precios[~df_precios.index.isin(p_df_productos.index)]
	p_df_precios=p_df_precios[df_precios.index.isin(df_productos.index)]

	df_precios_suc = p_df_precios.set_index('sucursal_id')
	df_sucursales = p_df_sucursales.set_index('id')
	p_df_precios=p_df_precios[df_precios_suc.index.isin(df_sucursales.index)]

	return p_df_precios

#Definimos una función para dividir precio por cantidad y unificar las unidades de medida
def precioXUnidad(cantidad: float,  unidad: object, precio: float):
    if unidad in ('un', 'mt','kg','lt'):
        return round(precio/cantidad,2) , unidad
    elif unidad in ('pack'):
        return round(precio,2) ,'pack'
    elif unidad in ('gr'):
        return round(precio/cantidad*1000,2) , 'kg'
    elif unidad in ('ml','cc'):
        return round(precio/cantidad*1000,2) , 'lt'

#funcion para determinar el desvío estámdard poblacional
def pop_std(x):
      return x.std(ddof=0)

def q_at(y):
    @rename(f'q{y:0.2f}')
    def q(x):
        return x.quantile(y)
    return q


def unionDatos(p_df_precios,p_df_productos,p_df_sucursales):
	presentacion_splitted = p_df_productos["presentacion"].str.split(" ", n = 1, expand = True)
	p_df_productos["cantidad"]= presentacion_splitted[0] 
	p_df_productos["cantidad"] = pd.to_numeric(p_df_productos["cantidad"])  
	p_df_productos["unidad"]= presentacion_splitted[1] 	

	dfNew=pd.merge(p_df_precios,p_df_productos,left_on='producto_id',right_on='id',how='inner').drop(columns = 'id')
    
    #agregamos pack como unidad de medida a las promociones que involucran mas de un producto
	dfNew.loc[dfNew['nombre'].str.contains('\\b3x2\\b|\\b2x1\\b|\\b4x3\\b|\\bPaga\\b|\\bLleva\\b|\\bpack\\b', na=False),'unidad']='pack'

	#Creamos dos nuevas columnas denominadas PrecioXUnidad y nuevaUnidad para los valores "homogeneizados"        
	dfNew[['PrecioXUnidad','nuevaUnidad']] = dfNew.swifter.apply(lambda x: precioXUnidad(x['cantidad'],x['unidad'],x['precio']),axis=1, result_type='expand')	

	# Obtenemos un dataset de medias, desvios, mediana, q1 y q3 para fecha y producto (Agregue la mediana para poder di podía realizar el metodo Modified Z-score, pero es más efectivo el Z-score ). las columnas q1 y q3 sirven para calcular IRQ
	df_product_mean_std=dfNew[['fecha','producto_id','PrecioXUnidad']].groupby(['fecha','producto_id'],as_index=False).agg(['mean','median',pop_std,q_at(0.25) ,q_at(0.75)])
	df_product_mean_std.columns = ['media','mediana','desvio','q1','q3']
	df_product_mean_std = df_product_mean_std.reset_index()

	#Unimos los dataset de medias y desvios al dataset original para obtener un dataset resultante con media y desvio de cada producto
	dfNew=pd.merge(dfNew,df_product_mean_std)

	dfNew['IQR']=dfNew.q3-dfNew.q1
	
	dfNew=pd.merge(dfNew,p_df_sucursales,left_on='sucursal_id',right_on='id',how='inner').drop(columns = 'id')

	return dfNew

def outlier(p_df):
	#Sacamos los outliers utilizando z score
	r_df=p_df[(np.abs(p_df.PrecioXUnidad - p_df.media) <= (3 * p_df.desvio))] 

	#Sacamos los outliers utilizando IQR
	r_df=r_df[~((r_df.PrecioXUnidad < (r_df.q1 - 1.5 * r_df.IQR)) | (r_df.PrecioXUnidad > (r_df.q3 + 1.5 * r_df.IQR)))]

	return r_df

#función definida para reemplazar la marca del nombre, sabiendo que no tienen que haber nulos
def replace_substring(x):
    return (
         x["nombre"].replace(x["marca"], "")
    )	

def optimizarDatos(p_df):

	#Dejar solo las columnas necesarias en el dataset
    p_df = p_df[['fecha','sucursal_id', 'sucursalNombre','sucursalTipo','producto_id', 'nombre', 'marca','region','nom_provincia','localidad','cantidad','unidad','nuevaUnidad','precio','PrecioXUnidad','media','desvio']]

    #Renombrar columnas
    p_df.rename(columns={'PrecioXUnidad':'precioXUnidad'},inplace=True)

    p_df['fecha'] = pd.to_datetime(p_df['fecha'], format="%Y%m%d")    

    return p_df

def generarMeta():
	meta = {"fecha": "fecha de precios de producto informado en la pagina de precios claros"
	        ,"sucursal_id": "ID de la sucursal. Los dos primeros numeros determinan la cadena."
	        ,"sucursalNombre": "Nombre de sucursal"
	        ,"sucursalTipo" : "Tipo de sucursal Supermercado, Hipermercado o Auteservicio"
	        ,"producto_id" : "Código EAN del producto"
	        ,"nombre" : "Nombre comercial de Producto"
	        ,"marca" : "Marca del producto"
	        ,"region" : "Región de la sucursal en donde se informe el precio del producto. Las regiones pueden ser Patagonia, Centro, Norte Grande o Cuyo"
	        ,"nom_provincia" : "Nombre de la provincia donde se encuentra la sucursal que informa el precio del producto"
	        ,"localidad" : "Nombre de la localidad donde se encuentra la sucursal que informa el precio del producto"
	        ,"cantidad" : "Cantidad de producto informado"
	        ,"unidad" : "Unidad de medida de producto informado"
	        ,"um_homogenea": "Unidad de medida llevado a lt, kg o unidades segun corresponda"
	        ,"precio" : "Precio del producto en pesos argentinos"
	        ,"precioXUnidad" : "Precio del producto en pesos argentinos por lt, kg o unidades homogeneizado"
	        ,"media" : "Media estadistica por fecha y producto_-id"
	        ,"desvio" : "Desvio poblacional por fecha y producto_id"
	        ,"Created": "13/07/2020"}
	dfMeta = pd.DataFrame.from_dict(meta, orient='index').rename(columns = {0:'Descripción'})    

	return dfMeta

def generarDummies(p_df):
	p_df=pd.get_dummies(p_df, columns=['nom_provincia','sucursalTipo','um_homogenea'])

	return p_df

def obtenerPrecioRelativo(p_df,p_id_producto):

	#Armamos un dataset con la media de PrecioXUnidad por fecha y provincia del bien numerico seleccionado
	dfbn=p_df[(p_df.producto_id==p_id_producto)][['fecha','nom_provincia','precioXUnidad']].groupby(['fecha','nom_provincia'],as_index=False).agg(['mean'])
	dfbn.columns = ['bienNumerico']
	dfbn = dfbn.reset_index()

	#Agregamos al dataset una nueva columna con el precio relativo 
	datasetPR=pd.merge(left=p_df,right=dfbn,how='inner',left_on=['fecha','nom_provincia'],right_on =['fecha','nom_provincia'])
	datasetPR['precioRelativo']=datasetPR.precioXUnidad/datasetPR.bienNumerico	

	return datasetPR

def guardarArchivo(p_df):
	pd.to_pickle(p_df, 'sucursal_producto_precio.pkl', compression="zip")




def main():

	#Carga de datos
	precios=carga_precios()
	productos=carga_productos()
	sucursales=carga_sucursales()

    #Tratamiento de valores faltantes
	precios=drop_nan_precios(precios)
	productos=drop_nan_productos(productos)

    #Consistencia de datos
	precios=consistenciaDatos(precios,productos,sucursales)

    #Union fuentes de datos generando un nuevo dataset
	datasetNew=unionDatos(precios,productos,sucursales)

    #Tratamiento Outliers
	datasetNew=outlier(datasetNew)

    #Realizamos pasos deseables de limpieza y ordenamiento de columnas
	datasetNew=optimizarDatos(datasetNew)

    #Generamos metadata
	meta=generarMeta()

	#Generamos precio relativo en función de un id de producto Crema dental colgate
	idProducto='7793100111563'
	datasetNew=obtenerPrecioRelativo(datasetNew,idProducto)

	print("Total filas Finales", datasetNew.shape)

	#Guardar dataset en archivo
    guardarArchivo(datasetNew)

if __name__ == "__main__":
    main()

print("Guru99")