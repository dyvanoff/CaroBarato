from unidecode import unidecode
import pandas as pd
import numpy as np
import re
import time

spanish_stopwords = ['ante', 'bajo', 'cabe', 'con', 'contra', 'de', 'desde',
                     'durante', 'en', 'entre', 'hacia', 'hasta', 'mediante',
                     'para', 'por', 'segun', 'sin', 'sobre', 'tras', 'la', 'las', 'los', 'del', 'el', 'a', 'y']


def function_model(df):
    """
    Calcula el precio relativo de los productos en base al precio del bien numerario elegido
    :param df: Dataframe PRODUCTO_SUCURSAL_PRECIO
    :return: Dataframe con 'precio_relativo' calculado
    """
    try:
        start = time.time()
        df = df
        stop = time.time()
        print("function_model:", round(stop - start, 3), "segs")
        return df
    except Exception as e:
        raise


def get_mean_precio(df):
    """
    Calcula el precio medio del producto por region
    :param df:
    :return:
    """
    try:
        mean_precio_x_prod = df[['precio', 'producto_id']].groupby('producto_id',).mean().rename(columns={'precio':'precio_producto_mean'})
        df = pd.merge(df, mean_precio_x_prod, left_on='producto_id', right_index=True)
        return df
    except Exception as e:
        raise


def get_precio_numerario_normalizado(df, producto_numerario_id):
    """
    Calcula el precio normalizado del bien numerario considerando la media por fecha
    :param df: Dataframe PRODUCTO_SUCURSAL_PRECIO
    :return: Dataframe con 'precio_relativo' calculado
    """
    try:
        start = time.time()
        bien_numerario_by_fecha = df[df['producto_id'] == producto_numerario_id].groupby('fecha') \
                                        .agg(precio_numerario_normalizado=('precio_normalizado', 'mean')).reset_index()

        df = pd.merge(df, bien_numerario_by_fecha[['fecha', 'precio_numerario_normalizado']],
                                                                                    left_on='fecha', right_on='fecha')
        stop = time.time()
        print("get_precio_numerario_normalizado:", round(stop - start, 3), "segs")
        return df
    except Exception as e:
        raise


def get_factor_normalizador(df):
    """
    Calcula el factor con el cual llevar todas las unidades de medidas secundarias a unidades de medida primarias.
    Con este factor luego se extrapola el calculo del precio normalizado correspondiente
    :param df: Dataframe PRODUCTO_SUCURSAL_PRECIO
    :return: Dataframe con 'factor_normalizador' calculado
    """
    try:
        start = time.time()
        um_primaria = ['kg', 'lt', 'un', 'mt', 'pack']
        um_secundaria = ['gr', 'ml', 'cc']

        df.loc[df['um_limpia'].isin(um_primaria), 'factor_normalizador'] = 1 / df['cant_limpia']
        df.loc[df['um_limpia'].isin(um_secundaria), 'factor_normalizador'] = 1 / df['cant_limpia'] * 1000
        stop = time.time()
        print("get_factor_normalizador:", round(stop - start, 3), "segs")
        return df
    except Exception as e:
        raise


def get_precio_normalizado(df):
    """
    Calcula el precio normalizado de los productos en base a la 'um_limpia' y 'cant_limpia'
    :param df: Dataframe PRODUCTO_SUCURSAL_PRECIO
    :return: Dataframe con 'precio_relativo' calculado
    """
    try:
        start = time.time()
        df = get_factor_normalizador(df)
        df['precio_normalizado'] = df['precio'] * df['factor_normalizador']
        stop = time.time()
        print("get_precio_normalizado:", round(stop - start, 3), "segs")
        return df
    except Exception as e:
        raise


def get_precio_relativo(df, producto_numerario_id='7794000960077'):
    """
    Calcula el precio relativo de los productos en base al precio del bien numerario elegido
    :param df: Dataframe PRODUCTO_SUCURSAL_PRECIO
    :param producto_numerario_id: ID del bien numerario seleccionado.
    :return: Dataframe con 'precio_relativo' calculado
    """
    try:
        start = time.time()
        df = get_precio_numerario_normalizado(df, producto_numerario_id)
        df['precio_relativo'] = df['precio_normalizado'] / df['precio_numerario_normalizado']
        stop = time.time()
        print("get_precio_relativo:", round(stop - start, 3), "segs")
        return df
    except Exception as e:
        raise


def get_fecha_anterior(fecha_actual, fechas):
    try:
        idx_actual = np.where(fechas == fecha_actual)[0][0]
        if idx_actual == 0:
            return fecha_actual
        else:
            return fechas[idx_actual - 1]
    except Exception as e:
        raise


def get_precio_anterior(df):
    try:
        start = time.time()
        fechas = df['fecha'].unique()

        df['fecha_anterior'] = [get_fecha_anterior(row['fecha'], fechas) for idx, row in df.iterrows()]
        precios_grouped = df.groupby(['producto_id', 'fecha', 'provincia'])['precio'].mean()

        lista_precios = []
        bandera = 50000
        i = 0

        for idx, row in df.iterrows():
            try:
                precio_old = precios_grouped.loc[(row['producto_id'], row['fecha_anterior'], row['provincia'])]
            except:
                precio_old = row['precio']
            lista_precios.append(precio_old)
            if i > bandera:
                duration = round((time.time() - start), 3)
                print(">>>", bandera, '- Duration:', duration)
                bandera = bandera + 50000
                start = time.time()
            i += 1
        df['precio_anterior'] = lista_precios

        stop = time.time()
        print("get_precio_anterior:", round(stop - start, 3), "segs")
        return df
    except Exception as e:
        raise


def get_mean(df):
    """
    Funcion que calcula la media  por 'producto_id' y 'fecha'
    e inserta estas nuevas columnas al dataset.
    :param df: Recibe el dataframe de PRECIOS
    :return: Dataframe original con las columnas ['mean' , 'std']
    """
    try:
        start = time.time()
        precio_mean = df.groupby(['producto_id', 'fecha']).agg(precio_mean=('precio', 'mean'))
        stop = time.time()
        print("get_mean:", round(stop-start, 3), "segs")
        return pd.merge(df, precio_mean, left_on=['producto_id', 'fecha'], right_index=True)
    except Exception as e:
        raise


def get_df_precios_a_borrar(dict_precio):
    """
    Recorre el diccionario toma por cada key el 'precio_mean_diff' de mayor valor.
    :param dict_precio: Diccionario de ('producto_id','fecha'): [precio_mean_diff]
    :return: Dataframe de producto_id, fecha y max(precio_mean_diff)
    """
    try:
        df = pd.DataFrame()
        for idx, diff in dict_precio.items():
            df = df.append({'producto_id': idx[0], 'fecha': idx[1], 'precio_mean_diff': max(diff)}, ignore_index=True)
        return df
    except Exception as e:
        raise


def drop_precios_sin_sucursal(df):
    """
    Elimina aquellos registros del dataframe de precios_sucursales para los cuales el id de sucursal asociado
    al registro en precios no existe en sucursales
    :param df: Dataframe PRECIOS_SUCURSAL
    :return: Dataframe PRECIOS_SUCURSAL depurado
    """
    try:
        print(">>DROP PRECIOS SIN SUCURSALES")
        reg_df = len(df)
        print("Cantidad de Registros del Dataframe:", reg_df)
        df = df[df['id'].notna()]
        print("Cantidad de Registros del Dataframe Limpio:", len(df))
        print('\n> > > > Se han limpiado', (reg_df - len(df)), 'registros\n')
        return df
    except Exception as e:
        raise


def drop_precios_duplicados(df):
    """
    Limpia del dataframe de PRECIOS aquellos registros cuyo 'producto_id', 'fecha' y 'sucursal_id' son iguales,
    manteniendo aquel registro cuya diferencia con la media por 'producto_id' y 'fecha' es  menor.
    No se utiliza la media por sucursal ya que al haber 2 productos mal reportados, la distancia de ambos con
    la media es la misma.
    :param df: Dataframe de PRECIOS
    :return: Dataframe de PRECIOS depurado
    """
    try:
        start = time.time()
        print(">>DROP DE PRECIOS DUPLICADOS")
        reg_df = len(df)
        print("Cantidad de Registros del Dataframe:", reg_df)
        df = get_mean(df)
        df['precio_mean_diff'] = abs(df['precio'] - df['precio_mean'])

        precios_duplicados = df[df.duplicated(subset=['producto_id', 'fecha', 'sucursal_id'], keep=False)]

        dict_precio_mean_diff = precios_duplicados.groupby(['producto_id', 'fecha'])['precio_mean_diff'].apply(
            lambda x: x.tolist()).to_dict()

        df_precios_duplicados_borrar = get_df_precios_a_borrar(dict_precio_mean_diff)

        for idx, row in df_precios_duplicados_borrar.iterrows():
            filtro = (df.producto_id == row.producto_id) \
                     & (df.fecha == row.fecha) \
                     & (df.precio_mean_diff == row.precio_mean_diff)

            df = df[~filtro]

        print("Cantidad de Registros del Dataframe Limpio:", len(df))
        print('\n> > > > Se han limpiado', (reg_df - len(df)), 'registros\n')

        stop = time.time()
        return df
    except Exception as e:
        raise


def get_quantiles(df, group_by_col, col_precio, col_q25_name, col_q75_name):
    """
    Calcula cuantil 25% y 75% para los datos agrupados por 'producto_id', 'fecha' y 'region'
    :param df: Dataframe de Precios y Sucursales
    :return: La union de dataframe anterior con el resultante de los datos agrupados
    """
    try:
        start = time.time()
        grp_quantile = df.groupby(group_by_col).agg(q25=(col_precio, lambda x: np.quantile(x, .25)), q75=(col_precio, lambda x: np.quantile(x, .75)))
        grp_quantile = grp_quantile.rename(columns={"q25": col_q25_name, "q75": col_q75_name})
        stop = time.time()
        print('get_quantiles:', round(stop - start, 3), "segs")

        return pd.merge(df, grp_quantile, left_on=group_by_col, right_on=group_by_col, how='left')
    except Exception as e:
        raise


def is_outlier(df, precio, cuartil_25, cuartil_75, columna_resultado):
    try:
        df.loc[precio > cuartil_75 + (3 * (cuartil_75 - cuartil_25)), columna_resultado] = 'extremo superior'
        df.loc[precio < cuartil_25 - (3 * (cuartil_75 - cuartil_25)), columna_resultado] = 'extremo inferior'
        df.loc[(cuartil_75 + 3 * (cuartil_75 - cuartil_25) >= precio) & (
                                    precio >= cuartil_25 - 3 * (cuartil_75 - cuartil_25)), columna_resultado] = 'normal'
        df.loc[(cuartil_25 == cuartil_75), columna_resultado] = 'normal'

        return df
    except Exception as e:
        raise


def drop_outliers_by_precios(df, group_by_col, col_precio, dataframe):
    """
    Elimina los registros cuyos precios son outliers segun el rango intercuartilico de la distribucion de precios
    agrupada por producto_id, fecha y region
    :param df: Dataframe sobre el cual realizar el drop de registros
    :param group_by_col: Lista de columnas a agrupar el calculo del rango intercuartilico
    :param col_precio: Nombre de la columna precio sobre el cual se calcula el rengo
    :param dataframe: Nombre en string del dataframe sobre el cual se aplican los calculos
    :return: Dataframe sin precios outliers
    """
    try:
        start = time.time()
        reg_df = len(df)

        if dataframe == 'precio_sucursal':
            print(">>DROP DE OUTLIERS DE PRECIO_SUCURSAL")
            print("Cantidad de Registros del Dataframe:", reg_df)

            df = get_quantiles(df, group_by_col, col_precio, 'cuartil_25', 'cuartil_75')
            df = is_outlier(df, df['precio'], df['cuartil_25'], df['cuartil_75'], 'rdo_ri_geo')
            df = df[df['rdo_ri_geo'] == 'normal']
        else:
            print(">>DROP DE OUTLIERS DE PRECIO_SUCURSAL_PRODUCTO")
            print("Cantidad de Registros del Dataframe:", reg_df)
            df = get_quantiles(df, group_by_col, col_precio, 'relat_cuartil_25', 'relat_cuartil_75')
            df = is_outlier(df, df['precio_relativo'], df['relat_cuartil_25'], df['relat_cuartil_75'], 'rdo_ri_geo_relativo')
            df = df[df['rdo_ri_geo_relativo'] == 'normal']

        print("Cantidad de Registros del Dataframe Limpio:", len(df))
        print('\n> > > > Se han limpiado', (reg_df - len(df)), 'registros\n')
        stop = time.time()
        print("drop_outliers_precios_sucursales:", round(stop - start, 3), "segs")
        return df
    except Exception as e:
        raise


def limpiar_palabras(columna, limpiar_stopwords=True):
    """

    :param columna:
    :param limpiar_stopwords:
    :return:
    """
    try:
        columna_limpia = []

        for contenido in columna:
            contenido = contenido.lower()
            contenido = unidecode(contenido)

            if limpiar_stopwords:
                palabra_limpia = ''
                palabras = contenido.split(' ')

                for palabra in palabras:
                    if palabra not in spanish_stopwords:
                        palabra_limpia += palabra.strip() + ' '
                columna_limpia.append(palabra_limpia.strip())
            else:
                columna_limpia.append(contenido)

        return columna_limpia
    except Exception as e:
        raise


def get_dummy_column(column, top_rank=10, default_word='others'):
    """
    Esta funcion acepta una columna y una lista de palabras mas frecuentes
    Con ello, crea una nueva lista la cual contiene la palabra original (si se encuentra dentro de las mas frecuentes)
    o le pone la palabra suministrada
    :param column: Columna sobre la cual aplicar la "dummyzacion"
    :param top_rank: Cantidad de palabras a considerar para la "dummyzacion"
    :param default_word: Se utiliza esta palabra como valor por defecto si no se encuentra dentro de las mas frecuentes
    :return: Columna con palabras dummys
    """
    try:
        dummy_list = column.value_counts().reset_index()[:top_rank]['index'].tolist()
        dummy_column = []
        for row in column:
            valor_dummy = default_word
            if row in dummy_list:
                valor_dummy = row
            dummy_column.append(valor_dummy)

        return dummy_column
    except Exception as e:
        raise


def get_provincia_dummy(df):
    """

    :param df:
    :return:
    """
    try:
        start = time.time()
        df['provincia_depurada'] = df['nom_provincia'].str.lower()
        for letra, reemplazo in zip(['á', 'é', 'í', 'ó', 'ú'], ['a', 'e', 'i', 'o', 'u']):
            df['provincia_depurada'] = df['provincia_depurada'].str.replace(letra, reemplazo)
        df['provincia_depurada'] = df['provincia_depurada'].str.replace(' ', '_')
        stop = time.time()
        print("get_provincia_dummy:", round(stop - start, 3), "segs")
        return df
    except Exception as e:
        raise


def get_banderaDescripcion_dummy(df):
    """

    :param df_sucursales:
    :return:
    """
    try:
        start = time.time()
        df['banderaDescripcion_depurado'] = df['banderaDescripcion'].str.lower()

        for letra, reemplazo in zip(['á', 'é', 'í', 'ó', 'ú'], ['a', 'e', 'i', 'o', 'u']):
            df['banderaDescripcion_depurado'] = df['banderaDescripcion_depurado'].str.replace(letra, reemplazo)

        df['banderaDescripcion_depurado'] = df['banderaDescripcion_depurado'].str.replace('.', '')
        df['banderaDescripcion_depurado'] = df['banderaDescripcion_depurado'].str.replace(' ', '_')

        df['banderaDescripcion_dummy'] = get_dummy_column(df['banderaDescripcion_depurado'], 15, 'otras_bandDesc')
        stop = time.time()

        print("get_sucursales_dummy:", round(stop - start, 3), "segs")
        return df
    except Exception as e:
        raise


def get_marca_producto_dummy(df, top_n_palabras=10):
    """
    Devuelve el Dataframe con la columna de "marca_dummy" lista.
    :param df_productos: Dataframe de PRODUCTOS
    :return: Devuelve el Dataframe con la columna de "marca_dummy" lista
    """
    try:
        stopwords = ['a', 'de', 'en', 'y', 'x', '&', 'la', 'al', 'un', 'con', 'del', 'el', '.', 'para', 'sin'] + \
                    df['um_limpia'].unique().tolist()

        start = time.time()

        df['marca_depurada'] = df['marca'].str.lower()

        for letra, reemplazo in zip(['á', 'é', 'í', 'ó', 'ú', 'ñ'], ['a', 'e', 'i', 'o', 'u', 'n']):
            df['nombre_depurado'] = df['nombre_depurado'].str.replace(letra, reemplazo)
            df['marca_depurada'] = df['marca_depurada'].str.replace(letra, reemplazo)

        df['nombre_marca'] = df['nombre_depurado'] + ' ' + df['marca_depurada']
        df['nombre_marca'] = df['nombre_marca'].str.split()  # Separo las palabras
        df['nombre_marca_depurado'] = df['nombre_marca'].apply(lambda x: set(x))  # Utilizo la funcion set para quitar duplicados
        df['nombre_marca_depurado'] = df['nombre_marca_depurado'].apply(lambda x: ' '.join(x))  # Vuelvo a juntar las palabras
        df['nombre_marca_depurado'] = df['nombre_marca_depurado'].str.replace(r'\d', '')
        df['nombre_marca_depurado'] = df['nombre_marca_depurado'].apply(lambda x: [word for word in x.split() if word not in stopwords])

        df_mp = pd.DataFrame(data=[item for sublist in df['nombre_marca_depurado'].tolist() for item in sublist],
                             columns=['palabras'])  # Armo un dataframe que contiene una columna con TODAS las palabras

        top_palabras = df_mp['palabras'].value_counts().reset_index().rename(columns={'index': 'palabras', 'palabras': 'cantidad'})
        lista_palabras_frecuentes = top_palabras['palabras'][:top_n_palabras].to_list() + ['otras']

        df['nombre_marca_depurado'] = df['nombre_marca_depurado'].apply(lambda x: ' '.join(x))  # Vuelvo a juntar las palabras
        df = get_top_words_dummy_vars(df, 'nombre_marca_depurado', lista_palabras_frecuentes, 'producto_')

        stop = time.time()
        print("get_marca_producto_dummy:", round(stop - start, 3), "segs")
        return df
    except Exception as e:
        raise


def get_idreferencia(df):
    """
    Funcion que extrae el ID de Referencia para solucionar el problema de inconsistencia
    en donde un mismo producto cuyo ('nombre, 'marca', 'presentacion') son iguales
    :param df: Recibe el dataframe de PRODUCTOS
    :return: columna con ID de Referencia
    """
    try:
        start = time.time()
        diccionario = df.groupby(['nombre', 'marca', 'presentacion'])['id'].apply(lambda x: x.tolist()).to_dict()
        stop = time.time()
        print("get_idreferencia:", round(stop-start, 3),"segs")
        return [sorted(diccionario.get((producto.nombre, producto.marca, producto.presentacion)))[0]
            for idx, producto in df.iterrows()]
    except Exception as e:
        raise


def is_float(number):
    """
    Funciona auxiliar para determinar si un string puede convertirse en float
    :param number:  Numero en string
    :return: True or False si el numero puede convertirse a float
    """
    try:
        float(number)
        return True
    except Exception as e:
        return False


def get_initial_cleanup(df_productos):
    """
    Realiza una serie de limpiezas y extracciones iniciales
    :param df_productos: Dataframe PRODUCTOS
    :return: Dataframe con nuevas columnas
    """
    try:
        start = time.time()
        df = df_productos.copy()
        df['nombre_depurado'] = df['nombre'].str.lower().fillna('')
        df['presentacion_depurada'] = df['presentacion'].str.lower()
        df['um_en_presentacion'] = df['presentacion_depurada'].str[-2:]
        df['cantidad_en_presentacion'] = df['presentacion_depurada'].str[0:-3]
        stop = time.time()
        print("get_initial_cleanup:", round(stop-start, 3),"segs")
        return df
    except Exception as e:
        raise


def get_top_words_dummy_vars(df, word_column, top_word_list, prefix):
    """
    Toma una columna con palabras y una lista de "palabras mas frecuentes" y devuelve una matriz dummy
    en donde cada columna representa una palabra frecuenta y mapea los correspondientes '1' si en la
    columna con palabras aparece el target deseado.

    :param df: DataFrame con la lista de palabras a convetir a dummy
    :param word_column: Columna el DataFrame 'df' que contiene las palabras
    :param top_word_list: Lista con las palabras sobre las cuales mapear las dummys
    :param prefix: Prefijo para agregar a las nueva columnas
    :return: DataFrame de dummies
    """
    try:
        dummy_df = pd.DataFrame(0, index=df.index, columns=top_word_list)
        for idx, row in df.iterrows():
            palabras_encontradas = []
            for palabra in row[word_column].split(' '):
                for top_word in top_word_list:
                    if palabra == top_word:
                        palabras_encontradas.append(top_word)
            if len(palabras_encontradas) == 0:
                dummy_df.loc[idx, top_word_list[-1]] = 1
            else:
                dummy_df.loc[idx, palabras_encontradas] = 1

        return dummy_df.add_prefix(prefix)
    except Exception as e:
        raise


def get_um_presentacion_from_nombre(df):
    """
    Extraer la 'um' y 'presentacion' del nombre del producto
    :param df: Dataframe de PRODUCTOS
    :return:   Columnas ['um_en_nombre_prod', 'cant_en_nombre_prod']
    """
    try:
        start = time.time()
        cant_en_nombre_prod = []
        um_en_nombre_prod = []
        pattern_um = r'(?P<cant_en_nombre_prod>[\d\.]+)\s?(?P<um_en_nombre_prod>\D{1,3}$)'

        for idx, producto in df.iterrows():
            # Si no encuentro un match, lo dejo como esta en la presentacion
            um = producto['um_en_presentacion']
            # Si no encuentro un match, lo dejo como esta en la presentacion
            cantidad = producto['cantidad_en_presentacion']

            match = re.search(pattern_um, producto["nombre_depurado"])
            if match is not None:
                if is_float(match.group('cant_en_nombre_prod')):
                    cantidad = match.group('cant_en_nombre_prod')
                    um = match.group('um_en_nombre_prod')

            cant_en_nombre_prod.append(cantidad)
            um_en_nombre_prod.append(um)
        stop = time.time()
        print("get_um_presentacion_from_nombre:", round(stop-start, 3),"segs")
        return cant_en_nombre_prod, um_en_nombre_prod

    except Exception as e:
        raise


def get_um_fixed(df):
    """
    Aplica un set expresiones regulares para arreglar algunas 'um'
    :param df: Dataframe de PRODUCTOS
    :return: Dataframe con columna 'um_en_nombre_prod' corregida
    """
    try:
        df['um_en_nombre_prod'] = df['um_en_nombre_prod'].str.replace('[^a-zA-Z]', '')
        df['um_en_nombre_prod'] = df['um_en_nombre_prod'].str.replace('k\w+', 'kg')
        df['um_en_nombre_prod'] = df['um_en_nombre_prod'].str.replace('g\w+', 'gr')
        df['um_en_nombre_prod'] = df['um_en_nombre_prod'].str.replace('un\w+', 'un')
        df['um_en_nombre_prod'] = df['um_en_nombre_prod'].str.replace('^g$', 'gr')
        df['um_en_nombre_prod'] = df['um_en_nombre_prod'].str.replace('^l$', 'lt')
        df['um_en_nombre_prod'] = df['um_en_nombre_prod'].str.replace('^u$', 'un')
        df['um_en_nombre_prod'] = df['um_en_nombre_prod'].str.replace('^c$', 'cc')

        return df

    except Exception as e:
        raise


def is_pack(nombre_producto):
    # En el peor de los casos, si la palabra 'pack' esta al ultimo de la cadena, la funcion demora "23.5 µs per loop"
    for palabra in nombre_producto.split(' '):
        if re.search(r'^pack', palabra) is not None:
            return True
    return False


def is_pack(nombre_producto):
    """
    Funcion auxiliar para detectar si una cadena tiene la palabra 'pack'
    :param nombre_producto:
    :return: True si la palabra contiene la regex 'pack'
    """
    try:
        for palabra in nombre_producto.split(' '):
            if re.search(r'^pack', palabra) is not None:
                return True
        return False
    except Exception as e:
        raise


def get_presentacion_limpia(df):
    """
    Obtiene la mejor presentacion posible de aquel producto cuya presentacion original y la presentacion
    presente en el nombre no coincida
    :param df: Dataframe PRODUCTOS
    :return: Columnas ['um_limpia', 'cant_limpia]
    """
    try:
        start = time.time()
        um_limpia = []
        cant_limpia = []

        pack_pattern = r'^.*\s\d{1,3}\s{0,1}\w{1,3}\s[y\+]\s.*\d{1,3}\s{0,1}\w{1,3}.*$'
        un_pattern = r'(?P<cant>[\d]+)\sun'

        for idx, producto in df.iterrows():
            producto_um_limpia = ''
            producto_cant_limpia = ''

            if is_pack(producto['nombre_depurado']):
                producto_um_limpia = 'pack'
                producto_cant_limpia = 1

            elif re.search(pack_pattern, producto['nombre_depurado']) is not None:
                producto_um_limpia = 'pack'
                producto_cant_limpia = 1

            elif producto['um_en_presentacion'] == producto['um_en_nombre_prod']:
                producto_um_limpia = producto['um_en_presentacion']

                if float(producto['cantidad_en_presentacion']) == float(producto['cant_en_nombre_prod']):
                    producto_cant_limpia = producto['cantidad_en_presentacion']
                else:
                    producto_cant_limpia = producto['cant_en_nombre_prod']

            elif float(producto['cantidad_en_presentacion']) == float(producto['cant_en_nombre_prod']):
                producto_cant_limpia = producto['cant_en_nombre_prod']
                if producto['um_en_nombre_prod'] in df['um_en_presentacion'].unique():
                    producto_um_limpia = producto['um_en_nombre_prod']
                else:
                    producto_um_limpia = producto['um_en_presentacion']

            else:
                extraer_unid = re.search(un_pattern, producto['nombre_depurado'])
                if extraer_unid is not None:
                    producto_um_limpia = 'un'
                    producto_cant_limpia = extraer_unid.group(
                        'cant')  ## el 'cant' sale de un "extractor" que se define en el patron.
                elif producto['um_en_presentacion'] == 'un':
                    producto_um_limpia = producto['um_en_presentacion']
                    producto_cant_limpia = producto['cantidad_en_presentacion']

                elif producto['um_en_nombre_prod'] in df['um_en_presentacion'].unique():
                    producto_um_limpia = producto['um_en_nombre_prod']
                    producto_cant_limpia = producto['cant_en_nombre_prod']

                else:
                    producto_um_limpia = producto['um_en_presentacion']
                    producto_cant_limpia = producto['cantidad_en_presentacion']

            um_limpia.append(producto_um_limpia)
            cant_limpia.append(float(producto_cant_limpia))

        stop = time.time()
        print("get_presentacion_limpia:", round(stop-start, 3),"segs")
        return um_limpia, cant_limpia

    except Exception as e:
        raise


