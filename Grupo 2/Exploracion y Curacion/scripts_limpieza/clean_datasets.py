import logging
import os
from logging.handlers import RotatingFileHandler

import pandas as pd
import time

from mentoria.scripts import custom_tools as tools, import_datasets as impds

logger = logging.getLogger(__name__)


def setup_logger(log_path, log_name):
    log_file = '.'.join([log_name, 'log'])
    handler = RotatingFileHandler(filename=os.path.join(log_path, log_file), maxBytes=2 * 1024 * 1024, backupCount=1)
    handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(handler)
    logger.setLevel(logging.ERROR)


def main():
    try:
        start = time.time()
        productos = impds.get_productos_df()
        sucursales = impds.get_sucursales_df()
        precios = impds.get_precios_df()

        productos = productos[productos['marca'].notna()]
        precios = precios[precios['precio'].notna()]

        productos['id_referencia'] = tools.get_idreferencia(productos)

        precios = tools.get_mean_precio(precios)

        # Preparo el dataset con columnas auxiliarees de cantidad y um
        productos = tools.get_initial_cleanup(productos)

        # Llamo a la funcion extractora de 'um' y 'cantidad' del nombre_depurado
        cant_en_nombre_prod, um_en_nombre_prod = tools.get_um_presentacion_from_nombre(productos)

        # Guardo el resultado de la funcion en las respectivas NUEVAS columnas
        productos['cant_en_nombre_prod'] = cant_en_nombre_prod
        productos['um_en_nombre_prod'] = um_en_nombre_prod

        # BORRADO DE VARIABLES
        del cant_en_nombre_prod
        del um_en_nombre_prod
        productos = tools.get_um_fixed(productos)

        um_limpia, cant_limpia = tools.get_presentacion_limpia(productos)

        productos['um_limpia'] = um_limpia
        productos['cant_limpia'] = cant_limpia

        # BORRADO DE VARIABLES
        del um_limpia
        del cant_limpia

        productos_dummy_df = tools.get_marca_producto_dummy(productos, 400)
        productos = pd.merge(productos, productos_dummy_df, left_index=True, right_index=True)

        del productos_dummy_df

        productos = pd.concat([productos, pd.get_dummies(productos['um_limpia'], prefix='um')], axis=1)

        productos_col_drop = ['marca', 'nombre', 'presentacion', 'categoria1', 'categoria2', 'categoria3',
                              'id_referencia', 'nombre_depurado', 'presentacion_depurada', 'um_en_presentacion',
                              'cantidad_en_presentacion', 'cant_en_nombre_prod', 'um_en_nombre_prod']

        productos.drop(columns=productos_col_drop, inplace=True)

        sucursales = tools.get_provincia_dummy(sucursales)
        sucursales = pd.concat([sucursales, pd.get_dummies(sucursales['provincia_depurada'], prefix='prov')], axis=1)

        sucursales['sucursaltipo_depurado'] = sucursales['sucursalTipo'].str.lower()
        sucursales = pd.concat([sucursales, pd.get_dummies(sucursales['sucursaltipo_depurado'], prefix='suctipo')], axis=1)

        sucursales = tools.get_banderaDescripcion_dummy(sucursales)
        sucursales = pd.concat([sucursales, pd.get_dummies(sucursales['banderaDescripcion_dummy'], prefix='banddesc')], axis=1)

        sucursales_col_drop = ['comercioId', 'banderaId', 'banderaDescripcion', 'comercioRazonSocial',
                                 'localidad', 'direccion', 'lat', 'lng', 'sucursalNombre', 'sucursalTipo',
                                 'nom_provincia', 'provincia_depurada', 'sucursaltipo_depurado',
                                 'banderaDescripcion_depurado', 'banderaDescripcion_dummy']

        sucursales.drop(columns=sucursales_col_drop, inplace=True)

        precios = tools.drop_precios_duplicados(precios)
        precios = pd.concat([precios, pd.get_dummies(precios['fecha'], prefix='fecha')], axis=1)

        precio_sucursal = pd.merge(precios, sucursales, left_on='sucursal_id', right_on='id', how='left')

        precio_sucursal = tools.get_precio_anterior(precio_sucursal)

        del precios
        del sucursales

        precio_sucursal = tools.drop_precios_sin_sucursal(precio_sucursal)
        precio_sucursal.drop(columns='id', inplace=True)

        precio_sucursal = tools.drop_outliers_by_precios(precio_sucursal, ['producto_id', 'fecha', 'region'], 'precio', 'precio_sucursal')

        precio_sucursal_producto = pd.merge(precio_sucursal, productos, left_on='producto_id', right_on='id', how='left').drop(columns='id')

        del productos
        del precio_sucursal

        precio_sucursal_producto = tools.get_precio_normalizado(precio_sucursal_producto)

        producto_numerario_id = '7794000960077'
        precio_sucursal_producto = tools.get_precio_relativo(precio_sucursal_producto, producto_numerario_id)

        precio_sucursal_producto = tools.drop_outliers_by_precios(precio_sucursal_producto, ['producto_id', 'fecha', 'region'], 'precio_relativo','precio_sucursal_producto')

        stop = time.time()
        print("\n\n>>>>>>>>>>>>FIN:", round(stop - start, 3), "segs")

        precio_sucursal_producto.drop(columns=['precio_mean', 'precio_mean_diff', 'um_limpia', 'cant_limpia', 'factor_normalizador', 'precio_normalizado'],
                                      inplace=True)

        precio_sucursal_producto.drop(columns=['precio', 'producto_id', 'sucursal_id', 'fecha',
                                               'region', 'cuartil_25', 'cuartil_75',
                                               'rdo_ri_geo', 'marca_depurada', 'nombre_marca', 'nombre_marca_depurado',
                                               'precio_numerario_normalizado', 'relat_cuartil_25', 'relat_cuartil_75',
                                               'rdo_ri_geo_relativo'], inplace=True)




        # pd.to_pickle(precio_sucursal_producto, '../models/full_precio_sucursal_producto.pkl', compression="zip", protocol=4)

        pd.to_pickle(precio_sucursal_producto, '../models/precio_sucursal_producto_400.pkl', compression="zip", protocol=4)
    except Exception as e:
        logger.error('%s | %s', 'main', str(e))


if __name__ == '__main__':
    script_path = os.path.dirname(__file__)
    script_name = os.path.basename(__file__).split('.')[0]
    setup_logger(script_path, script_name)
    main()
