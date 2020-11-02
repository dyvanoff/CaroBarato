import pandas as pd
from io import StringIO
import time


def get_sucursales_df():
    try:
        start = time.time()
        sucursales = pd.DataFrame()

        try:
            sucursales = pd.read_pickle('../models/sucursales.pkl', compression='zip')
        except:
            sucursal_url = 'https://raw.githubusercontent.com/solujan/mentoria_2020/master/raw_dataset/sucursales.csv'
            sucursales = pd.read_csv(sucursal_url)
            provincia_txt = """provincia	nom_provincia	region
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
                    AR-Y	Jujuy 	Norte Grande
                    AR-Z	Santa Cruz	Patagonia
                    """
            provincia_csv = StringIO(provincia_txt)
            entidad_provincia = pd.read_csv(provincia_csv, sep=('\t'))
            entidad_provincia['provincia'] = entidad_provincia['provincia'].str.strip()
            sucursales = sucursales.merge(entidad_provincia, on='provincia')

            pd.to_pickle(sucursales, '../models/sucursales.pkl', compression="zip")

        finally:
            stop = time.time()
            print("get_sucursales_df:", round(stop-start, 3),"segs")
            return sucursales
    except Exception as e:
        raise


def get_productos_df():
    try:
        start = time.time()
        productos = pd.DataFrame()
        try:
            productos = pd.read_pickle('../models/productos.pkl', compression='zip')
        except:
            producto_url = 'https://raw.githubusercontent.com/solujan/mentoria_2020/master/raw_dataset/productos.csv'
            productos = pd.read_csv(producto_url)
            pd.to_pickle(productos, '../models/productos.pkl', compression="zip")
        finally:
            stop = time.time()
            print("get_productos_df:", round(stop-start, 3),"segs")
            return productos
    except Exception as e:
        raise


def get_precios_df():
    try:
        start = time.time()

        precios = pd.DataFrame()

        try:
            precios = pd.read_pickle('../models/precios.pkl', compression='zip')
        except:
            precios_20200412_20200413 = pd.read_csv(
                'https://raw.githubusercontent.com/solujan/mentoria_2020/master/raw_dataset/precios_20200412_20200413.csv')
            precios_20200419_20200419 = pd.read_csv(
                'https://raw.githubusercontent.com/solujan/mentoria_2020/master/raw_dataset/precios_20200419_20200419.csv')
            precios_20200426_20200426 = pd.read_csv(
                'https://raw.githubusercontent.com/solujan/mentoria_2020/master/raw_dataset/precios_20200426_20200426.csv')
            precios_20200502_20200503 = pd.read_csv(
                'https://raw.githubusercontent.com/solujan/mentoria_2020/master/raw_dataset/precios_20200502_20200503.csv')
            precios_20200518_20200518 = pd.read_csv(
                'https://raw.githubusercontent.com/solujan/mentoria_2020/master/raw_dataset/precios_20200518_20200518.csv')

            lista_df_px = [precios_20200412_20200413, precios_20200419_20200419, precios_20200426_20200426,
                           precios_20200502_20200503, precios_20200518_20200518]
            fecha_px = ['20200412', '20200419', '20200426', '20200502', '20200518']

            for df, fecha in zip(lista_df_px, fecha_px):
                df['fecha'] = fecha
                precios = pd.concat([precios, df])

            pd.to_pickle(precios, '../models/precios.pkl', compression="zip")
        finally:
            stop = time.time()
            print("get_precios_df:", round(stop - start, 3), "segs")
            return precios

    except Exception as e:
        raise
