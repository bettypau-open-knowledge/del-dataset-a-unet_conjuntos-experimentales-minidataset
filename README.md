# 🧪 Diseño de conjuntos experimentales para segmentación de imágenes

Este repositorio contiene una práctica introductoria de **diseño y verificación de conjuntos de entrenamiento, validación y prueba**, desarrollada como parte del taller **«Del dataset a U-Net: segmentación de imágenes con Deep Learning»**.

Para esta actividad utilizaremos una **versión reducida y reorganizada del Oxford-IIIT Pet Dataset**, que contiene fotografías de perros y gatos, sus correspondientes máscaras de segmentación y archivos de anotaciones.

El propósito es construir dos configuraciones de partición del minidataset, estudiar la distribución de las categorías entre los conjuntos y verificar que las imágenes y sus máscaras permanezcan asociadas, sin duplicaciones ni omisiones de identificadores.

---

## 🎯 Objetivo

Diseñar conjuntos experimentales reproducibles mediante Python, aplicando **particiones aleatorias estratificadas por raza** con proporciones **70/15/15** y **80/10/10** para entrenamiento (*train*), validación (*validation*) y prueba (*test*), respectivamente. Examinar los resultados mediante tablas de contingencia y comprobar la integridad de los identificadores antes de utilizar las particiones en experimentos de segmentación.

---

## 📁 Organización del proyecto

El notebook utiliza las siguientes ubicaciones dentro del repositorio:

```text
del-dataset-a-unet_conjuntos-experimentales-minidataset/
├── config/
│   └── dataset.json
├── data/
│   ├── manifests/
│   │   └── manifest.csv
│   ├── raw/
│   │   ├── images/
│   │   │   ├── chihuahua/
│   │   │   │   ├── chihuahua_12.jpg
│   │   │   │   ├── chihuahua_12.png
│   │   │   │   ├── chihuahua_12.xml
│   │   │   │   └── ...
│   │   │   ├── Maine_Coon/
│   │   │   │   ├── Maine_Coon_13.jpg
│   │   │   │   ├── Maine_Coon_13.png
│   │   │   │   ├── Maine_Coon_13.xml
│   │   │   │   └── ...
│   │   │   ├── Siamese/
│   │   │   │   ├── Siamese_101.jpg
│   │   │   │   ├── Siamese_101.png
│   │   │   │   ├── Siamese_101.xml
│   │   │   │   └── ...
│   │   │   └── Sphynx/
│   │   │       ├── Sphynx_19.jpg
│   │   │       ├── Sphynx_19.png
│   │   │       ├── Sphynx_19.xml
│   │   │       └── ...
│   │   └── list.txt
│   └── splits/                       # Generado al ejecutar la práctica
│       ├── split_70_15_15.csv
│       └── split_80_10_10.csv
└── notebooks/
    └── conjuntos_experimentales.ipynb
```

Los archivos originales del minidataset se encuentran en `data/raw/`. Cada categoría cuenta con su propio directorio dentro de `images/`, donde se almacenan conjuntamente las fotografías, máscaras y anotaciones individuales. Los archivos de una misma muestra comparten el **mismo nombre base**, pero tienen extensiones diferentes (por ejemplo, `Sphynx_19.jpg`, `Sphynx_19.png` y `Sphynx_19.xml`).

El archivo `data/manifests/manifest.csv` permite identificar las muestras y localizar sus imágenes y máscaras. La configuración de rutas se administra mediante `config/dataset.json`. Al ejecutar el notebook, se crea el directorio `data/splits/` para guardar las particiones obtenidas.

> [!NOTE]
> El minidataset contiene únicamente una pequeña selección de imágenes del Oxford-IIIT Pet Dataset original, reorganizada con fines didácticos. No representa el conjunto completo de datos ni está diseñado para entrenar o evaluar modelos de segmentación de manera representativa.

---

## 🧩 Diseño de los conjuntos experimentales

En la práctica se generan dos configuraciones:

| Configuración | Train | Validation | Test |
|---|---:|---:|---:|
| **70/15/15** | 70 % | 15 % | 15 % |
| **80/10/10** | 80 % | 10 % | 10 % |

La partición se realiza con `train_test_split()` de **scikit-learn**, utilizando una semilla fija (`random_state=42`) para facilitar la reproducibilidad. Se emplea la columna `breed` como criterio de **estratificación**, con el propósito de conservar aproximadamente la distribución de razas en cada conjunto.

Dado que el minidataset es pequeño, los porcentajes efectivos pueden diferir ligeramente de los solicitados por el redondeo a números enteros. La estratificación tampoco garantiza proporciones idénticas en cada partición.

### 📄 Archivos de salida

Cada configuración se guarda en un archivo CSV dentro de `data/splits/`:

- `split_70_15_15.csv`
- `split_80_10_10.csv`

Los archivos contienen únicamente las siguientes columnas:

| Columna | Descripción |
|---|---|
| `image_id` | Identificador único de la muestra. |
| `breed` | Raza asociada con la imagen. |
| `image_path` | Ruta de la fotografía. |
| `mask_path` | Ruta de la máscara de segmentación correspondiente. |
| `split` | Conjunto asignado: `train`, `validation` o `test`. |

La imagen y su máscara **no se separan**: ambas permanecen referenciadas en una misma fila. Las particiones se registran mediante CSV, sin mover ni duplicar los archivos originales.

### 🔎 Verificación de los resultados

El notebook permite inspeccionar la distribución de razas por conjunto mediante `pd.crosstab()` y verificar con instrucciones `assert` que:

- No existan identificadores `image_id` repetidos dentro de una configuración.
- Los identificadores de cada configuración coincidan exactamente con los del manifest original.
- Las etiquetas de partición correspondan a `train`, `validation` o `test`.

Estas comprobaciones permiten detectar errores de asignación, pero **no descartan por sí solas** imágenes casi idénticas o muestras relacionadas con identificadores distintos.

---

## 💻 Requisitos

Para ejecutar la práctica se utilizan las siguientes bibliotecas:

| Biblioteca | Uso |
|---|---|
| `pathlib` | Construcción y manejo de rutas del proyecto. |
| `json` | Lectura y actualización del archivo de configuración. |
| `pandas` | Lectura del manifest, organización de particiones, tablas de contingencia y exportación a CSV. |
| `scikit-learn` | División aleatoria y estratificada mediante `train_test_split()`. |

También necesitarás un entorno compatible con **Jupyter Notebooks**. Las bibliotecas `pathlib` y `json` forman parte de la biblioteca estándar de Python.

---

## 📚 Dataset de referencia

**Oxford-IIIT Pet Dataset**

Parkhi, O. M., Vedaldi, A., Zisserman, A., & Jawahar, C. V. (2012).

*Cats and Dogs.*

IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

[Consultar el dataset original →](https://www.robots.ox.ac.uk/~vgg/data/pets/)

El conjunto de imágenes utilizado en esta práctica es una selección reducida y reorganizada con fines didácticos. Su utilización debe respetar las condiciones de uso de las imágenes del dataset original.

---

## 🔗 Taller

Esta práctica forma parte del **Módulo 2 · Ingeniería del dataset** del taller:

[**Del dataset a U-Net: segmentación de imágenes con Deep Learning →**](https://github.com/bettypau-open-knowledge/del-dataset-a-unet)

**Bettypau Open Knowledge**

*Aprende · Experimenta · Construye · Comparte*
