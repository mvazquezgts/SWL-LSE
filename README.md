## Uso de Mediapipe

Integra:
1. Mediapipe Holistic (Legacy)
2. Mediapipe Hands & Pose
3. Mediapipe Holistic ( En construcción - la versión actual presenta bugs )

-------

# Requisitos:

- python >= 3.8
- mediapipe >= 0.10.10

-------

# Contenido:

En la carpeta API se integra las diferentes implementación / uso de mediapipe de modo que sea transparente que versión utilizar con unicamente instanciar una u otra clase, clases posibles:
* GenKeypointsPoseHands
* GenKeypointsHolistic
* GenKeypointsHolisticLegacy

Por otro lado, la carpeta API incluye genFeatures con las funciones de preprocesado y adecuación de los keypoints y generación de los diferentes vectores de características que utilizaremos en el entrenamiento/inferencia.

En la carpeta models se incluyen los modelos utilizados con la nueva API de mediapipe, incluyendo la versión "heavy" del modelo de extración de los puntos del cuerpo.

En la carpeta raiz se incluye diferentes scripts:
 - useMediapipeAPI, script todo en uno permite dada una carpeta con vídeos y extraer tanto el vídeo obtenido con mediapipe, como extraer los kps y los vectores de características.
 - generate_mediapipe.py, permite extraer la salida del modelo dada una carpeta de vídeos y guardando esta salida como ficheros .pickle.
 - generate_arr_keypoints.py, a partir de los pickles previamente extraídos se puede obtener ficheros .npy con los keypoints utilizados en Signamed.
 - generate_features.py, permite obtener los vectores de características utilizados en los modelos de Signamed a partir de los ficheros .npy

-------

# Modo de Uso

## useMediapipeAPI
![Pipeline generate dataset](imgs/original_mediapipe.png)

    parser.add_argument('--model', required=True, choices=['POSE_HANDS', 'HOLISTIC_LEGACY', 'HOLISTIC'])
    parser.add_argument('--video_path', required=True, type=str)
    parser.add_argument('--folder_out_video', required=False, type=str)
    parser.add_argument('--folder_out_keypoints', required=False, type=str)
    parser.add_argument('--folder_out_features', required=False, type=str)
    parser.add_argument('--type_kps', required=False, default='C4_xyzc', type=str)


python useMediapipeAPI.py --model POSE_HANDS --video_path /home/tmpvideos/mvazquez/ISLR_workspace/KEYPOINTS_MEDIAPIPE/DATA/VIDEOS_IN/ania_lejos.mp4 --folder_out_video ../out_videos --folder_out_keypoints ../out_keypoints

python useMediapipeAPI.py --model HOLISTIC_LEGACY --video_path /home/tmpvideos/mvazquez/ISLR_workspace/KEYPOINTS_MEDIAPIPE/videos_in/ania_lejos.mp4 --folder_out_keypoints ../out_keypoints

# PIPELINE: VIDEOS -> DATASET

![Pipeline generate dataset](imgs/pipeline_preprocessing.png)


## generate_mediapipe

    parser.add_argument('--folder_input_videos', required=True, type=str)
    parser.add_argument('--pose_hands', action='store_true')
    parser.add_argument('--holistic', action='store_true')
    parser.add_argument('--holistic_legacy', action='store_true')
    parser.add_argument('--folder_output_mediapipe', required=True, type=str)

python generate_mediapipe.py --pose_hands --folder_input_videos ../DATA/VIDEOS_IN --folder_output_mediapipe ../DATA/MEDIAPIPE/POSE_HANDS_05

python generate_mediapipe.py --holistic_legacy --folder_input_videos ../DATA/VIDEOS_IN --folder_output_mediapipe ../DATA/MEDIAPIPE/HOLISTIC_LEGACY_05


## generate_arr_keypoints

    parser.add_argument('--pose_hands', action='store_true')
    parser.add_argument('--holistic', action='store_true')
    parser.add_argument('--holistic_legacy', action='store_true')

    parser.add_argument('--folder_input_mediapipe', default='', type=str)
    parser.add_argument('--folder_output_kps', required=True, type=str)

    parser.add_argument('--world', action='store_true')

python generate_arr_keypoints.py --folder_input_mediapipe ../DATA/MEDIAPIPE/POSE_HANDS_05 --pose_hands --folder_output_kps ../DATA/KEYPOINTS/POSE_HANDS_IMAGE_05

python generate_arr_keypoints.py --world --folder_input_mediapipe ../DATA/MEDIAPIPE/POSE_HANDS_05 --pose_hands --folder_output_kps ../DATA/KEYPOINTS/POSE_HANDS_WORLD_05

python generate_arr_keypoints.py --folder_input_mediapipe ../DATA/MEDIAPIPE/HOLISTIC_LEGACY_05 --holistic_legacy --folder_output_kps ../DATA/KEYPOINTS/HOLISTIC_LEGACY_IMAGE_05



## generate_features

    parser.add_argument('--folder_in_kps', required=True, type=str)
    parser.add_argument('--folder_out_features', required=True, type=str)

    parser.add_argument('--type_kps', required=False, default='C4_xyzc', type=str)
    parser.add_argument('--offset', action='store_true')
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--noFramesLimit', action='store_true')


python generate_features.py --type_kps C4_xyzc --offset --normalize --folder_in_kps ../DATA/KEYPOINTS/POSE_HANDS_IMAGE_05 --folder_out_features ../DATA/FEATURES/POSE_HANDS_IMAGE_05



