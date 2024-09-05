import numpy as np

data_npy_cerca = '/home/thirus/WS_keypoints/mediapipe_keypoints/DATA/KEYPOINTS/POSE_HANDS_IMAGE_05/ania_cerca.npy'
data_npy_lejos = '/home/thirus/WS_keypoints/mediapipe_keypoints/DATA/KEYPOINTS/POSE_HANDS_IMAGE_05/ania_lejos.npy'

data_npy_cerca_w = '/home/thirus/WS_keypoints/mediapipe_keypoints/DATA/KEYPOINTS/POSE_HANDS_WORLD_05/ania_cerca.npy'
data_npy_lejos_w = '/home/thirus/WS_keypoints/mediapipe_keypoints/DATA/KEYPOINTS/POSE_HANDS_WORLD_05/ania_lejos.npy'

data_cerca = np.load(data_npy_cerca)
data_lejos = np.load(data_npy_lejos)
data_cerca_w = np.load(data_npy_cerca_w)
data_lejos_w = np.load(data_npy_lejos_w)

print(data_cerca.shape)
print(data_lejos.shape)
print(data_cerca_w.shape)
print(data_lejos_w.shape)


print(data_lejos[:,9,2])
print(data_cerca[:,9,2])

print(data_lejos_w[:,9,2])
print(data_cerca_w[:,9,2])


print('CALCULAR MAX:')

diff_cerca = np.max(data_cerca[:,9,2])  -  np.min(data_cerca[:,9,2])
diff_lejos = np.max(data_lejos[:,9,2])  -  np.min(data_lejos[:,9,2])
diff_cerca_w = np.max(data_cerca_w[:,9,2])  -  np.min(data_cerca_w[:,9,2])
diff_lejos_w = np.max(data_lejos_w[:,9,2])  -  np.min(data_lejos_w[:,9,2])
                                                      
print('max_menos_min data_cerca', diff_cerca   )
print('max_menos_min data_lejos', diff_lejos    )
print('max_menos_min data_cerca_w', diff_cerca_w    )
print('max_menos_min data_lejos_w', diff_lejos_w    )


print('dif_lejos-cerca',  diff_lejos / diff_cerca )
print(  diff_lejos_w / diff_cerca_w )

print( diff_cerca /diff_cerca_w )
print( diff_lejos / diff_lejos_w )



""" 

thirus@thirus:~/WS_keypoints/mediapipe_keypoints/src$ python reportZ.py 
(37, 61, 4)
(37, 61, 4)
(37, 61, 4)
(37, 61, 4)

[-0.39203113 -0.94001895 -1.07033837 -1.32699823 -1.19254768 -1.53266704
 -1.31063986 -1.49139094 -1.3163321  -1.39691532 -1.23914802 -1.27226627
 -1.24262929 -1.24508393 -1.4031781  -1.37274563 -1.33509076 -1.39755952
 -1.28418219 -1.30469787 -1.30283058 -1.35301566 -1.30667496 -1.34195209
 -1.27892113 -1.32851601 -1.25809264 -1.27377152 -1.2688961  -1.40108156
 -1.23645508 -1.50589716 -1.70512748 -1.65083468 -1.30181468 -1.29391503
 -1.29702401]

[-1.05387735 -0.69958788 -0.90728468 -0.88443774 -0.94143587 -1.20239067
 -1.13271117 -2.2991426  -2.38747478 -2.5829885  -2.34707022 -2.21459651
 -2.26190209 -2.14541936 -2.15567422 -2.0451076  -2.33906817 -2.43438697
 -2.41977835 -2.41034412 -2.55439591 -2.52814317 -2.42009449 -2.31941175
 -2.26948619 -2.18062663 -1.94019234 -2.35414767 -2.1586647  -2.19278574
 -2.47320724 -2.40690136 -1.84212267 -1.9929539  -2.02236629 -1.91189432
 -1.03084016]

[-0.0629624  -0.12961181 -0.15483944 -0.22145033 -0.21131967 -0.25538298
 -0.24358991 -0.25240231 -0.24996115 -0.28996369 -0.28654608 -0.28320938
 -0.28105471 -0.27493319 -0.29233232 -0.28877854 -0.282291   -0.28669357
 -0.2855044  -0.28387854 -0.27621374 -0.29058078 -0.28705609 -0.28930527
 -0.28236139 -0.28925323 -0.27845609 -0.27852872 -0.27892783 -0.29686227
 -0.27550614 -0.30456927 -0.31538332 -0.31423029 -0.24704234 -0.23769639
 -0.23438583]
 
[-0.11799461 -0.0773328  -0.09202949 -0.10103656 -0.09722006 -0.11801998
 -0.10422286 -0.30175769 -0.32829508 -0.33057055 -0.33182609 -0.32818887
 -0.32955888 -0.32568991 -0.291816   -0.28497231 -0.31303108 -0.33814895
 -0.33777642 -0.32431546 -0.34812564 -0.33992165 -0.33769372 -0.32768804
 -0.32184583 -0.32031268 -0.28654772 -0.30418953 -0.30447602 -0.31440163
 -0.32597837 -0.33091688 -0.24884556 -0.24946778 -0.26429394 -0.25637937
 -0.12537386]

CALCULAR MAX:
max_menos_min data_cerca 1.8834006190299988
max_menos_min data_lejos 1.3130963444709778
max_menos_min data_cerca_w 0.2707928344607353
max_menos_min data_lejos_w 0.25242091715335846
0.697194389342007
0.9321550832614783
6.955134624520834
5.202010828893413 

"""