# import os
# import numpy as np
# from datetime import datetime
# import cv2
# import mediapipe as mp
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision
# from mediapipe import solutions
# from mediapipe.framework.formats import landmark_pb2
# from mediapipe.tasks.python.vision.holistic_landmarker import HolisticLandmarker, HolisticLandmarkerOptions
# import time
# from tqdm import tqdm
# BaseOptions = mp.tasks.BaseOptions
# VisionRunningMode = mp.tasks.vision.RunningMode

# class GenKeypointsHolistic():
#     def __init__(self, cfg, use_model=True):
#         self.detector_holistic = None
#         self.cfg = cfg
#         if use_model:
#             self.load_model()

#     def load_model(self):
#         base_options = python.BaseOptions(model_asset_path='../models/holistic_landmarker.task', delegate= self.cfg['MODEL_DELEGATE'])
#         options = HolisticLandmarkerOptions(
#                 base_options=base_options,
#                 running_mode=VisionRunningMode.VIDEO,
#                 min_face_detection_confidence = 0.5,
#                 min_face_suppression_threshold = 0.5,
#                 min_face_landmarks_confidence = 0.5,
#                 min_pose_detection_confidence = 0.5,
#                 min_pose_suppression_threshold = 0.5,
#                 min_pose_landmarks_confidence = 0.5,
#                 min_hand_landmarks_confidence = 0.5,
#                 output_face_blendshapes = False,
#                 output_segmentation_mask = False,
#             )
#         self.detector_holistic = HolisticLandmarker.create_from_options(options)

#     def load_image(self, image):
#         image = cv2.flip(image, 1)
#         return mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

#     def get_mediapipeResults(self, image):
#         results = {}
#         results['holistic'] = self._get_holisticResults(image)
#         return results

#     def _get_holisticResults(self, image):
#         timestamp = int(time.time() * 1000)
#         return self.detector_holistic.detect_for_video(image, timestamp)
    
#     def get_results(self, mediapipe_results, world=True):

#         detection_result_holistic = mediapipe_results['holistic']
#         data_pose = []
#         data_left_hand = []
#         data_right_hand = []
#         if world:
#             data_pose = detection_result_holistic.pose_world_landmarks
#         else:
#             data_pose = detection_result_holistic.pose_landmarks

#         if world:
#             data_left_hand = holistic_results.left_hand_world_landmarks
#             data_right_hand = holistic_results.right_hand_world_landmarks
#         else:
#             data_left_hand = holistic_results.left_hand_landmarks
#             data_right_hand = holistic_results.right_hand_landmarks

#         return data_pose, data_left_hand, data_right_hand
    
#     def insert_pose_info(self, kps_frame_idx, data, idx):
#         kps_frame_idx[0] = data[idx].x
#         kps_frame_idx[1] = data[idx].y
#         kps_frame_idx[2] = data[idx].z
#         kps_frame_idx[3] = data[idx].visibility
    
#     def insert_hands_info(self, kps_frame_idx, data, idx):
#         kps_frame_idx[0] = data[idx].x
#         kps_frame_idx[1] = data[idx].y
#         kps_frame_idx[2] = data[idx].z
#         kps_frame_idx[3] = 1

#     def get_offset(self, data):
#         offset_hand_left = [data[9][0] - data[19][0], data[9][1] - data[19][1], data[9][2] - data[19][2], 0, 0]
#         offset_hand_right = [data[10][0] - data[40][0], data[10][1] - data[40][1], data[10][2] - data[40][2], 0, 0]
#         return offset_hand_left, offset_hand_right

#     def draw_landmarks_on_image_holistic(image, mediapipe_results):

#         detection_result_holistic = mediapipe_results['holistic']
#         pose_landmarks_list = [detection_result_holistic.pose_landmarks]
#         annotated_image = np.copy(image)

#         for idx in range(len(pose_landmarks_list)):
#             pose_landmarks = pose_landmarks_list[idx]
#             pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
#             pose_landmarks_proto.landmark.extend([
#             landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
#             ])
#             solutions.drawing_utils.draw_landmarks(
#             annotated_image,
#             pose_landmarks_proto,
#             solutions.pose.POSE_CONNECTIONS,
#             solutions.drawing_styles.get_default_pose_landmarks_style())

#         hand_landmarks_data = []
#         hand_landmarks_data.append(detection_result_holistic.left_hand_landmarks)
#         hand_landmarks_data.append(detection_result_holistic.right_hand_landmarks)
#         for idx in range(len(hand_landmarks_data)):
#             hand_landmarks = hand_landmarks_data[idx]
#             hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
#             hand_landmarks_proto.landmark.extend([
#             landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
#             ])
#             solutions.drawing_utils.draw_landmarks(
#             annotated_image,
#             hand_landmarks_proto,
#             solutions.hands.HAND_CONNECTIONS,
#             solutions.drawing_styles.get_default_hand_landmarks_style(),
#             solutions.drawing_styles.get_default_hand_connections_style())

#         return annotated_image

#     def gen_keypoints(self, data_pose, data_left_hand, data_right_hand, type):
#         if type == 'HP':
#             idx_kps_pose = list(range(33))
#             num_total_kps = len(idx_kps_pose) + 21 + 21
#             kps_frame = np.zeros((num_total_kps, 4))

#             if data_pose:
#                 for idx, idx_mediapipe in enumerate(idx_kps_pose):
#                     kps_frame[idx][0] = data_pose[0][idx_mediapipe].x
#                     kps_frame[idx][1] = data_pose[0][idx_mediapipe].y
#                     kps_frame[idx][2] = data_pose[0][idx_mediapipe].z
#                     kps_frame[idx][3] = data_pose[0][idx_mediapipe].presence
            
#             if data_left_hand:
#                 for idx in range(21):
#                     kps_frame[idx+len(idx_kps_pose)][0] = data_left_hand[0][idx].x
#                     kps_frame[idx+len(idx_kps_pose)][1] = data_left_hand[0][idx].y
#                     kps_frame[idx+len(idx_kps_pose)][2] = data_left_hand[0][idx].z
#                     kps_frame[idx+len(idx_kps_pose)][3] = data_left_hand[0][idx].presence

#             if data_right_hand:
#                 for idx in range(21):
#                     kps_frame[idx+len(idx_kps_pose)+21][0] = data_right_hand[0][idx].x
#                     kps_frame[idx+len(idx_kps_pose)+21][1] = data_right_hand[0][idx].y
#                     kps_frame[idx+len(idx_kps_pose)+21][2] = data_right_hand[0][idx].z
#                     kps_frame[idx+len(idx_kps_pose)+21][3] = data_right_hand[0][idx].presence
            
#             return kps_frame

#         elif type == 'SIGNAMED':

#             idx_kps_pose = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
#             num_total_kps = len(idx_kps_pose) + 21 + 21
#             kps_frame = np.zeros((num_total_kps, 4))

#             if data_pose:
#                 for idx, idx_mediapipe in enumerate(idx_kps_pose):
#                     kps_frame[idx][0] = data_pose[0][idx_mediapipe].x
#                     kps_frame[idx][1] = data_pose[0][idx_mediapipe].y
#                     kps_frame[idx][2] = data_pose[0][idx_mediapipe].z
#                     kps_frame[idx][3] = data_pose[0][idx_mediapipe].presence

#             if data_left_hand:
#                 for idx in range(21):
#                     kps_frame[idx+len(idx_kps_pose)][0] = data_left_hand[0][idx].x
#                     kps_frame[idx+len(idx_kps_pose)][1] = data_left_hand[0][idx].y
#                     kps_frame[idx+len(idx_kps_pose)][2] = data_left_hand[0][idx].z
#                     kps_frame[idx+len(idx_kps_pose)][3] = data_left_hand[0][idx].presence

#             if data_right_hand:
#                 for idx in range(21):
#                     kps_frame[idx+len(idx_kps_pose)+21][0] = data_right_hand[0][idx].x
#                     kps_frame[idx+len(idx_kps_pose)+21][1] = data_right_hand[0][idx].y
#                     kps_frame[idx+len(idx_kps_pose)+21][2] = data_right_hand[0][idx].z
#                     kps_frame[idx+len(idx_kps_pose)+21][3] = data_right_hand[0][idx].presence
            
#             return kps_frame