import cv2 
import mediapipe as mp
import numpy as np

_mp_drawing = mp.solutions.drawing_utils
_mp_holistic = mp.solutions.holistic

def transform_video(video:list):
    """
    Return keypoints from the video.
    Input: video : cv2.VideoCapture
    Output: key_points : numpy.array
    """
    with _mp_holistic.Holistic(
        min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        return _process_video(video, holistic)

def _process_video(video, holistic):
    processed_points = []
    for frame in video:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(frame_rgb)
        _draw_landmarks(frame, results,"RHand")
        _draw_landmarks(frame, results,"LHand")
        _draw_landmarks(frame, results,"Pose")
        _draw_landmarks(frame, results,"Face")
        processed_points.append(_extract_keypoints(results=results))
    return np.array(processed_points)
    



def _extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])




def _return_configurations(type:str,results):
    configuration_draw = {
        "RHand":{
            "results":results.right_hand_landmarks,
            "connections": _mp_holistic.HAND_CONNECTIONS,
            "drawing1": _mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
            "drawing2":_mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        },
        "LHand":{
            "results":results.left_hand_landmarks,
            "connections": _mp_holistic.HAND_CONNECTIONS,
            "drawing1":  _mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
            "drawing2":_mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
        },
        "Pose":{
            "results":results.pose_landmarks,
            "connections": _mp_holistic.POSE_CONNECTIONS,
            "drawing1": _mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
            "drawing2":_mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
        },
        "Face":{
            "results":results.face_landmarks,
            "connections": _mp_holistic.FACEMESH_TESSELATION,
            "drawing1": _mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
            "drawing2":_mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
        }
    } 
    return configuration_draw[type]

def _draw_landmarks(frame, results, type:str):
    configuration = _return_configurations(type,results)
    _mp_drawing.draw_landmarks(frame, configuration["results"], configuration["connections"], 
                                    configuration["drawing1"], 
                                    configuration["drawing2"]
                                    )