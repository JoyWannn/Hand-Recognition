import cv2
import mediapipe as mp
import numpy as np
from math import sqrt
import socket

class HandGestureRecognizer:
    def __init__(self):
        # 初始化MediaPipe手部模型
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        # mediapipe 繪圖方法
        self.mp_draw = mp.solutions.drawing_utils

        # 初始化UDP接口進行Unity傳輸
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.serverAddressPort = ("127.0.0.1", 5052)  # Unity接口

        # Previous gesture for avoiding repeated sends
        self.previous_gesture = None

    # 抓手指節點距離
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        # sqrt取正平方根，找point1與point2距離
        return sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)

    # 通過節點距離辨識動作
    def recognize_gesture(self, landmarks):
        """
        Recognize gestures based on hand landmarks
        Returns the name of the gesture
        """
        # 獲得指尖位置
        thumb_tip = landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        index_tip = landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]

        # 獲得指根位置
        thumb_base = landmarks.landmark[self.mp_hands.HandLandmark.THUMB_MCP]
        index_base = landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]
        middle_base = landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        ring_base = landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_MCP]
        pinky_base = landmarks.landmark[self.mp_hands.HandLandmark.PINKY_MCP]

        # 計算指尖到指根距離
        distances = {
            'thumb': self.calculate_distance(thumb_tip, thumb_base),
            'index': self.calculate_distance(index_tip, index_base),
            'middle': self.calculate_distance(middle_tip, middle_base),
            'ring': self.calculate_distance(ring_tip, ring_base),
            'pinky': self.calculate_distance(pinky_tip, pinky_base)
        }

        # 定義臨界點判斷手指是否彎折(0.13相對穩定)
        threshold = 0.1  # 這邊可隨測試狀況更動，default=0.08
        #adaptive

        # 判斷手指是否彎折
        fingers_extended = {
            'thumb': distances['thumb'] > threshold,
            'index': distances['index'] > threshold,
            'middle': distances['middle'] > threshold,
            'ring': distances['ring'] > threshold,
            'pinky': distances['pinky'] > threshold
        }

        #辨識手指是否揮動(目前暫時辨識不到)
        x,y = int(index_tip.x),int(index_tip.y)
        prev_x = 0
        prev_y = 0
        dx = x - prev_x
        dy = y - prev_y
        if dx > 50:
            dx = x
            return "right"
        elif dx < -50:
            dx = x
            return "left"

        # 辨識特定動作
        if all(fingers_extended.values()):
            #開掌
            return "OPEN_PALM"
        elif not any(fingers_extended.values()):
            #握拳
            return "CLOSED_FIST"
        elif fingers_extended['thumb'] and not any(v for k, v in fingers_extended.items() if k != 'thumb') and  thumb_tip.y < pinky_tip.y:
            #大拇指朝上(距離要再抓一下)
            return "up"
        elif fingers_extended['thumb'] and not any(v for k, v in fingers_extended.items() if k != 'thumb') and  thumb_tip.y > pinky_tip.y:
            #大拇指朝下(距離要再抓一下)
            return "down"
        elif self.calculate_distance(thumb_tip, index_tip)<0.1:
            #拿東西
            return 'Take'
        elif fingers_extended['index'] and not any(v for k, v in fingers_extended.items() if k != 'index'):
            #伸出食指點擊
            return "POINTING"
        else:
            return "UNKNOWN"

    def process_frame(self, frame):
        """Process a single frame and return the frame with annotations"""
        # 把手的顏色從BGR轉RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 放到呈現結果上
        results = self.hands.process(rgb_frame)

        # 找節點、辨識、顯示手部動作
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 畫出手上的節點
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )

                # 辨識手部節點的動作
                gesture = self.recognize_gesture(hand_landmarks)

                # 顯示手部動作的名稱
                wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
                h, w, c = frame.shape
                x = int(wrist.x * w)
                y = int(wrist.y * h)
                cv2.putText(frame, gesture, (x - 50, y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return frame

    # def getRecognizeResult(self,frame):
    #     # 把手的顏色從BGR轉RGB
    #     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #
    #     # 放到呈現結果上
    #     results = self.hands.process(rgb_frame)
    #     # 找節點、辨識、顯示手部動作
    #     if results.multi_hand_landmarks:
    #         for hand_landmarks in results.multi_hand_landmarks:
    #             # 辨識手部節點的動作
    #             gesture = self.recognize_gesture(hand_landmarks)
    #
    #     return gesture

    def send_gesture_to_unity(self, gesture):
        """Send gesture recognition result to Unity via UDP"""
        try:
            # Only send if gesture has changed to reduce network traffic
            if gesture != self.previous_gesture:
                print(f"Sending gesture to Unity: {gesture}")
                self.sock.sendto(gesture.encode(), self.serverAddressPort)
                self.previous_gesture = gesture
        except Exception as e:
            print(f"Error sending gesture: {e}")


def main():
    # 呼叫gesture recognizer這個class
    recognizer = HandGestureRecognizer()

    # 開啟電腦內置相機
    cap = cv2.VideoCapture(0)

    while True:
        # 閱讀相機畫面
        ret, frame = cap.read()
        if not ret:
            break

        # 處理視窗
        processed_frame = recognizer.process_frame(frame)

        #呼叫landmark模型
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = recognizer.hands.process(rgb_frame)

        #傳送手勢內容
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                gesture = recognizer.recognize_gesture(hand_landmarks)
                recognizer.send_gesture_to_unity(gesture)

        # 顯示視窗
        cv2.imshow('Hand Gesture Recognition', processed_frame)

        #串接socket & 設定Port
        # sock = socket.socket(socket.AF_INET , socket.SOCK_DGRAM)
        # sock.sendto(recognizer.getRecognizeResult(frame).encode(),("127.0.0.1", 5052))

        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    # Clean up
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


