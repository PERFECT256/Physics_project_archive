import cv2
import numpy as np
import pandas as pd
import math
import os

# 1. 파일명 설정 
video_path = "C:/Users/kimyh/OneDrive/문서/바탕 화면/Physics_Project/FFFF.mp4"

if not os.path.exists(video_path):
    print(f"오류: '{video_path}' 파일을 찾을 수 없습니다. 파이썬 파일과 같은 폴더에 있는지 확인하세요.")
    exit()

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0: fps = 30.0 # 메타데이터가 없을 경우 기본값

# 2. 하얀색 마커를 잡기 위한 HSV 범위 설정
# H(색상)은 무관, S(채도)는 낮게, V(명도)는 매우 높게 (하얀색 특성)
lower_white = np.array([0, 0, 200])
upper_white = np.array([180, 50, 255])

data_log = []
frame_count = 0

print("영상 분석을 시작합니다. (종료하려면 영상 창을 클릭하고 'q'를 누르세요)")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break # 영상이 끝나면 루프 탈출
        
    # 영상을 BGR에서 흑백(Grayscale)으로 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 핵심 1: 밝기(명도)가 230 이상인 '진짜 눈부신 하얀색'만 추출
    _, mask = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
    
    # 핵심 2: 노란색 매트 완벽 차단! 화면 아래쪽 40%를 아예 까맣게 지워버림
    height = mask.shape[0]
    mask[int(height * 0.60):, :] = 0 
    
    # 노이즈 제거 (자잘한 먼지 지우기)
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # 마커의 외곽선(Contours) 찾기
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 너무 작은 노이즈(먼지 등) 필터링 후 면적 기준 내림차순 정렬
    valid_contours = [c for c in contours if cv2.contourArea(c) > 10]
    valid_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)[:2]
    
    # 마커가 정확히 2개 인식되었을 때만 계산
    if len(valid_contours) == 2:
        points = []
        for cnt in valid_contours:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                points.append((cx, cy))
        
        # 핵심 로직: x좌표 기준으로 정렬 (points[0]이 항상 왼쪽 마커)
        points = sorted(points, key=lambda p: p[0])
        
        # 두 점 사이의 각도 계산 (dy, dx)
        dx = points[1][0] - points[0][0]
        dy = points[1][1] - points[0][1] # OpenCV는 y축이 아래로 갈수록 증가하므로 방향 주의
        
        # 라디안을 디그리로 변환
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)
        
        # 마이너스 각도 보정 (항상 지면 대비 기울기로 통일하기 위함)
        if angle_deg < 0:
            angle_deg += 180 
            
        # 기준 각도(수평 0도) 대비 실제 틸트 각도
        tilt_angle = angle_deg - 180 if angle_deg > 90 else angle_deg
        
        # 데이터 기록
        t = frame_count / fps
        data_log.append({"Time(s)": t, "Angle(deg)": tilt_angle})
        
        # 화면에 시각화 (디버깅용)
        cv2.circle(frame, points[0], 5, (0, 0, 255), -1) # 왼쪽 마커 (빨강)
        cv2.circle(frame, points[1], 5, (255, 0, 0), -1) # 오른쪽 마커 (파랑)
        cv2.line(frame, points[0], points[1], (0, 255, 0), 2) # 두 마커 연결선 (초록)
        
        cv2.putText(frame, f"Angle: {tilt_angle:.2f} deg", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    
    # 결과 영상 보여주기
    cv2.imshow('Tracking (White Markers)', frame)
    
    # 키보드 'q' 누르면 강제 종료 (약 10ms 대기 - 재생 속도 조절)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
        
    frame_count += 1

cap.release()
cv2.destroyAllWindows()

# 3. 추출된 데이터를 CSV 파일로 저장
if data_log:
    df = pd.DataFrame(data_log)
    df.to_csv('C:/Users/kimyh/OneDrive/문서/바탕 화면/Physics_Project/phantom_torque_experiment.csv', index=False)
    print("데이터 추출 완료! 'phantom_torque_experiment.csv' 파일이 저장되었습니다.")
else:
    print("인식된 데이터가 없습니다. HSV 색상 범위를 조절해 보세요.")