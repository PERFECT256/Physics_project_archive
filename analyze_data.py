import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 1. 실험 데이터 불러오기
csv_file = 'phantom_torque_experiment.csv'
try:
    df = pd.read_csv(csv_file)
except FileNotFoundError:
    print(f"오류: '{csv_file}' 파일을 찾을 수 없습니다.")
    exit()

slow_motion_factor = 8.0  # 120fps 촬영 / 30fps 재생인 경우 4배 (240fps면 8.0)

exp_time = df['Time(s)'].values / slow_motion_factor
exp_angle = df['Angle(deg)'].values



# 2. 물리 파라미터 세팅
# 실제 실험하신 반원 껍질의 대략적인 수치로 맞춰주세요. (현재는 기본값)
R = 0.05  # 반경 (예: 10cm = 0.1m)
g = 9.81

theta_0_deg = 32.4       # 시작 각도
theta_max_deg = -45.0      # 도달한 최대 각도

theta_0_rad = np.radians(theta_0_deg)
theta_max_rad = np.radians(theta_max_deg)

# -45도까지 올라가기 위해 처음에 가해진 초기 각속도 계산
cos_0 = np.cos(theta_0_rad)
cos_max = np.cos(theta_max_rad)

num_v = (2 * g / (R * np.pi)) * (cos_0 - cos_max)
den_v = 1 - (2 * cos_0 / np.pi)
theta_dot_0 = np.sqrt(num_v / den_v)  # rad/s (양수 방향으로 푸쉬)

# 3. 미분 방정식 정의 (시뮬레이터와 동일)
def ode_case_a(t, state):
    """ 잘못된 방정식 (팬텀 토크 무시) """
    theta, theta_dot = state
    num = -(np.sin(theta) / np.pi) * (g / R)
    den = 1 - (2 * np.cos(theta) / np.pi)
    return [theta_dot, num / den]

def ode_case_b(t, state):
    """ 올바른 방정식 (팬텀 토크 포함) """
    theta, theta_dot = state
    num = -(np.sin(theta) / np.pi) * (theta_dot**2) - (np.sin(theta) / np.pi) * (g / R)
    den = 1 - (2 * np.cos(theta) / np.pi)
    return [theta_dot, num / den]

# 4. 수치해석 (ODE 풀기)
t_span = (0, exp_time[-1]) # 실험 데이터 시간만큼만 시뮬레이션
t_eval = np.linspace(0, exp_time[-1], 500)
state_0 = [theta_0_rad, theta_dot_0] 

sol_A = solve_ivp(ode_case_a, t_span, state_0, t_eval=t_eval, method='RK45')
sol_B = solve_ivp(ode_case_b, t_span, state_0, t_eval=t_eval, method='RK45')

# 결과를 디그리(deg) 단위로 변환
theta_A_deg = np.degrees(sol_A.y[0])
theta_B_deg = np.degrees(sol_B.y[0])

from scipy.signal import find_peaks

# ==========================================

# 1. 실험 데이터 주기 계산
peaks_exp, _ = find_peaks(exp_angle, prominence=5)
if len(peaks_exp) > 1:
    T_exp = np.mean(np.diff(exp_time[peaks_exp]))
else:
    T_exp = 0

# 2. 정답 이론(팬텀 토크 O) 주기 계산
peaks_B, _ = find_peaks(theta_B_deg, prominence=5)
T_B = np.mean(np.diff(sol_B.t[peaks_B]))

# 3. 오답 이론(팬텀 토크 X) 주기 계산
peaks_A, _ = find_peaks(theta_A_deg, prominence=5)
T_A = np.mean(np.diff(sol_A.t[peaks_A]))

print("\n --- 진동 주기(Period) 분석 결과 ---")
print(f"1. 실제 실험 주기 (Experiment): {T_exp:.3f} 초")
print(f"2. 정답 이론 주기 (Phantom O) : {T_B:.3f} 초 (오차: {abs(T_exp - T_B)/T_exp * 100:.2f}%)")
print(f"3. 오답 이론 주기 (Phantom X) : {T_A:.3f} 초 (오차: {abs(T_exp - T_A)/T_exp * 100:.2f}%)")
print("--------------------------------------\n")

# 5. 그래프 시각화 (논문/보고서 삽입용)
plt.figure(figsize=(10, 6))

# 시뮬레이션 결과 Plot
plt.plot(sol_A.t, theta_A_deg, 'r--', linewidth=2, label='Theory (Naive, No Phantom Torque)')
plt.plot(sol_B.t, theta_B_deg, 'b-', linewidth=2.5, label='Theory (Exact, With Phantom Torque)')

# 실제 실험 데이터 Plot (투명도 alpha를 주어 시뮬레이션 선과 겹쳐 보이게 함)
plt.plot(exp_time, exp_angle, 'o', color='green', markersize=3, alpha=0.6, label='Experiment Data')

# 그래프 꾸미기
plt.axhline(0, color='black', linewidth=1)
plt.title('Angular Trajectory: Theory vs Experiment', fontsize=14, fontweight='bold')
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Tilt Angle (degrees)', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(fontsize=10, loc='upper right')

# 그래프 상하 여백 조절
plt.ylim(min(exp_angle)-10, max(exp_angle)+10)

plt.tight_layout()

# 이미지 파일로 저장 & 화면에 띄우기
plt.savefig('Section7_Verification_Plot.png', dpi=300)
print("그래프 저장 완료: Section7_Verification_Plot.png")
plt.show()