# ==========================================
# Asymmetric Rolling Bodies & Phantom Torque
# Streamlit Interactive Dashboard
# ==========================================

import streamlit as st
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import plotly.graph_objects as go

# ------------------------------------------
# 1. Page Configuration & Session State Setup
# ------------------------------------------
st.set_page_config(page_title="Phantom Torque Simulation", layout="wide")

if "history" not in st.session_state:
    st.session_state.history = []
    st.session_state.run_id = 1

# ------------------------------------------
# 2. Sidebar Controls
# ------------------------------------------
st.sidebar.header("Simulation Controls")
theta_0_deg = st.sidebar.slider("Initial Angle (θ₀) [degrees]", 10.0, 85.0, 60.0, 1.0)
R = st.sidebar.slider("Hoop Radius (R) [m]", 0.5, 2.0, 1.0, 0.1)
M = st.sidebar.slider("Hoop Mass (M) [kg]", 0.5, 5.0, 1.0, 0.1)

run_button = st.sidebar.button("Run and Update Animation")

# ------------------------------------------
# 3. Physics Engine (ODEs & Energy)
# ------------------------------------------
g = 9.81
L = 2 * R / np.pi

def calc_energy(theta, theta_dot, M, R):
    # I_P = 2*M*R^2 * (1 - 2*cos(theta)/pi)
    I_P = 2 * M * (R**2) * (1 - (2 * np.cos(theta) / np.pi))
    # E = 0.5 * I_P * theta_dot^2 + M*g*L*(1 - cos(theta))
    return 0.5 * I_P * (theta_dot**2) + M * g * L * (1 - np.cos(theta))

def ode_case_a(t, state):
    """ Case A: Incorrect - Neglecting Phantom Torque """
    theta, theta_dot = state
    num = -(np.sin(theta) / np.pi) * (g / R)
    den = 1 - (2 * np.cos(theta) / np.pi)
    theta_ddot = num / den
    return [theta_dot, theta_ddot]

def ode_case_b(t, state):
    """ Case B: Correct - Including Phantom Torque """
    theta, theta_dot = state
    num = -(np.sin(theta) / np.pi) * (theta_dot**2) - (np.sin(theta) / np.pi) * (g / R)
    den = 1 - (2 * np.cos(theta) / np.pi)
    theta_ddot = num / den
    return [theta_dot, theta_ddot]

# ------------------------------------------
# 4. Main Execution Block
# ------------------------------------------
st.title("Phantom Torque Physics Simulator")
st.markdown("Based on the paper: *Asymmetric rolling bodies and the phantom torque (Am. J. Phys. 78, 2010)*")

# Only calculate if button is pressed or if it's the first run (history is empty)
if run_button or len(st.session_state.history) == 0:
    
    # Setup initial conditions
    theta_0 = np.radians(theta_0_deg)
    state_0 = [theta_0, 0.0]
    
    # Time vector for integration (10 seconds, 250 frames for smooth animation)
    t_span = (0, 10)
    t_eval = np.linspace(t_span[0], t_span[1], 250)
    
    # Solve ODEs
    sol_A = solve_ivp(ode_case_a, t_span, state_0, t_eval=t_eval, method='RK45')
    sol_B = solve_ivp(ode_case_b, t_span, state_0, t_eval=t_eval, method='RK45')
    
    # Extract results
    theta_A, dot_A = sol_A.y
    theta_B, dot_B = sol_B.y
    
    # Calculate Energy errors
    E_A = calc_energy(theta_A, dot_A, M, R)
    E_B = calc_energy(theta_B, dot_B, M, R)
    
    E0 = E_A[0] # Initial energy is the same for both
    err_A = np.abs(E_A - E0)
    err_B = np.abs(E_B - E0)
    
    max_err_A = np.max(err_A)
    max_err_B = np.max(err_B)
    
    # Determine behavior description
    if max_err_A > 0.5:
        divergence_desc = "Severe Energy Violation & Chaos"
    elif max_err_A > 0.1:
        divergence_desc = "Noticeable Amplitude/Period Shift"
    else:
        divergence_desc = "Stable (Small Angle Approx. holds)"
        
    # Update History Data
    if run_button:
        new_record = {
            "Run ID": st.session_state.run_id,
            "θ₀ (deg)": theta_0_deg,
            "Radius (m)": R,
            "Mass (kg)": M,
            "Case A Max Error (J)": round(max_err_A, 5),
            "Case B Max Error (J)": round(max_err_B, 5),
            "Behavior Divergence": divergence_desc
        }
        st.session_state.history.append(new_record)
        st.session_state.run_id += 1

    # ------------------------------------------
    # 5. Plotly Animation Rendering
    # ------------------------------------------
    # Helper to generate arc geometry
    def get_hoop_coords(theta_val, R, L):
        # Center of the semicircular hoop rolls to x = R*theta
        x_O = R * theta_val
        y_O = R
        # Arc points (parameterized by angle phi)
        # Semicircle spans from pi to 2pi when theta=0
        # When rolled by theta, it rotates CCW by -theta (or CW by theta)
        phi = np.linspace(np.pi + theta_val, 2 * np.pi + theta_val, 50)
        x_arc = x_O + R * np.cos(phi)
        y_arc = y_O + R * np.sin(phi)
        
        # Center of mass coords
        x_cm = R * theta_val - L * np.sin(theta_val)
        y_cm = R - L * np.cos(theta_val)
        return x_arc, y_arc, x_cm, y_cm

    fig_anim = go.Figure()

    # Add floor line
    fig_anim.add_trace(go.Scatter(
        x=[-3*R, 3*R], y=[0, 0], 
        mode='lines', line=dict(color='black', width=3), 
        name='Floor', hoverinfo='none'
    ))

    # Initial frames setup
    x_arc_A, y_arc_A, x_cm_A, y_cm_A = get_hoop_coords(theta_A[0], R, L)
    x_arc_B, y_arc_B, x_cm_B, y_cm_B = get_hoop_coords(theta_B[0], R, L)

    # Trace: Case A (Red - No Phantom Torque)
    fig_anim.add_trace(go.Scatter(
        x=x_arc_A, y=y_arc_A, mode='lines', 
        line=dict(color='red', width=4, dash='dash'), name='Case A (Incorrect)'
    ))
    fig_anim.add_trace(go.Scatter(
        x=[x_cm_A], y=[y_cm_A], mode='markers', 
        marker=dict(color='red', size=8, symbol='x'), showlegend=False
    ))

    # Trace: Case B (Blue - With Phantom Torque)
    fig_anim.add_trace(go.Scatter(
        x=x_arc_B, y=y_arc_B, mode='lines', 
        line=dict(color='blue', width=4), name='Case B (Correct)'
    ))
    fig_anim.add_trace(go.Scatter(
        x=[x_cm_B], y=[y_cm_B], mode='markers', 
        marker=dict(color='blue', size=8, symbol='circle'), showlegend=False
    ))

    # Build Animation Frames
    frames = []
    for i in range(len(t_eval)):
        xa, ya, xca, yca = get_hoop_coords(theta_A[i], R, L)
        xb, yb, xcb, ycb = get_hoop_coords(theta_B[i], R, L)
        
        frame = go.Frame(
            data=[
                go.Scatter(x=[-3*R, 3*R], y=[0, 0]), # Keep floor static
                go.Scatter(x=xa, y=ya),
                go.Scatter(x=[xca], y=[yca]),
                go.Scatter(x=xb, y=yb),
                go.Scatter(x=[xcb], y=[ycb])
            ],
            name=f"fr{i}"
        )
        frames.append(frame)
        
    fig_anim.frames = frames

    # Animation Layout Configuration
    fig_anim.update_layout(
        xaxis=dict(range=[-2.5*R, 2.5*R], autorange=False, title="Horizontal Position (m)"),
        yaxis=dict(range=[-0.2*R, 2.5*R], autorange=False, scaleanchor="x", scaleratio=1),
        title=f"Live Physics Engine (Initial Angle: {theta_0_deg}°)",
        updatemenus=[dict(
            type="buttons",
            buttons=[
                dict(label="Play", method="animate", args=[None, dict(frame=dict(duration=30, redraw=False), fromcurrent=True, mode='immediate')]),
                dict(label="Pause", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate')])
            ],
            direction="left", pad={"r": 10, "t": 87}, showactive=False, x=0.1, xanchor="right", y=0, yanchor="top"
        )],
        sliders=[dict(
            steps=[dict(method='animate', args=[[f"fr{i}"], dict(mode='immediate', frame=dict(duration=30, redraw=False))], label=f"{t:.1f}s") for i, t in enumerate(t_eval)],
            active=0, transition=dict(duration=0), x=0.1, len=0.9, xanchor="left", y=0, yanchor="top"
        )],
        height=500
    )

    # ------------------------------------------
    # 6. Streamlit Layout Assembly
    # ------------------------------------------
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("Physical Animation")
        st.plotly_chart(fig_anim, use_container_width=True)

    with col2:
        st.subheader("Mechanical Energy Deviation")
        # Energy Chart
        fig_energy = go.Figure()
        fig_energy.add_trace(go.Scatter(x=t_eval, y=err_A, mode='lines', name='Case A Error (Missing Torque)', line=dict(color='red')))
        fig_energy.add_trace(go.Scatter(x=t_eval, y=err_B, mode='lines', name='Case B Error (Correct)', line=dict(color='blue')))
        fig_energy.update_layout(xaxis_title="Time (s)", yaxis_title="Energy Deviation |E(t) - E(0)| [Joules]", height=300, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_energy, use_container_width=True)

        # Dynamic Data Table
        st.subheader("Simulation History Logs")
        if st.session_state.history:
            df_history = pd.DataFrame(st.session_state.history)
            st.dataframe(df_history.set_index("Run ID"), use_container_width=True)
            
            if st.button("Clear History"):
                st.session_state.history = []
                st.session_state.run_id = 1
                st.rerun()