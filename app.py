import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import streamlit as st

# ------------------------------------------------------------------------------
# KONFIGURASI HALAMAN STREAMLIT
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="GEO5 Retaining Wall Designer",
    page_icon="🏗️",
    layout="wide"
)

# ------------------------------------------------------------------------------
# ENGINE KALKULASI GEO5 & SNI
# ------------------------------------------------------------------------------
class GEO5CantileverWallEngine:
    def __init__(self, geometry, elevations, soils, seismic, structural):
        self.geom = geometry
        self.elev = elevations
        self.soils = soils
        self.seismic = seismic
        self.struct = structural
        
        self.H_stem = self.elev['EL_top'] - (self.elev['EL_base'] + self.geom['h_f'])
        self.H_total = self.elev['EL_top'] - self.elev['EL_base']
        
    def _get_soil_at_elev(self, el):
        for layer in self.soils:
            if layer['top_el'] >= el >= layer['bot_el']:
                return layer
        return self.soils[-1]

    def calc_seismic_coefficients(self):
        kh = self.seismic['kh']
        kv = self.seismic['kv']
        
        theta = math.atan(kh / (1 - kv)) if (1 - kv) > 0 else 0
        phi = math.radians(self.soils[0]['phi'])
        delta = math.radians(self.soils[0]['delta'])
        beta = 0.0
        alpha = 0.0
        
        num_ka = math.sin(math.pi/2 + phi)**2
        den_ka = math.sin(math.pi/2)**2 * math.sin(math.pi/2 - delta) * (
            1 + math.sqrt((math.sin(phi + delta) * math.sin(phi - beta)) /
                          (math.sin(math.pi/2 - delta) * math.sin(math.pi/2 + beta)))
        )**2
        Ka = num_ka / den_ka if den_ka != 0 else 0.333
        
        num_kae = math.cos(phi - theta - alpha)**2
        den_kae = math.cos(theta) * (math.cos(alpha)**2) * math.cos(delta + alpha + theta) * (
            1 + math.sqrt((math.sin(phi + delta) * math.sin(phi - theta - beta)) /
                          (math.cos(delta + alpha + theta) * math.cos(beta - alpha)))
        )**2
        Kae = num_kae / den_kae if den_kae != 0 else Ka
        dKae = max(0.0, Kae - Ka)
        
        return Ka, Kae, dKae, theta

    def calculate(self):
        Ka, Kae, dKae, theta = self.calc_seismic_coefficients()
        gamma_w = 9.81
        
        n_steps = 100
        dz = self.H_total / n_steps
        P_A = 0.0
        M_PA = 0.0
        sigma_v_eff = 0.0
        
        gwl_active = min(self.elev['GWL_back'], self.elev['EL_drain'])
        
        for i in range(n_steps):
            z_curr = (i + 0.5) * dz
            el_curr = self.elev['EL_top'] - z_curr
            soil = self._get_soil_at_elev(el_curr)
            
            gamma_eff = (soil['gamma_sat'] - gamma_w) if el_curr < gwl_active else soil['gamma']
            sigma_v_eff += gamma_eff * dz
            
            ka_i = math.tan(math.radians(45 - soil['phi']/2))**2
            sigma_h_eff = max(0.0, sigma_v_eff * ka_i - 2 * soil['c'] * math.sqrt(ka_i))
            
            df = sigma_h_eff * dz
            P_A += df
            M_PA += df * (self.H_total - z_curr)

        y_PA = M_PA / P_A if P_A > 0 else self.H_total / 3.0
        
        gamma_avg = self.soils[0]['gamma']
        delta_P_AE = 0.5 * gamma_avg * (self.H_total**2) * (1 - self.seismic['kv']) * dKae
        y_delta_P_AE = 0.60 * self.H_total
        
        h_w_back = max(0.0, gwl_active - self.elev['EL_base'])
        h_w_toe = max(0.0, self.elev['GWL_toe'] - self.elev['EL_base'])
        U_back = 0.5 * gamma_w * (h_w_back**2)
        U_toe = 0.5 * gamma_w * (h_w_toe**2)
        U_net = max(0.0, U_back - U_toe)
        y_U_net = h_w_back / 3.0 if h_w_back > 0 else 0.0
        
        W_stem = (self.geom['g1'] + self.geom['stem_bot_w']) / 2.0 * self.H_stem * 24.0
        x_stem = self.geom['L_t'] + self.geom['stem_bot_w'] / 2.0
        
        W_base = self.geom['B'] * self.geom['h_f'] * 24.0
        x_base = self.geom['B'] / 2.0
        
        W_key = self.geom['b_key'] * self.geom['d_key'] * 24.0
        x_key = self.geom['x_key'] + self.geom['b_key'] / 2.0
        
        W_cf = (0.5 * self.geom['L_h'] * self.H_stem * self.geom['t_c'] * 24.0) / self.geom['s']
        x_cf = self.geom['L_t'] + self.geom['stem_bot_w'] + (self.geom['L_h'] / 3.0)
        
        W_wall_total = W_stem + W_base + W_key + W_cf
        M_R_wall = (W_stem * x_stem) + (W_base * x_base) + (W_key * x_key) + (W_cf * x_cf)
        
        W_soil_heel = self.geom['L_h'] * self.H_stem * self.soils[0]['gamma']
        x_soil_heel = self.geom['B'] - (self.geom['L_h'] / 2.0)
        M_R_soil = W_soil_heel * x_soil_heel
        
        u_avg = ((h_w_back + h_w_toe) / 2.0) * gamma_w
        U_buoyancy = u_avg * self.geom['B']
        
        V_eff = W_wall_total + W_soil_heel - U_buoyancy
        M_R_total = M_R_wall + M_R_soil - (U_buoyancy * self.geom['B'] / 2.0)
        
        P_I_wall = self.seismic['kh'] * W_wall_total
        y_I_wall = self.geom['h_f'] + (self.H_stem / 3.0)
        
        P_I_soil = self.seismic['kh'] * W_soil_heel
        y_I_soil = self.geom['h_f'] + (self.H_stem / 2.0)
        
        P_H_EQ = P_A + delta_P_AE + U_net + P_I_wall + P_I_soil
        M_O_EQ = (P_A * y_PA) + (delta_P_AE * y_delta_P_AE) + (U_net * y_U_net) + \
                 (P_I_wall * y_I_wall) + (P_I_soil * y_I_soil)
                 
        base_soil = self._get_soil_at_elev(self.elev['EL_base'])
        key_soil = self._get_soil_at_elev(self.elev['EL_key'])
        
        tan_phi_b = math.tan(math.radians(base_soil['phi']))
        R_h_base = V_eff * tan_phi_b + (base_soil['c'] * self.geom['B'])
        
        kp_key = math.tan(math.radians(45 + key_soil['phi']/2))**2
        gamma_key_eff = (key_soil['gamma_sat'] - gamma_w) if self.elev['EL_key'] < self.elev['GWL_toe'] else key_soil['gamma']
        
        P_pk_raw = 0.5 * gamma_key_eff * (self.geom['d_key']**2) * kp_key + \
                   2 * key_soil['c'] * math.sqrt(kp_key) * self.geom['d_key']
        P_pk_eff = 0.50 * P_pk_raw
        
        R_total_EQ = R_h_base + P_pk_eff
        
        SF_ot_EQ = M_R_total / M_O_EQ if M_O_EQ > 0 else 99.0
        SF_sl_EQ = R_total_EQ / P_H_EQ if P_H_EQ > 0 else 99.0
        
        e_EQ = (self.geom['B'] / 2.0) - ((M_R_total - M_O_EQ) / V_eff) if V_eff > 0 else 0.0
        B_prime = self.geom['B'] - 2.0 * e_EQ
        q_max_EQ = V_eff / B_prime if B_prime > 0 else 999.0
        q_ult = key_soil['c'] * 5.14 + V_eff * 0.5 * key_soil['gamma'] * kp_key
        SF_bc_EQ = q_ult / q_max_EQ if q_max_EQ > 0 else 0.0

        p_a_ult = (1.2 * P_A + 1.0 * delta_P_AE) / self.H_total
        M_u_stem = 0.10 * p_a_ult * (self.geom['s']**2)
        
        d_stem = (self.geom['stem_bot_w'] * 1000) - 75 - 8
        a = (1340 * self.struct['fy']) / (0.85 * self.struct['fc'] * 1000)
        phi_Mn_stem = 0.90 * 1340 * self.struct['fy'] * (d_stem - a/2) / 1e6
        
        return {
            'Ka': Ka, 'Kae': Kae, 'dKae': dKae, 'theta_deg': math.degrees(theta),
            'P_A': P_A, 'delta_P_AE': delta_P_AE, 'U_net': U_net,
            'P_I_wall': P_I_wall, 'P_I_soil': P_I_soil, 'P_H_EQ': P_H_EQ,
            'M_R_total': M_R_total, 'M_O_EQ': M_O_EQ,
            'SF_ot_EQ': SF_ot_EQ, 'SF_sl_EQ': SF_sl_EQ, 'SF_bc_EQ': SF_bc_EQ,
            'e_EQ': e_EQ, 'q_max_EQ': q_max_EQ,
            'M_u_stem': M_u_stem, 'phi_Mn_stem': phi_Mn_stem
        }

    def render_plot(self):
        fig, ax = plt.subplots(figsize=(10, 7), dpi=150)
        ax.set_aspect('equal', adjustable='box')
        
        soil_colors = ['#FCE8B2', '#E8C89C', '#D4AB7B', '#A88054']
        concrete_color = '#A6A6A6'
        water_color = '#0072BD'
        
        x_min, x_max = -2.0, self.geom['B'] + 3.0
        y_min, y_max = self.elev['EL_key'] - 1.0, self.elev['EL_top'] + 1.5
        
        for i, layer in enumerate(self.soils):
            top_y = min(layer['top_el'], self.elev['EL_top'])
            bot_y = max(layer['bot_el'], y_min)
            if top_y > bot_y:
                rect = patches.Rectangle((x_min, bot_y), x_max - x_min, top_y - bot_y,
                                         linewidth=0.5, edgecolor='#555555',
                                         facecolor=soil_colors[i % len(soil_colors)], zorder=1)
                ax.add_patch(rect)
                ax.text(x_max - 0.2, (top_y + bot_y)/2, f"{layer['name']}\nφ'={layer['phi']}°, c'={layer['c']}kPa",
                        verticalalignment='center', horizontalalignment='right',
                        fontsize=8, fontweight='bold', bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8), zorder=6)

        base_poly = [
            (0, self.elev['EL_base']),
            (self.geom['B'], self.elev['EL_base']),
            (self.geom['B'], self.elev['EL_base'] + self.geom['h_f']),
            (0, self.elev['EL_base'] + self.geom['h_f'])
        ]
        
        x_toe_end = self.geom['L_t']
        x_stem_back_bot = x_toe_end + self.geom['stem_bot_w']
        x_stem_back_top = x_toe_end + self.geom['g1']
        stem_poly = [
            (x_toe_end, self.elev['EL_base'] + self.geom['h_f']),
            (x_stem_back_bot, self.elev['EL_base'] + self.geom['h_f']),
            (x_stem_back_top, self.elev['EL_top']),
            (x_toe_end, self.elev['EL_top'])
        ]
        
        x_k1 = self.geom['x_key']
        x_k2 = x_k1 + self.geom['b_key']
        key_poly = [
            (x_k1, self.elev['EL_base']),
            (x_k2, self.elev['EL_base']),
            (x_k2, self.elev['EL_key']),
            (x_k1, self.elev['EL_key'])
        ]
        
        ax.add_patch(patches.Polygon(base_poly, closed=True, facecolor=concrete_color, edgecolor='black', lw=1.5, zorder=3))
        ax.add_patch(patches.Polygon(stem_poly, closed=True, facecolor=concrete_color, edgecolor='black', lw=1.5, zorder=3))
        ax.add_patch(patches.Polygon(key_poly, closed=True, facecolor=concrete_color, edgecolor='black', lw=1.5, zorder=3))
        
        cf_poly = [
            (x_stem_back_top, self.elev['EL_top']),
            (self.geom['B'], self.elev['EL_base'] + self.geom['h_f']),
            (x_stem_back_bot, self.elev['EL_base'] + self.geom['h_f'])
        ]
        ax.add_patch(patches.Polygon(cf_poly, closed=True, facecolor='none', edgecolor='black', linestyle='--', lw=1.2, zorder=4))

        gwl_b_y = min(self.elev['GWL_back'], self.elev['EL_drain'])
        ax.plot([x_stem_back_bot, x_max], [gwl_b_y, gwl_b_y], color=water_color, linestyle='-.', lw=1.5, zorder=5)
        
        gwl_t_y = self.elev['GWL_toe']
        ax.plot([x_min, 0], [gwl_t_y, gwl_t_y], color=water_color, linestyle='-.', lw=1.5, zorder=5)
        
        ax.plot([x_toe_end, x_stem_back_bot], [self.elev['EL_drain'], self.elev['EL_drain']], color='blue', lw=3, zorder=5)

        res = self.calculate()
        scale = 0.008
        
        y_pa = self.elev['EL_base'] + (self.H_total / 3.0)
        ax.annotate('', xy=(x_stem_back_bot, y_pa), xytext=(x_stem_back_bot + res['P_A']*scale, y_pa),
                    arrowprops=dict(facecolor='red', edgecolor='black', shrink=0, width=2, headwidth=7), zorder=7)

        y_dpae = self.elev['EL_base'] + (0.60 * self.H_total)
        ax.annotate('', xy=(x_stem_back_bot, y_dpae), xytext=(x_stem_back_bot + res['delta_P_AE']*scale, y_dpae),
                    arrowprops=dict(facecolor='magenta', edgecolor='black', shrink=0, width=2, headwidth=7), zorder=7)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("Lebar Struktur (m)", fontsize=9, fontweight='bold')
        ax.set_ylabel("Elevasi (m)", fontsize=9, fontweight='bold')
        ax.set_title("MODEL SKEMA GEO5 (SNI 8460:2017 / SNI 2847:2019)", fontsize=10, fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.5)
        
        return fig

# ------------------------------------------------------------------------------
# STREAMLIT INTERFACE (UI)
# ------------------------------------------------------------------------------
st.title("🏗️ GEO5 Retaining Wall Designer (SNI Compliant)")
st.caption("Aplikasi analisis stabilitas & struktur Dinding Penahan Tanah Cantilever / Counterfort")

st.sidebar.header("⚙️ Parameter Input")

# Geometry Input
st.sidebar.subheader("1. Geometri Dinding")
B = st.sidebar.number_input("Lebar Base Slab (B) [m]", value=4.80, step=0.1)
L_t = st.sidebar.number_input("Panjang Toe (L_t) [m]", value=1.00, step=0.1)
L_h = B - L_t - 0.60
h_f = st.sidebar.number_input("Tebal Base Slab (h_f) [m]", value=0.60, step=0.05)
d_key = st.sidebar.number_input("Kedalaman Shear Key (d_key) [m]", value=1.20, step=0.1)

# Elevations Input
st.sidebar.subheader("2. Elevasi Referensi")
EL_top = st.sidebar.number_input("Elevasi Top Dinding [m]", value=10.00, step=0.5)
EL_surf_toe = st.sidebar.number_input("Elevasi Muka Tanah Toe [m]", value=3.00, step=0.5)
EL_base = st.sidebar.number_input("Elevasi Dasar Base Slab [m]", value=1.90, step=0.5)
EL_key = EL_base - d_key
GWL_back = st.sidebar.number_input("Elevasi Air Belakang [m]", value=8.50, step=0.5)
GWL_toe = st.sidebar.number_input("Elevasi Air Depan Toe [m]", value=2.50, step=0.5)
EL_drain = st.sidebar.number_input("Elevasi Drainase Weep Holes [m]", value=3.50, step=0.5)

# Seismic Input
st.sidebar.subheader("3. Gempa Pseudostatik")
kh = st.sidebar.number_input("Koefisien Gempa Horizontal (kh)", value=0.201, step=0.01)

# Construct Data Dictionaries
geometry = {
    'g1': 0.40, 'stem_bot_w': 0.60, 'B': B, 'L_t': L_t, 'L_h': max(0.5, L_h),
    'h_f': h_f, 'd_key': d_key, 'b_key': 0.60, 'x_key': L_t + 0.40,
    's': 3.00, 't_c': 0.40
}

elevations = {
    'EL_top': EL_top, 'EL_surf_back': EL_top, 'EL_surf_toe': EL_surf_toe,
    'EL_base': EL_base, 'EL_key': EL_key, 'GWL_back': GWL_back,
    'GWL_toe': GWL_toe, 'EL_drain': EL_drain
}

soils = [
    {'name': 'Layer 1: Silt/Sand',   'top_el': 10.00, 'bot_el': 7.50, 'gamma': 18.0, 'gamma_sat': 19.5, 'phi': 32.0, 'c': 2.0,  'delta': 21.3},
    {'name': 'Layer 2: Sandy Clay',  'top_el': 7.50,  'bot_el': 3.00, 'gamma': 17.5, 'gamma_sat': 18.5, 'phi': 25.0, 'c': 10.0, 'delta': 16.7},
    {'name': 'Layer 3: Dense Sand',  'top_el': 3.00,  'bot_el': 1.10, 'gamma': 19.0, 'gamma_sat': 20.0, 'phi': 36.0, 'c': 0.0,  'delta': 24.0},
    {'name': 'Layer 4: Hard Clay',   'top_el': 1.10,  'bot_el': -5.0, 'gamma': 20.0, 'gamma_sat': 21.0, 'phi': 38.0, 'c': 25.0, 'delta': 25.0}
]

seismic = {'PGA': 0.35, 'Fa': 1.15, 'kh': kh, 'kv': 0.00}
structural = {'fc': 30.0, 'fy': 420.0}

# Run Engine
app = GEO5CantileverWallEngine(geometry, elevations, soils, seismic, structural)
res = app.calculate()

# Render Output Layout
col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("📊 Visualisasi Skematik Skala 1:1")
    fig = app.render_plot()
    st.pyplot(fig)

with col2:
    st.subheader("📋 Verifikasi Keamanan (SNI 8460:2017)")
    
    # Overturning
    sf_ot_status = "✅ PASSED" if res['SF_ot_EQ'] >= 1.10 else "❌ FAILED"
    st.metric("SF Guling / Overturning (EQ)", f"{res['SF_ot_EQ']:.2f}", delta=f"Syarat >= 1.10 ({sf_ot_status})")
    
    # Sliding
    sf_sl_status = "✅ PASSED" if res['SF_sl_EQ'] >= 1.10 else "❌ FAILED"
    st.metric("SF Geser / Sliding (EQ)", f"{res['SF_sl_EQ']:.2f}", delta=f"Syarat >= 1.10 ({sf_sl_status})")
    
    # Bearing Capacity
    sf_bc_status = "✅ PASSED" if res['SF_bc_EQ'] >= 1.15 else "❌ FAILED"
    st.metric("SF Daya Dukung Tanah (EQ)", f"{res['SF_bc_EQ']:.2f}", delta=f"Syarat >= 1.15 ({sf_bc_status})")
    
    st.divider()
    st.subheader("💪 Kapasitas Struktur Beton (SNI 2847:2019)")
    st.write(f"• **Momen Terfaktor (\(M_u\)):** {res['M_u_stem']:.2f} kNm/m")
    st.write(f"• **Kapasitas Lentur (\(\phi M_n\)):** {res['phi_Mn_stem']:.2f} kNm/m")
    if res['phi_Mn_stem'] >= res['M_u_stem']:
        st.success("Status Struktur: AMAN (Tulangan D16-150 Mampu Menahan Beban)")
    else:
        st.error("Status Struktur: PERLU REVISI TULANGAN")
