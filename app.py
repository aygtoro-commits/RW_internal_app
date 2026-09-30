import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import streamlit as st

# ==========================================
# 1. CLASS ENGINE PERHITUNGAN (SNI 8460 & SNI 2847)
# ==========================================

class SoilLayer:
    def __init__(self, name, thickness, gamma_dry, gamma_sat, phi_deg, c, delta_deg=None):
        self.name = name
        self.thickness = thickness
        self.gamma_dry = gamma_dry
        self.gamma_sat = gamma_sat
        self.phi = math.radians(phi_deg)
        self.c = c
        self.delta = math.radians(delta_deg) if delta_deg is not None else (2/3 * self.phi)

class WallGeometry:
    def __init__(self, H_stem, t_stem_top, t_stem_bot, B_toe, B_heel, t_base,
                 has_shear_key=False, key_depth=0.0, key_width=0.0, key_pos_from_toe=0.0,
                 has_counterfort=False, cf_thick=0.0, cf_spacing=0.0):
        self.H_stem = H_stem
        self.t_stem_top = t_stem_top
        self.t_stem_bot = t_stem_bot
        self.B_toe = B_toe
        self.B_heel = B_heel
        self.t_base = t_base
        self.B_total = B_toe + t_stem_bot + B_heel
        self.H_total = H_stem + t_base
        
        self.has_shear_key = has_shear_key
        self.key_depth = key_depth
        self.key_width = key_width
        self.key_pos = key_pos_from_toe
        
        self.has_counterfort = has_counterfort
        self.cf_thick = cf_thick
        self.cf_spacing = cf_spacing

class DesignMaterials:
    def __init__(self, fc_MPa=25, fy_MPa=420, gamma_conc=24.0, cover=0.075):
        self.fc = fc_MPa
        self.fy = fy_MPa
        self.gamma_conc = gamma_conc
        self.cover = cover
        self.phi_flexure = 0.90
        self.phi_shear = 0.75

class RetainingWallEngine:
    def __init__(self, geometry, materials, layers, mat_depth=None, 
                 topo_type="flat", beta_deg=0.0, L_berm=0.0, kh=0.0, kv=0.0):
        self.geom = geometry
        self.mat = materials
        self.layers = layers
        self.mat_depth = mat_depth if mat_depth is not None else geometry.H_total
        self.topo_type = topo_type
        self.beta = math.radians(beta_deg)
        self.L_berm = L_berm
        self.kh = kh
        self.kv = kv
        self.theta = math.atan(kh / (1 - kv)) if (1 - kv) > 0 else 0.0

    def calc_ka_coulomb(self, phi, beta, delta, alpha=math.pi/2):
        if (alpha + beta) <= 0 or (alpha - delta) <= 0:
            return 0.33
        num = math.sin(alpha + phi)**2
        try:
            denom_term = math.sqrt((math.sin(phi + delta) * math.sin(phi - beta)) / 
                                   (math.sin(alpha - delta) * math.sin(alpha + beta)))
            denom = (math.sin(alpha)**2) * math.sin(alpha - delta) * ((1 + denom_term)**2)
            return num / denom if denom != 0 else 0.33
        except ValueError:
            return 0.33

    def calc_kae_mononobe_okabe(self, phi, beta, delta, theta, alpha=math.pi/2):
        if (phi - theta) <= beta:
            ka_stat = self.calc_ka_coulomb(phi, beta, delta, alpha)
            dKa_eq = 0.75 * self.kh
            return ka_stat + dKa_eq
        
        try:
            num = math.cos(phi - theta - (math.pi/2 - alpha))**2
            denom_term = math.sqrt((math.sin(phi + delta) * math.sin(phi - theta - beta)) / 
                                   (math.cos(delta + (math.pi/2 - alpha) + theta) * math.cos(beta - (math.pi/2 - alpha))))
            denom = math.cos(theta) * (math.cos(math.pi/2 - alpha)**2) * \
                    math.cos(delta + (math.pi/2 - alpha) + theta) * ((1 + denom_term)**2)
            return num / denom
        except ValueError:
            return self.calc_ka_coulomb(phi, beta, delta, alpha) + 0.75 * self.kh

    def calc_kp_rankine(self, phi):
        return math.tan(math.pi/4 + phi/2)**2

    def analyze_earth_pressures(self, dz=0.05):
        H = self.geom.H_total
        n_steps = max(10, int(H / dz))
        
        sigma_v_eff = 0.0
        Pa_static_total, Mo_static_total = 0.0, 0.0
        P_hydro_total, Mo_hydro_total = 0.0, 0.0
        
        z_curr = 0.0
        for _ in range(n_steps):
            z_mid = z_curr + dz/2
            arm = H - z_mid
            layer = self.get_layer_at_depth(z_mid)
            
            is_submerged = z_mid > self.mat_depth
            gamma_eff = (layer.gamma_sat - 9.81) if is_submerged else layer.gamma_dry
            sigma_v_eff += gamma_eff * dz
            
            beta_eff = self.beta if (self.topo_type == "infinite_slope" or 
                                    (self.topo_type == "berm_slope" and z_mid >= self.L_berm)) else 0.0
            
            Ka = self.calc_ka_coulomb(layer.phi, beta_eff, layer.delta)
            pa_eff = max(0.0, Ka * sigma_v_eff - 2 * layer.c * math.sqrt(Ka))
            
            Pa_static_total += pa_eff * dz
            Mo_static_total += (pa_eff * dz) * arm
            
            if is_submerged:
                u = 9.81 * (z_mid - self.mat_depth)
                P_hydro_total += u * dz
                Mo_hydro_total += (u * dz) * arm
                
            z_curr += dz

        top_layer = self.layers[0]
        Kae = self.calc_kae_mononobe_okabe(top_layer.phi, self.beta, top_layer.delta, self.theta)
        Ka_top = self.calc_ka_coulomb(top_layer.phi, self.beta, top_layer.delta)
        dKae = max(0.0, Kae - Ka_top)
        
        dPAE = 0.5 * (1 - self.kv) * top_layer.gamma_dry * (H**2) * dKae
        Mo_seismic = dPAE * (0.6 * H)
        
        return {
            "Pa_stat": Pa_static_total, "Mo_stat": Mo_static_total,
            "P_hydro": P_hydro_total,   "Mo_hydro": Mo_hydro_total,
            "dPAE": dPAE,               "Mo_seismic": Mo_seismic,
            "P_active_total_stat": Pa_static_total + P_hydro_total,
            "Mo_total_stat": Mo_static_total + Mo_hydro_total,
            "P_active_total_eq": Pa_static_total + P_hydro_total + dPAE,
            "Mo_total_eq": Mo_static_total + Mo_hydro_total + Mo_seismic
        }

    def get_layer_at_depth(self, z):
        accum = 0.0
        for l in self.layers:
            accum += l.thickness
            if z <= accum:
                return l
        return self.layers[-1]

    def calc_wall_weight_and_center(self):
        g, m = self.geom, self.mat
        
        W_stem_rect = g.t_stem_top * g.H_stem * m.gamma_conc
        x_stem_rect = g.B_toe + g.t_stem_top / 2
        
        W_stem_tri = 0.5 * (g.t_stem_bot - g.t_stem_top) * g.H_stem * m.gamma_conc
        x_stem_tri = g.B_toe + g.t_stem_top + (g.t_stem_bot - g.t_stem_top) / 3
        
        W_base = g.B_total * g.t_base * m.gamma_conc
        x_base = g.B_total / 2
        
        W_key = (g.key_width * g.key_depth * m.gamma_conc) if g.has_shear_key else 0.0
        x_key = g.key_pos + g.key_width/2 if g.has_shear_key else 0.0
        
        W_cf = (0.5 * g.B_heel * g.H_stem * g.cf_thick * m.gamma_conc / g.cf_spacing) if (g.has_counterfort and g.cf_spacing > 0) else 0.0
        x_cf = g.B_toe + g.t_stem_bot + g.B_heel / 3 if g.has_counterfort else 0.0
        
        top_layer = self.layers[0]
        W_soil_heel = g.B_heel * g.H_stem * top_layer.gamma_dry
        x_soil_heel = g.B_toe + g.t_stem_bot + g.B_heel / 2
        
        W_total = W_stem_rect + W_stem_tri + W_base + W_key + W_cf + W_soil_heel
        Mr_total = (W_stem_rect * x_stem_rect + W_stem_tri * x_stem_tri + 
                    W_base * x_base + W_key * x_key + W_cf * x_cf + W_soil_heel * x_soil_heel)
        
        return W_total, Mr_total

    def evaluate_stability(self):
        ep = self.analyze_earth_pressures()
        W_total, Mr_total = self.calc_wall_weight_and_center()
        g = self.geom
        
        base_layer = self.layers[-1]
        Kp_base = self.calc_kp_rankine(base_layer.phi)
        
        D_passive = g.t_base + (g.key_depth if g.has_shear_key else 0.0)
        Pp = 0.5 * base_layer.gamma_dry * (D_passive**2) * Kp_base + 2 * base_layer.c * math.sqrt(Kp_base) * D_passive
        
        SF_overturning_stat = Mr_total / ep["Mo_total_stat"] if ep["Mo_total_stat"] > 0 else 999.0
        delta_base = 0.8 * base_layer.phi
        F_slide_res_stat = W_total * math.tan(delta_base) + (base_layer.c * g.B_total) + Pp
        SF_sliding_stat = F_slide_res_stat / ep["P_active_total_stat"] if ep["P_active_total_stat"] > 0 else 999.0
        
        X_net_stat = (Mr_total - ep["Mo_total_stat"]) / W_total
        e_stat = (g.B_total / 2) - X_net_stat
        q_max_stat = (W_total / g.B_total) * (1 + (6 * abs(e_stat) / g.B_total))
        
        SF_overturning_eq = Mr_total / ep["Mo_total_eq"] if ep["Mo_total_eq"] > 0 else 999.0
        F_slide_res_eq = (W_total * (1 - self.kv)) * math.tan(delta_base) + (base_layer.c * g.B_total) + Pp
        SF_sliding_eq = F_slide_res_eq / ep["P_active_total_eq"] if ep["P_active_total_eq"] > 0 else 999.0
        
        X_net_eq = (Mr_total - ep["Mo_total_eq"]) / W_total
        e_eq = (g.B_total / 2) - X_net_eq
        q_max_eq = (W_total / g.B_total) * (1 + (6 * abs(e_eq) / g.B_total))
        
        return {
            "SF_overturning_stat": SF_overturning_stat, "SF_overturning_req_stat": 1.5,
            "SF_sliding_stat": SF_sliding_stat,         "SF_sliding_req_stat": 1.5,
            "e_stat": e_stat,                           "e_max_req": g.B_total / 6.0,
            "q_max_stat": q_max_stat,
            "SF_overturning_eq": SF_overturning_eq,     "SF_overturning_req_eq": 1.1,
            "SF_sliding_eq": SF_sliding_eq,             "SF_sliding_req_eq": 1.1,
            "e_eq": e_eq,                               "q_max_eq": q_max_eq
        }

    def design_reinforcement(self):
        ep = self.analyze_earth_pressures()
        g, m = self.geom, self.mat
        
        if not g.has_counterfort:
            Mu_stem_base = 1.4 * ep["Mo_stat"] + 1.0 * ep["Mo_seismic"]
            b = 1.0
            d_stem = max(0.1, g.t_stem_bot - m.cover - 0.012)
            As_stem_req = self.calc_rebar_area(Mu_stem_base, b, d_stem)
        else:
            p_design = 1.4 * (ep["Pa_stat"] / g.H_stem) if g.H_stem > 0 else 0.0
            Mu_stem_base = 0.10 * p_design * (g.cf_spacing**2)
            b = 1.0
            d_stem = max(0.1, g.t_stem_bot - m.cover - 0.010)
            As_stem_req = self.calc_rebar_area(Mu_stem_base, b, d_stem)

        As_min_shrink = 0.0018 * 1.0 * g.t_stem_bot
        As_stem_final = max(As_stem_req, As_min_shrink)

        W_total, _ = self.calc_wall_weight_and_center()
        q_toe_avg = (W_total / g.B_total) * 1.4
        Mu_toe = 0.5 * q_toe_avg * (g.B_toe**2)
        d_toe = max(0.1, g.t_base - m.cover - 0.010)
        As_toe_req = max(self.calc_rebar_area(Mu_toe, 1.0, d_toe), As_min_shrink)

        top_layer = self.layers[0]
        q_soil_heel = top_layer.gamma_dry * g.H_stem * 1.4
        Mu_heel = 0.5 * q_soil_heel * (g.B_heel**2)
        d_heel = max(0.1, g.t_base - m.cover - 0.010)
        As_heel_req = max(self.calc_rebar_area(Mu_heel, 1.0, d_heel), As_min_shrink)

        As_counterfort_tie = 0.0
        if g.has_counterfort:
            F_total_lateral = 1.4 * ep["Pa_stat"] * g.cf_spacing
            Mu_cf = F_total_lateral * (g.H_stem / 3)
            d_cf = max(0.1, g.B_heel * math.cos(math.atan(g.H_stem / g.B_heel)) - m.cover)
            As_counterfort_tie = self.calc_rebar_area(Mu_cf, g.cf_thick, d_cf)

        return {
            "Mu_stem_kNm": Mu_stem_base,  "As_stem_mm2_m": As_stem_final * 1e6,
            "Mu_toe_kNm": Mu_toe,         "As_toe_mm2_m": As_toe_req * 1e6,
            "Mu_heel_kNm": Mu_heel,       "As_heel_mm2_m": As_heel_req * 1e6,
            "As_cf_tie_mm2": As_counterfort_tie * 1e6 if g.has_counterfort else 0.0
        }

    def calc_rebar_area(self, Mu_kNm, b_m, d_m):
        Mu = Mu_kNm * 1000.0
        if Mu <= 0:
            return 0.0
        fc, fy, phi = self.mat.fc * 1e6, self.mat.fy * 1e6, self.mat.phi_flexure
        A = (phi * (fy**2)) / (1.7 * fc * b_m)
        B = -phi * fy * d_m
        C = Mu
        disc = B**2 - 4 * A * C
        if disc < 0:
            return (0.0018 * b_m * d_m) * 2.0
        return (-B - math.sqrt(disc)) / (2 * A)

# ==========================================
# 2. FUNGSI GAMBAR GRAFIS INTERAKTIF (MATPLOTLIB)
# ==========================================

def plot_retaining_wall(engine):
    fig, ax = plt.subplots(figsize=(10, 8))
    g = engine.geom
    
    # 1. Base Footing
    base_rect = patches.Rectangle((0, 0), g.B_total, g.t_base, linewidth=1.5, edgecolor='black', facecolor='#b0bec5', label="Beton Wall")
    ax.add_patch(base_rect)
    
    # 2. Shear Key (Jika Ada)
    if g.has_shear_key:
        key_rect = patches.Rectangle((g.key_pos, -g.key_depth), g.key_width, g.key_depth, linewidth=1.5, edgecolor='black', facecolor='#b0bec5')
        ax.add_patch(key_rect)
        
    # 3. Stem Concrete
    stem_x = [g.B_toe, g.B_toe + g.t_stem_bot, g.B_toe + g.t_stem_top, g.B_toe]
    stem_y = [g.t_base, g.t_base, g.t_base + g.H_stem, g.t_base + g.H_stem]
    ax.fill(stem_x, stem_y, edgecolor='black', facecolor='#b0bec5', linewidth=1.5)
    
    # 4. Counterfort Indicator (Jika Ada)
    if g.has_counterfort:
        cf_x = [g.B_toe + g.t_stem_bot, g.B_total, g.B_toe + g.t_stem_top]
        cf_y = [g.t_base, g.t_base, g.t_base + g.H_stem]
        ax.fill(cf_x, cf_y, edgecolor='black', facecolor='#cfd8dc', linestyle='--', alpha=0.6, label="Counterfort")

    # 5. Profil Lapisan Tanah Belakang (Backfill)
    colors_soil = ['#d7ccc8', '#ffe0b2', '#c8e6c9', '#bbdefb']
    z_accum = 0.0
    
    for i, layer in enumerate(engine.layers):
        z_start = z_accum
        z_end = z_accum + layer.thickness
        y_top = (g.t_base + g.H_stem) - z_start
        y_bot = (g.t_base + g.H_stem) - z_end
        
        soil_x = [g.B_toe + g.t_stem_bot, g.B_total + 3.0, g.B_total + 3.0, g.B_toe + g.t_stem_bot]
        soil_y = [y_bot, y_bot, y_top, y_top]
        
        color = colors_soil[i % len(colors_soil)]
        ax.fill(soil_x, soil_y, facecolor=color, alpha=0.6, label=f"Tanah: {layer.name}")
        
        # Garis Batas Lapisan
        ax.axhline(y=y_bot, color='brown', linestyle=':', linewidth=1)
        z_accum = z_end

    # Topografi Miring/Berm
    if engine.topo_type == "infinite_slope" and engine.beta > 0:
        x_slope = [g.B_toe + g.t_stem_top, g.B_total + 3.0, g.B_toe + g.t_stem_top]
        y_slope = [g.t_base + g.H_stem, g.t_base + g.H_stem + 3.0 * math.tan(engine.beta), g.t_base + g.H_stem]
        ax.fill(x_slope, y_slope, facecolor='#d7ccc8', alpha=0.8)

    # 6. Muka Air Tanah (MAT)
    mat_y = (g.t_base + g.H_stem) - engine.mat_depth
    ax.axhline(y=mat_y, color='#0288d1', linestyle='--', linewidth=2, label=f"MAT (z={engine.mat_depth:.1f}m)")
    ax.plot([g.B_total + 2.5], [mat_y], marker='v', color='#0288d1', markersize=8)

    # 7. SKETSA TULANGAN BETON (Baja)
    cov = engine.mat.cover
    # Tulangan Utama Stem (Tarik Belakang)
    ax.plot([g.B_toe + g.t_stem_bot - cov, g.B_toe + g.t_stem_top - cov],
            [g.t_base + cov, g.t_base + g.H_stem - cov], color='red', linewidth=3, label="Tulangan Utama (Rebar)")
    # Tulangan Heel (Atas)
    ax.plot([g.B_toe + g.t_stem_bot, g.B_total - cov],
            [g.t_base - cov, g.t_base - cov], color='red', linewidth=2.5)
    # Tulangan Toe (Bawah)
    ax.plot([cov, g.B_toe + g.t_stem_bot],
            [cov, cov], color='red', linewidth=2.5)

    # 8. Annotations & Formatting
    ax.set_xlim(-1.0, g.B_total + 3.0)
    ax.set_ylim(-g.key_depth - 1.0, g.H_total + 2.0)
    ax.set_aspect('equal')
    ax.set_title("Visualisasi Geometri, Lapisan Tanah, MAT & Sketsa Penulangan", fontsize=12, fontweight='bold')
    ax.set_xlabel("Jarak Horizontal (m)")
    ax.set_ylabel("Elevasi (m)")
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
    
    plt.tight_layout()
    return fig

# ==========================================
# 3. STREAMLIT WEB GUI APPLICATION
# ==========================================

st.set_page_config(page_title="Cantilever Wall Designer SNI", layout="wide")

st.title("🏗️ Aplikasi Perhitungan & Desain Cantilever Retaining Wall")
st.caption("Sesuai Standar SNI 8460:2017 (Geoteknik) & SNI 2847:2019 (Beton Struktural)")

# --- SIDEBAR INPUT PANEL ---
st.sidebar.header("📋 Input Parameter Geometri")
H_stem = st.sidebar.number_input("Tinggi Stem (m)", value=5.0, step=0.5)
t_stem_top = st.sidebar.number_input("Tebal Stem Atas (m)", value=0.3, step=0.05)
t_stem_bot = st.sidebar.number_input("Tebal Stem Bawah (m)", value=0.5, step=0.05)
B_toe = st.sidebar.number_input("Panjang Toe (m)", value=1.0, step=0.1)
B_heel = st.sidebar.number_input("Panjang Heel (m)", value=2.0, step=0.1)
t_base = st.sidebar.number_input("Tebal Footing Base (m)", value=0.6, step=0.05)

st.sidebar.subheader("Shear Key & Counterfort")
has_shear_key = st.sidebar.checkbox("Tambahkan Shear Key", value=True)
key_depth = st.sidebar.number_input("Kedalaman Key (m)", value=0.5) if has_shear_key else 0.0
key_width = st.sidebar.number_input("Lebar Key (m)", value=0.4) if has_shear_key else 0.0
key_pos = st.sidebar.number_input("Posisi Key dari Toe (m)", value=1.2) if has_shear_key else 0.0

has_counterfort = st.sidebar.checkbox("Gunakan Counterfort", value=False)
cf_thick = st.sidebar.number_input("Tebal Counterfort (m)", value=0.3) if has_counterfort else 0.0
cf_spacing = st.sidebar.number_input("Spasi Counterfort (m)", value=3.0) if has_counterfort else 0.0

st.sidebar.header("🌊 Topografi & Water Table")
topo_type_opt = st.sidebar.selectbox("Kondisi Topografi Belakang", ["Flat (Datar)", "Infinite Slope (Miring)", "Berm Slope"])
topo_type = "flat" if "Flat" in topo_type_opt else ("infinite_slope" if "Infinite" in topo_type_opt else "berm_slope")
beta_deg = st.sidebar.number_input("Sudut Miring Tanah (°)", value=10.0) if topo_type != "flat" else 0.0
L_berm = st.sidebar.number_input("Panjang Berm Datar (m)", value=1.5) if topo_type == "berm_slope" else 0.0

mat_depth = st.sidebar.number_input("Kedalaman Muka Air Tanah / MAT (m dari atas)", value=3.0, step=0.5)

st.sidebar.header("⚡ Beban Seismik (Gempa)")
kh = st.sidebar.number_input("Koefisien Gempa Horisontal (kh)", value=0.15, step=0.01)
kv = st.sidebar.number_input("Koefisien Gempa Vertikal (kv)", value=0.0, step=0.01)

st.sidebar.header("🧱 Material Beton & Baja")
fc_MPa = st.sidebar.number_input("Kuat Tekan Beton fc' (MPa)", value=25.0)
fy_MPa = st.sidebar.number_input("Kuat Leleh Baja fy (MPa)", value=420.0)

# --- PROFIL LAPISAN TANAH ---
st.sidebar.header("⛰️ Lapisan Tanah (Profil)")
num_layers = st.sidebar.number_input("Jumlah Lapisan Tanah", min_value=1, max_value=3, value=2)

layers = []
for i in range(num_layers):
    st.sidebar.markdown(f"**Lapisan {i+1}**")
    name = f"Lapisan {i+1}"
    thick = st.sidebar.number_input(f"Tebal L{i+1} (m)", value=3.0 if i==0 else 3.0, key=f"th_{i}")
    g_dry = st.sidebar.number_input(f"γ dry L{i+1} (kN/m³)", value=18.0, key=f"gd_{i}")
    g_sat = st.sidebar.number_input(f"γ sat L{i+1} (kN/m³)", value=19.0, key=f"gs_{i}")
    phi = st.sidebar.number_input(f"Sudut Geser ϕ L{i+1} (°)", value=30.0 if i==0 else 28.0, key=f"ph_{i}")
    c = st.sidebar.number_input(f"Kohesi c L{i+1} (kPa)", value=0.0 if i==0 else 5.0, key=f"c_{i}")
    layers.append(SoilLayer(name, thick, g_dry, g_sat, phi, c))

# --- EKSEKUSI CALCULATIONS ---
geom = WallGeometry(H_stem, t_stem_top, t_stem_bot, B_toe, B_heel, t_base,
                    has_shear_key, key_depth, key_width, key_pos,
                    has_counterfort, cf_thick, cf_spacing)
mats = DesignMaterials(fc_MPa, fy_MPa)
engine = RetainingWallEngine(geom, mats, layers, mat_depth, topo_type, beta_deg, L_berm, kh, kv)

stab = engine.evaluate_stability()
rebar = engine.design_reinforcement()

# --- MAIN PAGE DASHBOARD ---
tab1, tab2, tab3 = st.tabs(["🖼️ Visualisasi Model", "📊 Stabilitas Eksternal (SNI 8460)", "🧱 Desain Penulangan (SNI 2847)"])

with tab1:
    st.subheader("Model Geometry & Subsurface Conditions")
    fig = plot_retaining_wall(engine)
    st.pyplot(fig)

with tab2:
    st.subheader("Evaluasi Stabilitas Eksternal Dinding Penahan Tanah")
    
    col1, col2, col3 = st.columns(3)
    
    # Statis
    with col1:
        st.markdown("### 🔴 Kondisi Statis")
        sf_o_stat = stab["SF_overturning_stat"]
        st.metric("SF Guling (Statis)", f"{sf_o_stat:.2f}", delta="Aman (≥ 1.5)" if sf_o_stat >= 1.5 else "TIDAK AMAN")
        
        sf_s_stat = stab["SF_sliding_stat"]
        st.metric("SF Geser (Statis)", f"{sf_s_stat:.2f}", delta="Aman (≥ 1.5)" if sf_s_stat >= 1.5 else "TIDAK AMAN")
        
        e_stat = stab["e_stat"]
        e_max = stab["e_max_req"]
        st.metric("Eksentrisitas (e)", f"{e_stat:.3f} m", delta=f"Batas e ≤ {e_max:.3f} m")

    # Gempa / Seismik
    with col2:
        st.markdown("### ⚡ Kondisi Gempa (Pseudo-Statis)")
        sf_o_eq = stab["SF_overturning_eq"]
        st.metric("SF Guling (Gempa)", f"{sf_o_eq:.2f}", delta="Aman (≥ 1.1)" if sf_o_eq >= 1.1 else "TIDAK AMAN")
        
        sf_s_eq = stab["SF_sliding_eq"]
        st.metric("SF Geser (Gempa)", f"{sf_s_eq:.2f}", delta="Aman (≥ 1.1)" if sf_s_eq >= 1.1 else "TIDAK AMAN")

    # Tegangan Kontak
    with col3:
        st.markdown("### 🦶 Tegangan Kontak Tanah")
        st.metric("q max (Statis)", f"{stab['q_max_stat']:.2f} kPa")
        st.metric("q max (Gempa)", f"{stab['q_max_eq']:.2f} kPa")

with tab3:
    st.subheader("Kebutuhan Luas Tulangan Lentur Beton Bertulang")
    
    res_col1, res_col2, res_col3 = st.columns(3)
    
    with res_col1:
        st.info("**Stem (Dinding)**")
        st.write(f"Momen Desain ($M_u$): **{rebar['Mu_stem_kNm']:.2f} kNm/m**")
        st.write(f"Luas Tulangan ($A_s$): **{rebar['As_stem_mm2_m']:.0f} mm²/m**")
        st.caption("Gunakan misal: D19-100 atau D22-150")

    with res_col2:
        st.info("**Toe Slab (Pelat Depan)**")
        st.write(f"Momen Desain ($M_u$): **{rebar['Mu_toe_kNm']:.2f} kNm/m**")
        st.write(f"Luas Tulangan ($A_s$): **{rebar['As_toe_mm2_m']:.0f} mm²/m**")
        st.caption("Gunakan misal: D16-150")

    with res_col3:
        st.info("**Heel Slab (Pelat Belakang)**")
        st.write(f"Momen Desain ($M_u$): **{rebar['Mu_heel_kNm']:.2f} kNm/m**")
        st.write(f"Luas Tulangan ($A_s$): **{rebar['As_heel_mm2_m']:.0f} mm²/m**")
        st.caption("Gunakan misal: D19-150")

    if has_counterfort:
        st.warning(f"**Tulangan Penarik Counterfort:** As = {rebar['As_cf_tie_mm2']:.0f} mm² per counterfort")
