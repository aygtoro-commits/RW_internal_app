import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class GEO5CantileverWallEngine:
    """
    Engine Kalkulasi & Visualisasi Dinding Penahan Tanah Cantilever/Counterfort
    Acuan Metode:
    - GEO5 Retaining Wall Manual (Teori Tekanan Tanah & Skema Counterfort)
    - SNI 8460:2017 (Persyaratan Perancangan Geoteknik & Gempa Mononobe-Okabe)
    - SNI 2847:2019 (Persyaratan Beton Struktural)
    - AASHTO LRFD / PCA Retaining Wall (Mekanisme Shear Key & Counterfort Horizontal Flexure)
    """
    def __init__(self, geometry, elevations, soils, seismic, structural):
        self.geom = geometry
        self.elev = elevations
        self.soils = soils
        self.seismic = seismic
        self.struct = structural
        
        # Derivasi Dimensi Utama
        self.H_stem = self.elev['EL_top'] - (self.elev['EL_base'] + self.geom['h_f'])
        self.H_total = self.elev['EL_top'] - self.elev['EL_base']
        
    def _get_soil_at_elev(self, el):
        """Mendapatkan properti lapisan tanah pada elevasi tertentu."""
        for layer in self.soils:
            if layer['top_el'] >= el >= layer['bot_el']:
                return layer
        return self.soils[-1]

    def calc_seismic_coefficients(self):
        """Menghitung koefisien gempa Mononobe-Okabe (SNI 8460:2017 Pasal 9)."""
        kh = self.seismic['kh']
        kv = self.seismic['kv']
        
        theta = math.atan(kh / (1 - kv)) if (1 - kv) > 0 else 0
        phi = math.radians(self.soils[0]['phi'])
        delta = math.radians(self.soils[0]['delta'])
        beta = 0.0  # Permukaan tanah horizontal
        alpha = 0.0 # Dinding tegak
        
        # Ka (Coulomb Statis)
        num_ka = math.sin(math.pi/2 + phi)**2
        den_ka = math.sin(math.pi/2)**2 * math.sin(math.pi/2 - delta) * (
            1 + math.sqrt((math.sin(phi + delta) * math.sin(phi - beta)) /
                          (math.sin(math.pi/2 - delta) * math.sin(math.pi/2 + beta)))
        )**2
        Ka = num_ka / den_ka if den_ka != 0 else 0.333
        
        # Kae (Mononobe-Okabe Dynamic)
        num_kae = math.cos(phi - theta - alpha)**2
        den_kae = math.cos(theta) * (math.cos(alpha)**2) * math.cos(delta + alpha + theta) * (
            1 + math.sqrt((math.sin(phi + delta) * math.sin(phi - theta - beta)) /
                          (math.cos(delta + alpha + theta) * math.cos(beta - alpha)))
        )**2
        Kae = num_kae / den_kae if den_kae != 0 else Ka
        dKae = max(0.0, Kae - Ka)
        
        return Ka, Kae, dKae, theta

    def calculate(self):
        """Melakukan kalkulasi analisis stabilitas luar (ASD) dan struktur internal (LRFD)."""
        Ka, Kae, dKae, theta = self.calc_seismic_coefficients()
        gamma_w = 9.81
        
        # 1. Integrasi Tekanan Tanah Aktif Statis (Multi-Layer Integration)
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
        
        # 2. Inkremen Gempa Dinamis (Seed-Whitman at 0.60 H)
        gamma_avg = self.soils[0]['gamma']
        delta_P_AE = 0.5 * gamma_avg * (self.H_total**2) * (1 - self.seismic['kv']) * dKae
        y_delta_P_AE = 0.60 * self.H_total
        
        # 3. Gaya Hidrostatik Netto (Differential Water Level)
        h_w_back = max(0.0, gwl_active - self.elev['EL_base'])
        h_w_toe = max(0.0, self.elev['GWL_toe'] - self.elev['EL_base'])
        U_back = 0.5 * gamma_w * (h_w_back**2)
        U_toe = 0.5 * gamma_w * (h_w_toe**2)
        U_net = max(0.0, U_back - U_toe)
        y_U_net = h_w_back / 3.0 if h_w_back > 0 else 0.0
        
        # 4. Berat Struktur & Tanah Penahan
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
        
        # 5. Inersia Seismik Internal Structure & Backfill Soil
        P_I_wall = self.seismic['kh'] * W_wall_total
        y_I_wall = self.geom['h_f'] + (self.H_stem / 3.0)
        
        P_I_soil = self.seismic['kh'] * W_soil_heel
        y_I_soil = self.geom['h_f'] + (self.H_stem / 2.0)
        
        # 6. Total Beban Pendorong & Momen Guling
        P_H_EQ = P_A + delta_P_AE + U_net + P_I_wall + P_I_soil
        M_O_EQ = (P_A * y_PA) + (delta_P_AE * y_delta_P_AE) + (U_net * y_U_net) + \
                 (P_I_wall * y_I_wall) + (P_I_soil * y_I_soil)
                 
        # 7. Tahanan Geser (Base + Shear Key AASHTO Mechanics)
        base_soil = self._get_soil_at_elev(self.elev['EL_base'])
        key_soil = self._get_soil_at_elev(self.elev['EL_key'])
        
        tan_phi_b = math.tan(math.radians(base_soil['phi']))
        R_h_base = V_eff * tan_phi_b + (base_soil['c'] * self.geom['B'])
        
        kp_key = math.tan(math.radians(45 + key_soil['phi']/2))**2
        gamma_key_eff = (key_soil['gamma_sat'] - gamma_w) if self.elev['EL_key'] < self.elev['GWL_toe'] else key_soil['gamma']
        
        P_pk_raw = 0.5 * gamma_key_eff * (self.geom['d_key']**2) * kp_key + \
                   2 * key_soil['c'] * math.sqrt(kp_key) * self.geom['d_key']
        P_pk_eff = 0.50 * P_pk_raw  # Reduction factor 0.50
        
        R_total_EQ = R_h_base + P_pk_eff
        
        # 8. Verifikasi Keamanan ASD (SNI 8460:2017)
        SF_ot_EQ = M_R_total / M_O_EQ
        SF_sl_EQ = R_total_EQ / P_H_EQ
        
        e_EQ = (self.geom['B'] / 2.0) - ((M_R_total - M_O_EQ) / V_eff)
        B_prime = self.geom['B'] - 2.0 * e_EQ
        q_max_EQ = V_eff / B_prime if B_prime > 0 else 999.0
        q_ult = key_soil['c'] * 5.14 + V_eff * 0.5 * key_soil['gamma'] * kp_key
        SF_bc_EQ = q_ult / q_max_EQ if q_max_EQ > 0 else 0.0

        # 9. Verifikasi Struktur LRFD (SNI 2847:2019 / PCA Counterfort Flexure)
        p_a_ult = (1.2 * P_A + 1.0 * delta_P_AE) / self.H_total
        M_u_stem = 0.10 * p_a_ult * (self.geom['s']**2)  # Horizontal flexure M_neg = 1/10 p s^2
        
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

    def render_geo5_proportional_plot(self, save_path="geo5_retaining_wall_model.png"):
        """Menghasilkan Gambar Teknik Proporsional Skala 1:1 Bergaya UI GEO5."""
        fig, ax = plt.subplots(figsize=(12, 9), dpi=150)
        ax.set_aspect('equal', adjustable='box')
        
        soil_colors = ['#FCE8B2', '#E8C89C', '#D4AB7B', '#A88054']
        concrete_color = '#A6A6A6'
        water_color = '#0072BD'
        
        x_min, x_max = -2.0, self.geom['B'] + 3.0
        y_min, y_max = self.elev['EL_key'] - 1.0, self.elev['EL_top'] + 1.5
        
        # 1. Stratigrafi Tanah
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

        # 2. Struktur Beton
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
        
        # Counterfort
        cf_poly = [
            (x_stem_back_top, self.elev['EL_top']),
            (self.geom['B'], self.elev['EL_base'] + self.geom['h_f']),
            (x_stem_back_bot, self.elev['EL_base'] + self.geom['h_f'])
        ]
        ax.add_patch(patches.Polygon(cf_poly, closed=True, facecolor='none', edgecolor='black', linestyle='--', lw=1.2, zorder=4))
        ax.text(self.geom['B'] - 0.8, self.elev['EL_base'] + self.geom['h_f'] + 1.2, "Counterfort\n(s = 3.0 m)",
                fontsize=7, fontstyle='italic', bbox=dict(boxstyle="square,pad=0.1", fc="#FFFFE0", ec="orange"), zorder=6)

        # 3. Muka Air
        gwl_b_y = min(self.elev['GWL_back'], self.elev['EL_drain'])
        ax.plot([x_stem_back_bot, x_max], [gwl_b_y, gwl_b_y], color=water_color, linestyle='-.', lw=1.5, zorder=5)
        ax.text(x_max - 0.5, gwl_b_y + 0.15, f"GWL Back (+{gwl_b_y:.2f} m)", color=water_color, fontweight='bold', fontsize=8, zorder=6)
        
        gwl_t_y = self.elev['GWL_toe']
        ax.plot([x_min, 0], [gwl_t_y, gwl_t_y], color=water_color, linestyle='-.', lw=1.5, zorder=5)
        ax.text(x_min + 0.2, gwl_t_y + 0.15, f"GWL Toe (+{gwl_t_y:.2f} m)", color=water_color, fontweight='bold', fontsize=8, zorder=6)
        
        ax.plot([x_toe_end, x_stem_back_bot], [self.elev['EL_drain'], self.elev['EL_drain']], color='blue', lw=3, zorder=5)
        ax.text(x_toe_end - 0.8, self.elev['EL_drain'], "Weep Holes", color='blue', fontsize=7, fontweight='bold', zorder=6)

        # 4. Vektor Gaya
        res = self.calculate()
        scale = 0.008
        
        y_pa = self.elev['EL_base'] + (self.H_total / 3.0)
        ax.annotate('', xy=(x_stem_back_bot, y_pa), xytext=(x_stem_back_bot + res['P_A']*scale, y_pa),
                    arrowprops=dict(facecolor='red', edgecolor='black', shrink=0, width=2, headwidth=7), zorder=7)
        ax.text(x_stem_back_bot + res['P_A']*scale + 0.1, y_pa, f"P_A = {res['P_A']:.1f} kN/m", color='red', fontweight='bold', fontsize=8, zorder=7)

        y_dpae = self.elev['EL_base'] + (0.60 * self.H_total)
        ax.annotate('', xy=(x_stem_back_bot, y_dpae), xytext=(x_stem_back_bot + res['delta_P_AE']*scale, y_dpae),
                    arrowprops=dict(facecolor='magenta', edgecolor='black', shrink=0, width=2, headwidth=7), zorder=7)
        ax.text(x_stem_back_bot + res['delta_P_AE']*scale + 0.1, y_dpae, f"ΔP_AE = {res['delta_P_AE']:.1f} kN/m", color='magenta', fontweight='bold', fontsize=8, zorder=7)

        y_pk = (self.elev['EL_base'] + self.elev['EL_key']) / 2.0
        ax.annotate('', xy=(x_k1, y_pk), xytext=(x_k1 - 60*scale, y_pk),
                    arrowprops=dict(facecolor='green', edgecolor='black', shrink=0, width=2, headwidth=7), zorder=7)
        ax.text(x_k1 - 60*scale - 1.2, y_pk - 0.1, "P_pk (Key Pasif)", color='green', fontweight='bold', fontsize=8, zorder=7)

        # 5. Formasi Layout
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("Lebar Structure / Penampang (m)", fontsize=9, fontweight='bold')
        ax.set_ylabel("Elevasi (m)", fontsize=9, fontweight='bold')
        ax.set_title("MODEL SKEMA GEO5: CANTILEVER WALL WITH SHEAR KEY & COUNTERFORT\n(Proporsional Skala 1:1 - SNI 8460:2017 / SNI 2847:2019)", fontsize=10, fontweight='bold', pad=12)
        ax.grid(True, which='both', linestyle=':', color='gray', alpha=0.5)
        
        status_box = f"VERIFIKASI DESAIN (SNI 8460:2017):\n" \
                     f"• SF Guling (EQ) = {res['SF_ot_EQ']:.2f} >= 1.10 [{ 'OK' if res['SF_ot_EQ']>=1.10 else 'FAIL' }]\n" \
                     f"• SF Geser (EQ)  = {res['SF_sl_EQ']:.2f} >= 1.10 [{ 'OK' if res['SF_sl_EQ']>=1.10 else 'FAIL' }]\n" \
                     f"• SF Daya Dukung = {res['SF_bc_EQ']:.2f} >= 1.15 [{ 'OK' if res['SF_bc_EQ']>=1.15 else 'FAIL' }]\n" \
                     f"• Eksentrisitas e = {res['e_EQ']:.2f}m <= B/6 ({self.geom['B']/6:.2f}m)"
        ax.text(x_min + 0.2, y_max - 0.3, status_box, verticalalignment='top', horizontalalignment='left',
                fontsize=8, fontweight='bold', bbox=dict(boxstyle="round,pad=0.4", fc="#E6F2FF", ec="#0055A5", lw=1.5), zorder=8)

        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        print(f"[SUCCESS] Model grafik proporsional GEO5 telah berhasil dibuat: {save_path}")
        plt.show()

    def print_geo5_report(self):
        """Mencetak laporan hasil verifikasi geoteknik & struktur ala GEO5."""
        res = self.calculate()
        print("="*83)
        print("                       GEO5-STYLE VERIFICATION REPORT")
        print("  MODULE: CANTILEVER WALL (MULTI-LAYER SOIL, DRAINAGE & PSEUDOSTATIC SEISMIC)")
        print("="*83)
        print("\n[1] ELEVATIONS & GEOMETRY")
        print(f"Top of Wall Elevation   (EL_top)      : {self.elev['EL_top']:.2f} m")
        # SUDAH DIPERBAIKI: Kunci dictionary menggunakan 'EL_surf_toe'
        print(f"Toe Surface Elevation   (EL_surf_toe) : {self.elev['EL_surf_toe']:.2f} m")
        print(f"Base Slab Bottom Elev.  (EL_base)     : {self.elev['EL_base']:.2f} m")
        print(f"Shear Key Bottom Elev.  (EL_key)      : {self.elev['EL_key']:.2f} m (Depth d_key = {self.geom['d_key']:.2f} m)")
        print(f"Base Width              (B)           : {self.geom['B']:.2f} m (L_t = {self.geom['L_t']:.2f}m, L_h = {self.geom['L_h']:.2f}m)")
        print(f"Controlled Water Back   (GWL_back)    : {self.elev['EL_drain']:.2f} m (Controlled by Weep Holes)")

        print("\n[2] SEISMIC PARAMETERS (SNI 8460:2017 / SNI 1726:2019)")
        print(f"Horiz. Seismic Coeff.   (kh)          : {self.seismic['kh']:.3f}")
        print(f"Seismic Inertia Angle   (θ)           : {res['theta_deg']:.2f} °")
        print(f"Static Coeff. Active    (Ka)          : {res['Ka']:.3f}")
        print(f"Seismic Coeff. Active   (Kae)         : {res['Kae']:.3f}")
        print(f"Dynamic Increm. Coeff.  (ΔKae)        : {res['dKae']:.3f}")

        print("\n[3] INTEGRATED LATERAL & INERTIAL FORCES UNDER SEISMIC CONDITION")
        print(f"Static Earth Force      (P_A)         : {res['P_A']:.2f} kN/m")
        print(f"Dynamic Increm. Force   (ΔP_AE)       : {res['delta_P_AE']:.2f} kN/m")
        print(f"Hydrostatic Water Force (U_net)       : {res['U_net']:.2f} kN/m")
        print(f"Wall Inertia Force      (P_I,wall)    : {res['P_I_wall']:.2f} kN/m")
        print(f"Soil Heel Inertia Force (P_I,soil)    : {res['P_I_soil']:.2f} kN/m")
        print(f"Total Horiz. Seismic Push (P_H,EQ)    : {res['P_H_EQ']:.2f} kN/m")

        print("\n[4] EXTERNAL STABILITY CHECK (SNI 8460:2017 - SEISMIC ASD METHOD)")
        
        status_ot = "OK" if res['SF_ot_EQ'] >= 1.10 else "NOT SATISFIED"
        print(f"--- A. OVERTURNING STABILITY (SEISMIC) ---")
        print(f"Total Resisting Moment  (M_R)  = {res['M_R_total']:.2f} kNm/m")
        print(f"Total Overturning Moment(M_O)  = {res['M_O_EQ']:.2f} kNm/m")
        print(f"Calculated Safety Factor (SF)  = {res['SF_ot_EQ']:.2f} (Required >= 1.10) --> {status_ot}")

        status_sl = "OK" if res['SF_sl_EQ'] >= 1.10 else "NOT SATISFIED"
        print(f"\n--- B. SLIDING STABILITY (SEISMIC) ---")
        print(f"Calculated Safety Factor (SF)  = {res['SF_sl_EQ']:.2f} (Required >= 1.10) --> {status_sl}")

        status_bc = "OK" if res['SF_bc_EQ'] >= 1.15 else "NOT SATISFIED"
        print(f"\n--- C. BEARING CAPACITY CHECK (SEISMIC) ---")
        print(f"Resultant Eccentricity (e_EQ)  = {res['e_EQ']:.2f} m (Limit <= B/6 = {self.geom['B']/6:.2f} m)")
        print(f"Max Effective Stress (q_max,EQ)= {res['q_max_EQ']:.2f} kPa")
        print(f"Calculated Safety Factor (SF)  = {res['SF_bc_EQ']:.2f} (Required >= 1.15) --> {status_bc}")

        print("\n[5] INTERNAL STRUCTURAL DESIGN (SNI 2847:2019 - LRFD SEISMIC)")
        status_st = "OK" if res['phi_Mn_stem'] >= res['M_u_stem'] else "NOT SATISFIED"
        print(f"Ultimate Bending Moment (Mu_EQ) = {res['M_u_stem']:.2f} kNm/m")
        print(f"Design Capacity (φMn)           = {res['phi_Mn_stem']:.2f} kNm/m --> {status_st}")
        print("="*83)

# ------------------------------------------------------------------------------
# EKSEKUSI UTAMA (DATA DEMO RUN)
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    geometry = {
        'g1': 0.40,           # Lebar atas stem (m)
        'stem_bot_w': 0.60,   # Lebar bawah stem (m)
        'B': 4.80,            # Lebar total base (m)
        'L_t': 1.00,          # Panjang toe (m)
        'L_h': 3.20,          # Panjang heel (m)
        'h_f': 0.60,          # Tebal base slab (m)
        'd_key': 1.20,        # Kedalaman shear key (m)
        'b_key': 0.60,        # Lebar shear key (m)
        'x_key': 1.40,        # Jarak key dari toe (m)
        's': 3.00,            # Jarak antar counterfort (m)
        't_c': 0.40           # Tebal counterfort (m)
    }

    elevations = {
        'EL_top': 10.00,
        'EL_surf_back': 10.00,
        'EL_surf_toe': 3.00,  # Kunci nama konsisten EL_surf_toe
        'EL_base': 1.90,
        'EL_key': 0.70,
        'GWL_back': 8.50,
        'GWL_toe': 2.50,
        'EL_drain': 3.50      # Elevasi weep holes
    }

    soils = [
        {'name': 'Layer 1: Silt/Sand',   'top_el': 10.00, 'bot_el': 7.50, 'gamma': 18.0, 'gamma_sat': 19.5, 'phi': 32.0, 'c': 2.0,  'delta': 21.3},
        {'name': 'Layer 2: Sandy Clay',  'top_el': 7.50,  'bot_el': 3.00, 'gamma': 17.5, 'gamma_sat': 18.5, 'phi': 25.0, 'c': 10.0, 'delta': 16.7},
        {'name': 'Layer 3: Dense Sand',  'top_el': 3.00,  'bot_el': 1.10, 'gamma': 19.0, 'gamma_sat': 20.0, 'phi': 36.0, 'c': 0.0,  'delta': 24.0},
        {'name': 'Layer 4: Hard Clay',   'top_el': 1.10,  'bot_el': -5.0, 'gamma': 20.0, 'gamma_sat': 21.0, 'phi': 38.0, 'c': 25.0, 'delta': 25.0}
    ]

    seismic = {
        'PGA': 0.35,
        'Fa': 1.15,
        'kh': 0.201,   # 0.5 * Fa * PGA (SNI 8460:2017)
        'kv': 0.00
    }

    structural = {
        'fc': 30.0,    # MPa
        'fy': 420.0    # MPa
    }

    # Inisialisasi dan Eksekusi GEO5 Engine
    app = GEO5CantileverWallEngine(geometry, elevations, soils, seismic, structural)
    app.print_geo5_report()
    app.render_geo5_proportional_plot("geo5_retaining_wall_model.png")
