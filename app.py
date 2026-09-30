def print_geo5_report(self):
        """Mencetak laporan hasil verifikasi geoteknik & struktur ala GEO5."""
        res = self.calculate()
        print("="*83)
        print("                       GEO5-STYLE VERIFICATION REPORT")
        print("  MODULE: CANTILEVER WALL (MULTI-LAYER SOIL, DRAINAGE & PSEUDOSTATIC SEISMIC)")
        print("="*83)
        print("\n[1] ELEVATIONS & GEOMETRY")
        print(f"Top of Wall Elevation   (EL_top)      : {self.elev['EL_top']:.2f} m")
        # DIPERBAIKI: Menggunakan 'EL_surf_toe' alih-alih 'EL_surf,toe'
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
