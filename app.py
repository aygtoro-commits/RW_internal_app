"""Single-file, auditable preliminary RC cantilever retaining-wall engine.

SI units only: m, kN, kPa, MPa, degrees.  All forces/moments are per metre run.
Reference hierarchy: SNI 8460:2017 -> SNI 2847:2019 -> US code/guidance ->
textbook -> GEO5 comparison only.  Run this file directly for an example.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import json
from math import atan, cos, floor, pi, radians, sin, sqrt, tan
from typing import Any, Literal


# ---------- References and audit trace ----------
SNI_8460 = "SNI 8460:2017"
SNI_2847 = "SNI 2847:2019"
ACI_318 = "ACI 318-19"
FHWA = "FHWA NHI-10-024, Earth Retaining Structures"
TEXTBOOK = "Das, Principles of Foundation Engineering"
GEO5 = "GEO5 Retaining Wall User Guide (comparison only)"
REF = {
    "active": (SNI_8460, TEXTBOOK), "seismic": (SNI_8460, FHWA),
    "water": (SNI_8460, FHWA), "stability": (SNI_8460,),
    "shear_key": (SNI_8460, FHWA, TEXTBOOK),
    "bearing": (SNI_8460, TEXTBOOK), "rc": (SNI_2847, ACI_318),
    "global": (SNI_8460, FHWA), "bearing_capacity": (SNI_8460, FHWA, TEXTBOOK), "seismic_water": (SNI_8460, FHWA, TEXTBOOK), "geo5": (GEO5,),
}


@dataclass(frozen=True)
class TraceItem:
    step: str; formula: str; values: dict[str, Any]; references: tuple[str, ...]; note: str = ""


@dataclass
class CalculationTrace:
    items: list[TraceItem] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add(self, step: str, formula: str, values: dict[str, Any], references: tuple[str, ...], note: str = "") -> None:
        self.items.append(TraceItem(step, formula, values, references, note))

    def warn(self, message: str) -> None:
        if message not in self.warnings: self.warnings.append(message)

    def as_dict(self) -> dict[str, Any]: return {"items": [asdict(x) for x in self.items], "warnings": self.warnings}


# ---------- Input model ----------
class TerrainKind(str, Enum):
    FLAT = "flat"; INFINITE_SLOPE = "infinite_slope"; FLAT_THEN_INFINITE_SLOPE = "flat_then_infinite_slope"


@dataclass(frozen=True)
class Terrain:
    kind: TerrainKind = TerrainKind.FLAT
    slope_deg: float = 0.0
    flat_length: float | None = None

    def __post_init__(self) -> None:
        if self.kind is TerrainKind.INFINITE_SLOPE and not 0 <= self.slope_deg < 89: raise ValueError("slope_deg must be [0,89)")
        if self.kind is TerrainKind.FLAT_THEN_INFINITE_SLOPE and (self.flat_length is None or self.flat_length < 0): raise ValueError("broken terrain needs flat_length >= 0")

    @property
    def beta_rad(self) -> float: return radians(0 if self.kind is TerrainKind.FLAT else self.slope_deg)


@dataclass(frozen=True)
class SoilLayer:
    thickness: float; gamma_dry: float; gamma_sat: float; phi_deg: float
    cohesion: float = 0.0; interface_delta_deg: float = 0.0; su_kpa: float = 0.0

    def __post_init__(self) -> None:
        if min(self.thickness, self.gamma_dry, self.gamma_sat) <= 0: raise ValueError("soil thickness/unit weights must be positive")
        if not 0 < self.phi_deg < 55: raise ValueError("phi_deg must be (0,55)")
        if self.su_kpa < 0: raise ValueError("undrained Su must be >= 0")


@dataclass(frozen=True)
class Groundwater:
    behind_depth: float | None = None   # depth below retained soil top
    front_depth: float | None = None    # depth below front soil top
    gamma_water: float = 9.81

    def __post_init__(self) -> None:
        # A negative depth is permitted: it represents free water above the
        # local ground level, e.g. a pond against the front wall face.
        if self.gamma_water <= 0: raise ValueError("gamma_water must be positive")


@dataclass(frozen=True)
class ShearKey:
    depth: float; thickness: float; x_from_toe: float

    def __post_init__(self) -> None:
        if min(self.depth, self.thickness) <= 0 or self.x_from_toe < 0: raise ValueError("invalid shear-key geometry")


@dataclass(frozen=True)
class Stiffener:
    """side=heel -> counterfort; side=toe -> buttress.

    From top_elevation to constant_depth_to_elevation the projection is
    depth_top. Below it, the projection tapers linearly to depth_base at base.
    """
    side: Literal["heel", "toe"]; spacing: float; thickness: float
    top_elevation: float; constant_depth_to_elevation: float; depth_top: float; depth_base: float

    def __post_init__(self) -> None:
        if self.side not in ("heel", "toe") or min(self.spacing, self.thickness, self.top_elevation) <= 0: raise ValueError("invalid stiffener")
        if not 0 <= self.constant_depth_to_elevation <= self.top_elevation or min(self.depth_top, self.depth_base) < 0: raise ValueError("invalid stiffener profile")


@dataclass(frozen=True)
class Concrete:
    fc_mpa: float = 28.0; fy_mpa: float = 420.0; unit_weight: float = 24.0
    cover_mm: float = 50.0; main_bar_dia_mm: float = 16.0


@dataclass(frozen=True)
class GlobalStabilityInput:
    factor_of_safety: float | None = None; required_factor_of_safety: float = 1.5; method: str = "external solver required"


@dataclass(frozen=True)
class RetainingWallInput:
    wall_height: float; stem_thickness_top: float; stem_thickness_base: float
    toe_width: float; heel_width: float; base_thickness: float
    base_top_elevation: float = 0.0; retained_top_elevation: float | None = None; front_soil_elevation: float = 0.0
    soils: tuple[SoilLayer, ...] = field(default_factory=tuple); front_soils: tuple[SoilLayer, ...] = field(default_factory=tuple)
    terrain: Terrain = field(default_factory=Terrain); groundwater: Groundwater = field(default_factory=Groundwater)
    surcharge_kpa: float = 0.0; base_friction_angle_deg: float | None = None; adhesion_kpa: float = 0.0
    allowable_bearing_kpa: float | None = None; kh: float = 0.0; kv: float = 0.0
    include_passive_front: bool = False; passive_reduction: float = .5
    shear_key: ShearKey | None = None; stiffener: Stiffener | None = None; concrete: Concrete = field(default_factory=Concrete)
    global_stability: GlobalStabilityInput = field(default_factory=GlobalStabilityInput)
    required_sliding_fs: float = 1.5; required_overturning_fs: float = 2.0; strength_load_factor: float = 1.6
    strength_mode: Literal["drained", "undrained"] = "drained"
    shear_strength_reduction: float = 1.0; base_interface_factor: float = 2/3; undrained_adhesion_factor: float = .5
    foundation_embedment_m: float = 0.0; bearing_safety_factor: float = 3.0
    pga_m_g: float | None = None; pga_reduction_factor: float = 1.0; return_period_years: float | None = None; vertical_to_horizontal_ratio: float = 0.0
    include_hydrodynamic_water: bool = False

    def __post_init__(self) -> None:
        if min(self.wall_height, self.stem_thickness_top, self.stem_thickness_base, self.base_thickness) <= 0 or min(self.toe_width, self.heel_width) < 0: raise ValueError("invalid wall dimensions")
        if not self.soils or sum(x.thickness for x in self.soils) + 1e-9 < self.wall_height: raise ValueError("retained-side soil layers must cover wall height")
        if self.retained_top_elevation is not None and abs(self.retained_top_elevation-self.base_top_elevation-self.wall_height) > 1e-6: raise ValueError("retained_top_elevation-base_top_elevation must equal wall_height")
        if not self.base_top_elevation-self.base_thickness <= self.front_soil_elevation <= self.base_top_elevation+self.wall_height: raise ValueError("front soil elevation must be between base-slab bottom and retained top")
        if self.front_soils and sum(x.thickness for x in self.front_soils) + 1e-9 < self.front_soil_height: raise ValueError("front-side layers must cover front soil height")
        if self.kh < 0 or not -.5 < self.kv < .5: raise ValueError("invalid pseudo-static coefficients")
        if self.shear_key and self.shear_key.x_from_toe > self.base_width: raise ValueError("shear key must lie within base")
        if self.stiffener:
            if self.stiffener.top_elevation > self.wall_height: raise ValueError("stiffener top exceeds wall")
            if max(self.stiffener.depth_top, self.stiffener.depth_base) > (self.heel_width if self.stiffener.side == "heel" else self.toe_width): raise ValueError("stiffener exceeds its projection")
        if self.strength_mode not in ("drained", "undrained"): raise ValueError("strength_mode must be drained or undrained")
        if not 0 < self.shear_strength_reduction <= 1 or not 0 <= self.base_interface_factor <= 1 or not 0 <= self.undrained_adhesion_factor <= 1: raise ValueError("invalid strength/interface reduction factor")
        if min(self.foundation_embedment_m, self.bearing_safety_factor, self.pga_reduction_factor) < 0: raise ValueError("invalid bearing or PGA input")

    @property
    def base_width(self) -> float: return self.toe_width+self.stem_thickness_base+self.heel_width
    @property
    def front_soil_height(self) -> float: return self.front_soil_elevation-self.base_top_elevation
    @property
    def front_soil_profile(self) -> tuple[SoilLayer, ...]: return self.front_soils or self.soils
    @property
    def design_kh(self) -> float: return self.pga_m_g*self.pga_reduction_factor if self.pga_m_g is not None else self.kh
    @property
    def design_kv(self) -> float: return self.design_kh*self.vertical_to_horizontal_ratio if self.pga_m_g is not None else self.kv


# ---------- Formula helpers ----------
def layer_at(layers: tuple[SoilLayer, ...], depth: float) -> SoilLayer:
    cumulative = 0.0
    for layer in layers:
        cumulative += layer.thickness
        if depth <= cumulative + 1e-9: return layer
    return layers[-1]


def rankine_ka(phi_deg: float, beta_rad: float = 0.0) -> float:
    phi = radians(phi_deg)
    if abs(beta_rad) < 1e-12: return (1-sin(phi))/(1+sin(phi))
    root = sqrt(max(cos(beta_rad)**2-cos(phi)**2, 0.0)); den = cos(beta_rad)*(cos(beta_rad)+root)
    if den <= 0: raise ValueError("invalid slope/soil combination")
    return cos(beta_rad)*(cos(beta_rad)-root)/den


def mononobe_okabe_kae(phi_deg: float, beta_rad: float, delta_deg: float, kh: float, kv: float) -> float:
    if kh == 0: return rankine_ka(phi_deg, beta_rad)
    phi, delta, theta = radians(phi_deg), radians(delta_deg), atan(kh/(1-kv))
    radicand = sin(phi+delta)*sin(phi-beta_rad-theta)/(cos(delta+theta)*cos(beta_rad-theta))
    if radicand < 0: raise ValueError("Mononobe-Okabe invalid for supplied parameters")
    value = cos(phi-theta)**2/(cos(theta)**2*cos(delta+theta)*(1+sqrt(radicand))**2)
    if value <= 0: raise ValueError("non-positive Mononobe-Okabe coefficient")
    return value


def design_shear_parameters(layer: SoilLayer, d: RetainingWallInput) -> tuple[float, float, float]:
    """Return design phi, cohesion, and interface delta after user reduction."""
    if d.strength_mode == "undrained":
        return 0.0, layer.su_kpa*d.shear_strength_reduction, 0.0
    eta=d.shear_strength_reduction
    phi=atan(eta*tan(radians(layer.phi_deg)))*180/3.141592653589793
    cohesion=eta*layer.cohesion
    delta=atan(eta*tan(radians(layer.interface_delta_deg)))*180/3.141592653589793
    return phi, cohesion, delta


# ---------- Result model and engine ----------
@dataclass(frozen=True)
class Force:
    name: str; horizontal_kn: float = 0.0; vertical_kn: float = 0.0; x_m: float = 0.0; y_m: float = 0.0

@dataclass(frozen=True)
class Check:
    name: str; demand: float; capacity: float | None; ratio_or_fs: float; required: float | None; ok: bool | None; units: str

@dataclass(frozen=True)
class RCDesign:
    component: str; mu_knm_per_m: float; vu_kn_per_m: float; effective_depth_mm: float
    as_required_mm2_per_m: float; as_provided_mm2_per_m: float; bar_diameter_mm: float; spacing_mm: int
    phi_vc_kn_per_m: float; flexure_ok: bool; shear_ok: bool; tension_face: str


@dataclass
class CalculationResult:
    input: RetainingWallInput; forces: list[Force]; checks: list[Check]; reinforcement: list[RCDesign]; trace: CalculationTrace
    @property
    def passed(self) -> bool: return all(c.ok is not False for c in self.checks)
    def as_dict(self) -> dict[str, Any]: return {"passed": self.passed, "forces": [asdict(x) for x in self.forces], "checks": [asdict(x) for x in self.checks], "reinforcement": [asdict(x) for x in self.reinforcement], "trace": self.trace.as_dict()}


class RetainingWallEngine:
    def run(self, d: RetainingWallInput) -> CalculationResult:
        trace = CalculationTrace(); beta = d.terrain.beta_rad
        if d.terrain.kind is TerrainKind.FLAT_THEN_INFINITE_SLOPE:
            trace.warn("Broken terrain is conservatively approximated as infinite slope for lateral pressure; final design needs a rigorous wedge model.")
        forces = self._lateral(d, beta, trace)
        vertical = self._vertical(d, beta, trace); forces += vertical
        if d.design_kh:
            wc = sum(x.vertical_kn for x in vertical if "self weight" in x.name or "stiffener" in x.name)
            forces.append(Force("seismic inertia of concrete wall", d.design_kh*wc, 0, 0, d.base_thickness+d.wall_height/2))
            trace.add("Wall pseudo-static inertia", "Fi=kh*W_concrete", {"PGA_m_g":d.pga_m_g,"PGA_reduction":d.pga_reduction_factor,"return_period_years":d.return_period_years,"kh":d.design_kh,"Fi_kN_per_m":d.design_kh*wc}, REF["seismic"], "kh is derived from PGAm times reduction factor when PGAm is supplied.")
        checks = self._stability(d, forces, trace); reinforcement = self._rc(d, forces, trace)
        g = d.global_stability
        if g.factor_of_safety is None:
            checks.append(Check("global stability (external)",0,None,0,g.required_factor_of_safety,None,"FS")); trace.warn("Global stability is NOT checked; supply independent LE/FEM FS.")
        else:
            checks.append(Check("global stability (external)",g.required_factor_of_safety,g.factor_of_safety,g.factor_of_safety,g.required_factor_of_safety,g.factor_of_safety>=g.required_factor_of_safety,"FS"))
        trace.add("Global stability interface", "FS_external >= FS_required", {"FS":g.factor_of_safety,"required_FS":g.required_factor_of_safety,"method":g.method}, REF["global"])
        return CalculationResult(d, forces, checks, reinforcement, trace)

    def _lateral(self, d: RetainingWallInput, beta: float, t: CalculationTrace) -> list[Force]:
        h, step, z, sv = d.wall_height, min(.025,d.wall_height/400), 0., 0.; ps=ms=pq=mq=pe=me=0.
        while z < h-1e-10:
            dz=min(step,h-z); mid=z+dz/2; soil=layer_at(d.soils,mid)
            phi,cohesion,delta=design_shear_parameters(soil,d); gamma=soil.gamma_sat if d.strength_mode=="undrained" else (max(.01,soil.gamma_sat-d.groundwater.gamma_water) if d.groundwater.behind_depth is not None and mid>d.groundwater.behind_depth else soil.gamma_dry)
            svm=sv+gamma*dz/2; sv+=gamma*dz; ka=1.0 if d.strength_mode=="undrained" else rankine_ka(phi,beta); pressure=max(0.,ka*svm-2*cohesion*sqrt(ka)); y=h-mid
            ps+=pressure*dz; ms+=pressure*dz*y; pq+=ka*d.surcharge_kpa*dz; mq+=ka*d.surcharge_kpa*dz*y
            if d.design_kh and d.strength_mode=="drained": inc=max(0.,(mononobe_okabe_kae(phi,beta,delta,d.design_kh,d.design_kv)-ka)*svm); pe+=inc*dz; me+=inc*dz*y
            z+=dz
        out=[Force("active effective-soil pressure",ps,0,0,ms/ps if ps else 0),Force("surcharge lateral pressure",pq,0,0,mq/pq if pq else 0)]
        t.add("Active earth pressure","sigma_h=max(0,Ka*sigma_v-2*c*sqrt(Ka)); P=integral(sigma_h dz)",{"Psoil_kN_per_m":ps,"Psurcharge_kN_per_m":pq,"strength_mode":d.strength_mode,"shear_reduction":d.shear_strength_reduction,"terrain":d.terrain.kind.value},REF["active"])
        t.add("Shear-strength reduction","tan(phi_design)=eta*tan(phi_input); c_design=eta*c_input; undrained: Su_design=eta*Su",{"eta":d.shear_strength_reduction,"mode":d.strength_mode},REF["stability"])
        if d.design_kh and d.strength_mode=="drained": out.append(Force("seismic earth-pressure increment",pe,0,0,me/pe if pe else 0)); t.add("Seismic earth pressure","Delta sigma_h=(Kae-Ka)*sigma_v; Kae=Mononobe-Okabe",{"kh":d.design_kh,"kv":d.design_kv,"DeltaP_kN_per_m":pe},REF["seismic"])
        elif d.design_kh: t.warn("Mononobe-Okabe is not used for undrained/cohesive total-stress analysis; obtain a project-specific seismic earth-pressure model.")
        if d.groundwater.behind_depth is not None and d.groundwater.behind_depth<h:
            hw=h-d.groundwater.behind_depth; p=.5*d.groundwater.gamma_water*hw**2; out.append(Force("behind hydrostatic pressure",p,0,0,hw/3)); t.add("Behind hydrostatic pressure","Pw=0.5*gamma_w*hw^2",{"Pw_kN_per_m":p},REF["water"])
        if d.groundwater.front_depth is not None:
            hw=max(0.,d.front_soil_height-d.groundwater.front_depth)
            if hw: p=.5*d.groundwater.gamma_water*hw**2; out.append(Force("front hydrostatic resistance",-p,0,0,hw/3)); t.add("Front hydrostatic pressure","Pw=0.5*gamma_w*hw^2 (resisting)",{"Pw_kN_per_m":p},REF["water"])
        if d.include_hydrodynamic_water and d.design_kh:
            hb=max(0.,h-(d.groundwater.behind_depth if d.groundwater.behind_depth is not None else h)); hf=max(0.,d.front_soil_height-(d.groundwater.front_depth if d.groundwater.front_depth is not None else d.front_soil_height)); pb=7/12*d.design_kh*d.groundwater.gamma_water*hb**2; pf=7/12*d.design_kh*d.groundwater.gamma_water*hf**2
            if pb: out.append(Force("seismic hydrodynamic water behind",pb,0,0,.4*hb))
            if pf: out.append(Force("seismic hydrodynamic water front",-pf,0,0,.4*hf))
            t.add("Hydrodynamic water","Pwd=(7/12)*kh*gamma_w*Hw^2; y=0.4Hw",{"Pbehind_kN_per_m":pb,"Pfront_kN_per_m":pf},REF["seismic_water"],"Westergaard-type simplified free-water increment; not excess pore pressure in soil.")
        return out

    def _vertical(self,d:RetainingWallInput,beta:float,t:CalculationTrace)->list[Force]:
        c,b,h=d.concrete,d.base_width,d.wall_height; stem_area=h*(d.stem_thickness_base+d.stem_thickness_top)/2
        sx=d.toe_width+(d.stem_thickness_base**2+d.stem_thickness_base*d.stem_thickness_top+d.stem_thickness_top**2)/(3*(d.stem_thickness_base+d.stem_thickness_top))
        out=[Force("base slab self weight",0,c.unit_weight*b*d.base_thickness,b/2),Force("stem self weight",0,c.unit_weight*stem_area,sx)]
        w=m=0.; n=100; dx=d.heel_width/n if d.heel_width else 0.
        for i in range(n):
            xr=(i+.5)*dx; column=h+tan(beta)*xr; wi=layer_at(d.soils,min(column/2,h-1e-9)).gamma_dry*column*dx; x=d.toe_width+d.stem_thickness_base+xr; w+=wi;m+=wi*x
        if w: out.append(Force("soil over heel",0,w,m/w))
        if d.surcharge_kpa and d.heel_width: out.append(Force("vertical surcharge over heel",0,d.surcharge_kpa*d.heel_width,d.toe_width+d.stem_thickness_base+d.heel_width/2))
        if d.shear_key: out.append(Force("shear-key self weight",0,c.unit_weight*d.shear_key.depth*d.shear_key.thickness,d.shear_key.x_from_toe))
        if d.stiffener:
            s=d.stiffener; area=s.depth_top*(s.top_elevation-s.constant_depth_to_elevation)+(s.depth_top+s.depth_base)*s.constant_depth_to_elevation/2; wi=c.unit_weight*s.thickness*area/s.spacing; x=d.toe_width+d.stem_thickness_base+s.depth_base/2 if s.side=="heel" else max(0,d.toe_width-s.depth_base/2); out.append(Force(f"{s.side} stiffener self weight (per m wall)",0,wi,x)); t.add("Stiffener geometry","A=d_top*(z_top-z_const)+(d_top+d_base)*z_const/2",{"side":s.side,"profile_area_m2":area,"weight_kN_per_m":wi},REF["rc"])
        t.add("Vertical weights","W=gamma*area; soil-over-heel integrated by terrain profile",{"base_width_m":b},REF["stability"],"No base uplift/drainage correction.")
        if d.groundwater.behind_depth is not None: t.warn("Assess saturation, buoyancy, drainage and uplift beneath heel separately; heel soil weight uses gamma_dry.")
        return out

    def _stability(self,d:RetainingWallInput,f:list[Force],t:CalculationTrace)->list[Check]:
        hor=[x for x in f if x.horizontal_kn]; ver=[x for x in f if x.vertical_kn]; drive=sum(max(0,x.horizontal_kn) for x in hor); front=-sum(min(0,x.horizontal_kn) for x in hor); n=sum(x.vertical_kn for x in ver)
        founding=layer_at(d.soils,d.wall_height+d.base_thickness); phi_soil,c_soil,_=design_shear_parameters(founding,d); phi=d.base_friction_angle_deg if d.base_friction_angle_deg is not None else atan(d.base_interface_factor*tan(radians(phi_soil)))*180/3.141592653589793; adhesion=d.adhesion_kpa if d.adhesion_kpa else d.undrained_adhesion_factor*c_soil if d.strength_mode=="undrained" else d.base_interface_factor*c_soil; friction=n*tan(radians(phi))+adhesion*d.base_width; passive=0.
        if d.include_passive_front and d.front_soil_height:
            soil=d.front_soil_profile[0]; pp,cp,_=design_shear_parameters(soil,d); kp=1. if d.strength_mode=="undrained" else (1+sin(radians(pp)))/(1-sin(radians(pp))); passive+=d.passive_reduction*(.5*kp*soil.gamma_dry*d.front_soil_height**2+2*cp*sqrt(kp)*d.front_soil_height)
            if d.shear_key:
                soil=d.front_soil_profile[-1];pp,cp,_=design_shear_parameters(soil,d);kp=1. if d.strength_mode=="undrained" else (1+sin(radians(pp)))/(1-sin(radians(pp))); q_key=n/d.base_width
                key_passive=kp*(q_key*d.shear_key.depth+.5*soil.gamma_dry*d.shear_key.depth**2)+2*cp*sqrt(kp)*d.shear_key.depth; passive+=d.passive_reduction*key_passive
                t.add("Shear-key passive resistance","Pp,key=Kp*(sigma_v,key*Dkey+0.5*gamma*Dkey^2)+2*c*sqrt(Kp)*Dkey",{"sigma_v_key_kPa":q_key,"key_depth_m":d.shear_key.depth,"Kp":kp,"Pp_key_unreduced_kN_per_m":key_passive,"reduction":d.passive_reduction},REF["shear_key"],"Uses average base contact pressure as overburden at key; confirm local stress/soil disturbance and key location for final design.")
        resistance=friction+passive+front; fs=resistance/drive if drive else float("inf"); mo=sum(max(0,x.horizontal_kn)*x.y_m for x in hor); mr=sum(x.vertical_kn*x.x_m for x in ver)+sum(-min(0,x.horizontal_kn)*x.y_m for x in hor); fsot=mr/mo if mo else float("inf"); xr=(mr-mo)/n if n else 0.; e=xr-d.base_width/2; q=n/d.base_width; qtoe=q*(1-6*e/d.base_width); qheel=q*(1+6*e/d.base_width)
        beff=max(.01,d.base_width-2*abs(e)); qn=1. if phi_soil<1e-6 else __import__('math').exp(__import__('math').pi*tan(radians(phi_soil)))*tan(radians(45)+radians(phi_soil)/2)**2; nc=5.14 if phi_soil<1e-6 else (qn-1)/tan(radians(phi_soil)); ng=0. if phi_soil<1e-6 else 2*(qn+1)*tan(radians(phi_soil)); q0=founding.gamma_dry*d.foundation_embedment_m; qult=c_soil*nc+q0*qn+.5*founding.gamma_dry*beff*ng; qallow_auto=qult/d.bearing_safety_factor; qallow=d.allowable_bearing_kpa if d.allowable_bearing_kpa is not None else qallow_auto
        t.add("Sliding stability","FS=(N*tan(delta_base)+c_a*B+Ppassive+Pfront-water)/Pdrive",{"FS_sliding":fs,"driving_kN_per_m":drive,"resistance_kN_per_m":resistance,"base_friction_angle_deg":phi,"base_adhesion_kPa":adhesion,"foundation_layer_phi_deg":phi_soil,"foundation_layer_c_or_Su_kPa":c_soil},REF["stability"]); t.add("Bearing capacity","qult=c*Nc+q*Nq+0.5*gamma*B'*Ngamma; B'=B-2|e|",{"B_effective_m":beff,"Nc":nc,"Nq":qn,"Ngamma":ng,"qult_kPa":qult,"FS_bearing":d.bearing_safety_factor,"qallow_kPa":qallow,"qtoe_kPa":qtoe,"qheel_kPa":qheel},REF["bearing_capacity"],"Strip footing, homogeneous governing founding layer, Meyerhof effective-width concept. Settlement, weak-layer and slope effects are not included.")
        if not d.include_passive_front: t.warn("Passive front resistance is excluded by default; enable only when permanent cover/development are assured.")
        if d.include_passive_front and d.groundwater.front_depth is not None: t.warn("Front passive resistance uses gamma_dry; assess water/effective-stress effect before relying on it.")
        out=[Check("sliding",drive,resistance,fs,d.required_sliding_fs,fs>=d.required_sliding_fs,"FS"),Check("overturning",mo,mr,fsot,d.required_overturning_fs,fsot>=d.required_overturning_fs,"FS"),Check("resultant within middle third",abs(e),d.base_width/6,abs(e)/(d.base_width/6),1.,qtoe>=0 and qheel>=0,"m")]
        out.append(Check("allowable bearing",max(qtoe,qheel),qallow,max(qtoe,qheel)/qallow,1.,max(qtoe,qheel)<=qallow and min(qtoe,qheel)>=0,"kPa"))
        return out

    def _strip(self,name:str,mu:float,vu:float,h_m:float,c:Concrete,t:CalculationTrace,face:str)->RCDesign:
        b,h=1000.,h_m*1000; d=h-c.cover_mm-c.main_bar_dia_mm/2
        if d<=0: raise ValueError(f"{name}: nonpositive effective depth")
        mn=abs(mu)*1e6/.9; coef=c.fy_mpa**2/(1.7*c.fc_mpa*b); disc=(c.fy_mpa*d)**2-4*coef*mn; asflex=float("inf") if disc<0 else (c.fy_mpa*d-sqrt(disc))/(2*coef); ar=max(asflex,.0018*b*h); abar=pi*c.main_bar_dia_mm**2/4; spacing=50 if ar==float("inf") else max(50,min(300,floor(1000*abar/ar))); aprob=1000*abar/spacing; phivc=.75*.17*sqrt(c.fc_mpa)*b*d/1000; r=RCDesign(name,abs(mu),abs(vu),d,ar,aprob,c.main_bar_dia_mm,spacing,phivc,ar!=float("inf") and aprob>=ar,abs(vu)<=phivc,face)
        t.add(f"RC {name} flexure","phi*Mn>=Mu; a=As*fy/(0.85*fc'*b); Mn=As*fy*(d-a/2)",{"Mu_kNm_per_m":abs(mu),"As_required_mm2_per_m":ar,"bar":f"D{c.main_bar_dia_mm}@{spacing}","ok":r.flexure_ok},REF["rc"],"Preliminary singly reinforced 1 m strip."); t.add(f"RC {name} one-way shear","phi*Vc=.75*.17*sqrt(fc')*b*d",{"Vu_kN_per_m":abs(vu),"phiVc_kN_per_m":phivc,"ok":r.shear_ok},REF["rc"])
        return r

    def _rc(self,d:RetainingWallInput,f:list[Force],t:CalculationTrace)->list[RCDesign]:
        hor=[x for x in f if x.horizontal_kn]; ver=[x for x in f if x.vertical_kn]; lm=sum(max(0,x.horizontal_kn)*x.y_m for x in hor); lf=sum(max(0,x.horizontal_kn) for x in hor); out=[self._strip("stem",d.strength_load_factor*lm,d.strength_load_factor*lf,d.stem_thickness_base,d.concrete,t,"retained/heel face")]
        n=sum(x.vertical_kn for x in ver); mo=lm; mr=sum(x.vertical_kn*x.x_m for x in ver)+sum(-min(0,x.horizontal_kn)*x.y_m for x in hor); e=(mr-mo)/n-d.base_width/2; q=n/d.base_width; qmax=max(q*(1-6*e/d.base_width),q*(1+6*e/d.base_width)); heelw=next((x.vertical_kn for x in f if x.name=="soil over heel"),0)/max(d.heel_width,1e-9); hn=qmax-heelw
        out += [self._strip("heel slab",d.strength_load_factor*hn*d.heel_width**2/2,d.strength_load_factor*abs(hn)*d.heel_width,d.base_thickness,d.concrete,t,"top if net upward; verify sign"),self._strip("toe slab",d.strength_load_factor*qmax*d.toe_width**2/2,d.strength_load_factor*qmax*d.toe_width,d.base_thickness,d.concrete,t,"bottom")]
        if d.stiffener:
            s=d.stiffener; tf=lf*s.spacing; out.append(self._strip(f"{s.side} stiffener",d.strength_load_factor*tf*s.top_elevation/2,d.strength_load_factor*tf,s.thickness,d.concrete,t,"wall-parallel face")); t.warn("Counterfort/buttress design is a preliminary tributary-width idealization; model slab/stiffener load path, connections and shear friction for final design.")
        t.warn("RC checks are preliminary: verify SNI load combinations, development, minimum/maximum steel, two-way shear, crack control, detailing and joints.")
        return out


@dataclass(frozen=True)
class GEO5Comparison:
    quantity: str; engine_value: float; geo5_value: float; difference: float; percent_difference: float | None


def compare_to_geo5(result: CalculationResult, geo5_values: dict[str, float]) -> list[GEO5Comparison]:
    """Compare manually supplied same-basis GEO5 values; GEO5 is non-governing."""
    forces={x.name:x.horizontal_kn for x in result.forces}; checks={x.name:x for x in result.checks}; values={"active_effective_soil_pressure":forces.get("active effective-soil pressure",0.),"behind_hydrostatic_pressure":forces.get("behind hydrostatic pressure",0.),"sliding_fs":checks["sliding"].ratio_or_fs,"overturning_fs":checks["overturning"].ratio_or_fs}
    for x in result.trace.items:
        if x.step=="Bearing capacity": values.update(qtoe_kpa=x.values["qtoe_kPa"],qheel_kpa=x.values["qheel_kPa"])
    if set(geo5_values)-set(values): raise ValueError(f"unsupported GEO5 comparison keys: {sorted(set(geo5_values)-set(values))}")
    out=[GEO5Comparison(k,values[k],v,values[k]-v,100*(values[k]-v)/v if v else None) for k,v in geo5_values.items()]
    result.trace.add("GEO5 comparison","difference=engine-GEO5; comparison only",{x.quantity:asdict(x) for x in out},REF["geo5"],"Match geometry, drainage, soil, pressure method, actions, factors and passive assumptions.")
    return out


# ---------- Engineering diagrams (drawn from the active input/result) ----------
def _svg_document(fragments: list[str]) -> str:
    """Join SVG elements and make the result usable inline or as a standalone file."""
    return ''.join(fragments).replace('<svg ', '<svg xmlns="http://www.w3.org/2000/svg" ', 1)


def _svg_wall_section(d: RetainingWallInput, result: CalculationResult | None = None) -> str:
    """Line drawing of the analysis model; one coordinate scale in both axes."""
    width, height, x_toe, y_base = 950, 680, 290.0, 480.0
    scale = min(48.0, 320.0 / max(d.wall_height, .01), 340.0 / max(d.base_width, .01))
    x_front = x_toe + d.toe_width * scale
    x_back = x_front + d.stem_thickness_base * scale
    x_top_back = x_front + d.stem_thickness_top * scale
    x_heel = x_toe + d.base_width * scale
    y_top = y_base - d.wall_height * scale
    y_bottom = y_base + d.base_thickness * scale
    y_front = y_base - d.front_soil_height * scale
    x_soil_end = 900.0
    flat = d.terrain.flat_length * scale if d.terrain.kind is TerrainKind.FLAT_THEN_INFINITE_SLOPE else 0.0
    if d.terrain.beta_rad > 0:
        x_soil_end = min(x_soil_end, x_top_back+flat+max(0.0, y_top-110.0)/tan(d.terrain.beta_rad))
    x_break = min(x_soil_end, x_top_back + flat)
    y_soil_end = y_top - max(0.0, x_soil_end - x_break) * tan(d.terrain.beta_rad)
    ground_path = f'M {x_top_back:.1f} {y_top:.1f} L {x_break:.1f} {y_top:.1f} L {x_soil_end:.1f} {y_soil_end:.1f}'
    parts = [f'''<svg viewBox="0 0 {width} {height}" role="img" aria-label="Scale-consistent retaining wall analysis drawing" style="width:100%;height:auto;display:block;background:white">
    <style>text{{font-family:Arial,sans-serif;fill:#263238}}.title{{font-size:19px;font-weight:700}}.label{{font-size:13px}}.small{{font-size:11px}}.dim{{font-size:12px;fill:#9b5b18}}.earth{{stroke:#d33b3b;fill:none;stroke-width:1.7}}.water{{stroke:#10a6b8;fill:none;stroke-width:1.6;stroke-dasharray:9 6}}.ground{{stroke:#30363b;fill:none;stroke-width:1.7}}.layer{{stroke:#9a9a9a;fill:none;stroke-width:1;stroke-dasharray:5 5}}</style>
    <rect x="1" y="1" width="{width-2}" height="{height-2}" fill="white" stroke="#444" stroke-width="1.5"/>
    <text x="28" y="36" class="title">RETAINING WALL — ANALYSIS SECTION</text>
    <text x="28" y="56" class="small">Geometry uses one scale in both axes; the pressure diagrams below use their own stated scale.</text>
    <path d="{ground_path}" class="ground"/><path d="M 38 {y_front:.1f} L {x_front:.1f} {y_front:.1f}" class="ground"/>
    <rect x="{x_toe:.1f}" y="{y_base:.1f}" width="{d.base_width*scale:.1f}" height="{d.base_thickness*scale:.1f}" fill="white" stroke="#27313b" stroke-width="2.2"/>
    <path d="M {x_front:.1f} {y_base:.1f} L {x_back:.1f} {y_base:.1f} L {x_top_back:.1f} {y_top:.1f} L {x_front:.1f} {y_top:.1f} Z" fill="white" stroke="#27313b" stroke-width="2.2"/>
    <text x="{x_front-12:.1f}" y="{y_top-10:.1f}" class="label">stem</text>
    <text x="48" y="{y_front-9:.1f}" class="label">front ground EL {d.front_soil_elevation:.2f} m</text>
    <text x="{x_top_back+220:.1f}" y="{y_top+24:.1f}" class="label">retained ground EL {d.base_top_elevation+d.wall_height:.2f} m</text>''']
    # Soil boundaries are drawn as linework, retaining the entered layer depths.
    depth = 0.0
    for index, layer in enumerate(d.soils[:-1], 1):
        depth += layer.thickness
        if 0 < depth < d.wall_height:
            yy = y_top + depth * scale
            parts.append(f'<line x1="{x_back:.1f}" y1="{yy:.1f}" x2="{x_soil_end:.1f}" y2="{yy:.1f}" class="layer"/><text x="{x_soil_end-62:.1f}" y="{yy-5:.1f}" class="small">layer {index+1}</text>')
    if d.stiffener:
        s = d.stiffener
        side = x_back if s.side == "heel" else x_front
        direction = 1 if s.side == "heel" else -1
        yt = y_base - s.top_elevation * scale
        yc = y_base - s.constant_depth_to_elevation * scale
        xt = side + direction * s.depth_top * scale
        xb = side + direction * s.depth_base * scale
        parts.append(f'<path d="M {side:.1f} {y_base:.1f} L {xb:.1f} {y_base:.1f} L {xt:.1f} {yc:.1f} L {xt:.1f} {yt:.1f} L {side:.1f} {yt:.1f} Z" fill="none" stroke="#555" stroke-width="1.5" stroke-dasharray="6 4"/><text x="{xt+8:.1f}" y="{yc-8:.1f}" class="small">{("counterfort" if s.side=="heel" else "buttress")} plane</text>')
    if d.shear_key:
        k = d.shear_key
        xk = x_toe + (k.x_from_toe-k.thickness/2) * scale
        parts.append(f'<rect x="{xk:.1f}" y="{y_bottom:.1f}" width="{k.thickness*scale:.1f}" height="{k.depth*scale:.1f}" fill="white" stroke="#27313b" stroke-width="2"/><text x="{xk-4:.1f}" y="{y_bottom+k.depth*scale+18:.1f}" class="small">shear key</text>')
    if d.groundwater.behind_depth is not None:
        yw = y_top + d.groundwater.behind_depth * scale
        parts.append(f'<line x1="{x_top_back:.1f}" y1="{yw:.1f}" x2="{x_soil_end:.1f}" y2="{yw:.1f}" class="water"/><text x="{x_soil_end-160:.1f}" y="{yw-7:.1f}" class="label" fill="#008ea0">GWT behind EL {d.base_top_elevation+d.wall_height-d.groundwater.behind_depth:.2f} m</text>')
    if d.groundwater.front_depth is not None:
        yw = y_front + d.groundwater.front_depth * scale
        parts.append(f'<line x1="38" y1="{yw:.1f}" x2="{x_front:.1f}" y2="{yw:.1f}" class="water"/><text x="48" y="{yw-8:.1f}" class="label" fill="#008ea0">front water EL {d.front_soil_elevation-d.groundwater.front_depth:.2f} m</text>')
    if d.surcharge_kpa:
        ya = max(92.0, min(y_top, y_soil_end)-30.0)
        x1, x2 = x_top_back+50, min(x_soil_end-10, x_top_back+230)
        parts.append(f'<line x1="{x1:.1f}" y1="{ya:.1f}" x2="{x2:.1f}" y2="{ya:.1f}" class="ground"/><text x="{x1:.1f}" y="{ya-9:.1f}" class="label">q = {d.surcharge_kpa:.1f} kPa</text>')
        for xx in range(int(x1+12), int(x2), 28):
            parts.append(f'<path d="M {xx} {ya:.1f} L {xx} {ya+20:.1f} m -4 -5 l 4 5 l 4 -5" class="ground"/>')
    # Reference levels and dimension chains.
    parts.append(f'<line x1="{x_toe-48:.1f}" y1="{y_top:.1f}" x2="{x_toe-48:.1f}" y2="{y_base:.1f}" stroke="#9b5b18"/><line x1="{x_toe-54:.1f}" y1="{y_top:.1f}" x2="{x_toe-42:.1f}" y2="{y_top:.1f}" stroke="#9b5b18"/><line x1="{x_toe-54:.1f}" y1="{y_base:.1f}" x2="{x_toe-42:.1f}" y2="{y_base:.1f}" stroke="#9b5b18"/><text x="{x_toe-120:.1f}" y="{(y_top+y_base)/2:.1f}" class="dim">H = {d.wall_height:.2f} m</text>')
    parts.append(f'<text x="{x_heel+13:.1f}" y="{y_bottom+4:.1f}" class="small">bottom base EL {d.base_top_elevation-d.base_thickness:.2f} m</text>')
    y_dim = min(620.0, y_bottom+76.0)
    for xa, xb, value, name in ((x_toe,x_front,d.toe_width,'toe'),(x_front,x_back,d.stem_thickness_base,'stem'),(x_back,x_heel,d.heel_width,'heel')):
        parts.append(f'<line x1="{xa:.1f}" y1="{y_bottom+8:.1f}" x2="{xa:.1f}" y2="{y_dim+10:.1f}" stroke="#aaa"/><line x1="{xa:.1f}" y1="{y_dim:.1f}" x2="{xb:.1f}" y2="{y_dim:.1f}" stroke="#9b5b18"/><text x="{(xa+xb)/2:.1f}" y="{y_dim-6:.1f}" text-anchor="middle" class="dim">{value:.2f}</text><text x="{(xa+xb)/2:.1f}" y="{y_dim+13:.1f}" text-anchor="middle" class="small">{name}</text>')
    parts.append(f'<line x1="{x_heel:.1f}" y1="{y_bottom+8:.1f}" x2="{x_heel:.1f}" y2="{y_dim+10:.1f}" stroke="#aaa"/><text x="{(x_toe+x_heel)/2:.1f}" y="{y_dim+26:.1f}" text-anchor="middle" class="dim">B = {d.base_width:.2f} m</text>')
    if result:
        values = {f.name:f for f in result.forces}
        earth = sum(max(0.0, values[name].horizontal_kn) for name in ("active effective-soil pressure","surcharge lateral pressure","seismic earth-pressure increment") if name in values)
        water = values.get("behind hydrostatic pressure")
        front = values.get("front hydrostatic resistance")
        for yy, label, color, direction in ((y_top+.62*d.wall_height*scale,f'P earth = {earth:.1f} kN/m','#d33b3b',-1),(y_top+.79*d.wall_height*scale,f'P water = {water.horizontal_kn:.1f} kN/m' if water else '', '#10a6b8',-1),(y_base-27.0,f'P front water = {abs(front.horizontal_kn):.1f} kN/m' if front else '', '#10a6b8',1)):
            if label:
                anchor = x_back+45 if direction==-1 else x_front-45
                tip = x_back+5 if direction==-1 else x_front-5
                parts.append(f'<path d="M {anchor:.1f} {yy:.1f} L {tip:.1f} {yy:.1f} m {-direction*6} -4 l {direction*6} 4 l {-direction*6} 4" fill="none" stroke="{color}" stroke-width="1.5"/><text x="{anchor+8 if direction==-1 else 55:.1f}" y="{yy-7:.1f}" class="small">{label}</text>')
    parts.append(f'<line x1="34" y1="642" x2="{34+scale:.1f}" y2="642" stroke="#222" stroke-width="2"/><line x1="34" y1="636" x2="34" y2="648" stroke="#222"/><line x1="{34+scale:.1f}" y1="636" x2="{34+scale:.1f}" y2="648" stroke="#222"/><text x="34" y="661" class="small">1 m section scale</text></svg>')
    return _svg_document(parts)


def _svg_reinforcement_section(d: RetainingWallInput, reinforcement: list[RCDesign]) -> str:
    """Section-scale concrete outline and indicative primary reinforcement routes."""
    width, height, x_toe, y_base = 1050, 620, 250.0, 452.0
    scale = min(47.0, 320.0 / max(d.wall_height, .01), 430.0 / max(d.base_width, .01))
    x_front = x_toe+d.toe_width*scale
    x_back = x_front+d.stem_thickness_base*scale
    x_top_back = x_front+d.stem_thickness_top*scale
    x_heel = x_toe+d.base_width*scale
    y_top = y_base-d.wall_height*scale
    y_bottom = y_base+d.base_thickness*scale
    cover = d.concrete.cover_mm/1000*scale
    x_stem_bar_top = x_top_back-cover
    x_stem_bar_bottom = x_back-cover
    y_heel_bar = y_base+cover
    y_toe_bar = y_bottom-cover
    schedule = {item.component:item for item in reinforcement}
    items = [f'''<svg viewBox="0 0 {width} {height}" role="img" aria-label="Scale-consistent indicative reinforcement section" style="width:100%;height:auto;display:block;background:white"><style>text{{font-family:Arial,sans-serif;fill:#263238}}.title{{font-size:19px;font-weight:700}}.label{{font-size:13px}}.small{{font-size:11px}}.bar{{stroke:#bc2837;stroke-width:2.6;fill:none}}.lead{{stroke:#bc2837;stroke-width:1.2;fill:none}}</style><rect x="1" y="1" width="{width-2}" height="{height-2}" fill="white" stroke="#444" stroke-width="1.5"/><text x="28" y="38" class="title">REINFORCEMENT — SECTION LAYOUT</text><text x="28" y="58" class="small">Concrete geometry to one scale in both axes. Bar line thickness and bends are diagrammatic.</text><rect x="{x_toe:.1f}" y="{y_base:.1f}" width="{d.base_width*scale:.1f}" height="{d.base_thickness*scale:.1f}" fill="white" stroke="#27313b" stroke-width="2"/><path d="M {x_front:.1f} {y_base:.1f} L {x_back:.1f} {y_base:.1f} L {x_top_back:.1f} {y_top:.1f} L {x_front:.1f} {y_top:.1f} Z" fill="white" stroke="#27313b" stroke-width="2"/>''']
    if d.shear_key:
        k=d.shear_key; xk=x_toe+(k.x_from_toe-k.thickness/2)*scale
        items.append(f'<rect x="{xk:.1f}" y="{y_bottom:.1f}" width="{k.thickness*scale:.1f}" height="{k.depth*scale:.1f}" fill="white" stroke="#27313b" stroke-width="2"/>')
    if d.stiffener:
        s=d.stiffener; side=x_back if s.side=="heel" else x_front; direction=1 if s.side=="heel" else -1
        items.append(f'<path d="M {side:.1f} {y_base:.1f} L {side+direction*s.depth_base*scale:.1f} {y_base:.1f} L {side+direction*s.depth_top*scale:.1f} {y_base-s.constant_depth_to_elevation*scale:.1f} L {side+direction*s.depth_top*scale:.1f} {y_base-s.top_elevation*scale:.1f}" stroke="#777" stroke-width="1.3" stroke-dasharray="5 4" fill="none"/>')
    stem = schedule.get('stem')
    if stem:
        items.append(f'<path d="M {x_stem_bar_top:.1f} {y_top+cover:.1f} L {x_stem_bar_bottom:.1f} {y_base+cover:.1f} L {x_stem_bar_bottom+max(15,4*cover):.1f} {y_base+cover:.1f}" class="bar"/><path d="M {x_stem_bar_top+3:.1f} {y_top+55:.1f} L 730 115" class="lead"/><text x="737" y="112" class="label">Stem retained face: D{stem.bar_diameter_mm:.0f} @ {stem.spacing_mm} mm</text>')
    heel = schedule.get('heel slab')
    if heel:
        items.append(f'<path d="M {x_back-cover:.1f} {y_heel_bar:.1f} L {x_heel-cover:.1f} {y_heel_bar:.1f}" class="bar"/><path d="M {(x_back+x_heel)/2:.1f} {y_heel_bar:.1f} L 730 258" class="lead"/><text x="737" y="255" class="label">Heel top: D{heel.bar_diameter_mm:.0f} @ {heel.spacing_mm} mm</text>')
    toe = schedule.get('toe slab')
    if toe:
        items.append(f'<path d="M {x_toe+cover:.1f} {y_toe_bar:.1f} L {x_front+cover:.1f} {y_toe_bar:.1f}" class="bar"/><path d="M {(x_toe+x_front)/2:.1f} {y_toe_bar:.1f} L 730 357" class="lead"/><text x="737" y="354" class="label">Toe bottom: D{toe.bar_diameter_mm:.0f} @ {toe.spacing_mm} mm</text>')
    items.append(f'<text x="{x_toe:.1f}" y="{y_bottom+38:.1f}" class="small">Toe</text><text x="{x_heel-35:.1f}" y="{y_bottom+38:.1f}" class="small">Heel</text><text x="700" y="430" class="small">Cover input: {d.concrete.cover_mm:.0f} mm</text><text x="700" y="450" class="small">Anchorage, laps, and transverse steel</text><text x="700" y="466" class="small">require separate detailing.</text><line x1="30" y1="571" x2="{30+scale:.1f}" y2="571" stroke="#222" stroke-width="2"/><line x1="30" y1="565" x2="30" y2="577" stroke="#222"/><line x1="{30+scale:.1f}" y1="565" x2="{30+scale:.1f}" y2="577" stroke="#222"/><text x="30" y="593" class="small">1 m section scale</text></svg>')
    return _svg_document(items)


def _pressure_profile(d: RetainingWallInput, count: int = 31) -> list[tuple[float,float,float,float,float]]:
    """Depth, static soil, surcharge, seismic increment, and water pressure in kPa."""
    values=[]; sigma=0.; dz=d.wall_height/(count-1)
    for i in range(count):
        z=i*dz
        if i:
            mid=z-dz/2; soil_mid=layer_at(d.soils,mid)
            gamma=soil_mid.gamma_sat if d.strength_mode=="undrained" else (max(.01,soil_mid.gamma_sat-d.groundwater.gamma_water) if d.groundwater.behind_depth is not None and mid>d.groundwater.behind_depth else soil_mid.gamma_dry)
            sigma += gamma*dz
        soil=layer_at(d.soils,min(z,max(0,d.wall_height-1e-9))); phi,cohesion,delta=design_shear_parameters(soil,d); ka=1. if d.strength_mode=="undrained" else rankine_ka(phi,d.terrain.beta_rad)
        ps=max(0.,ka*sigma-2*cohesion*sqrt(ka)); pq=ka*d.surcharge_kpa; pe=0.
        if d.design_kh and d.strength_mode=="drained": pe=max(0.,(mononobe_okabe_kae(phi,d.terrain.beta_rad,delta,d.design_kh,d.design_kv)-ka)*sigma)
        pw=d.groundwater.gamma_water*max(0.,z-d.groundwater.behind_depth) if d.groundwater.behind_depth is not None else 0.
        values.append((z,ps,pq,pe,pw))
    return values


def _svg_stress_diagrams(result: CalculationResult) -> str:
    d = result.input
    pts = _pressure_profile(d)
    width, height, y_top, plot_h = 1180, 470, 104.0, 250.0
    p_max = max(1.0, *(v[i] for v in pts for i in range(1, 5)))
    p_scale = 145.0 / p_max
    parts = [f'''<svg viewBox="0 0 {width} {height}" role="img" aria-label="Line diagrams for lateral stress and base pressure" style="width:100%;height:auto;display:block;background:white"><style>text{{font-family:Arial,sans-serif;fill:#263238}}.title{{font-size:19px;font-weight:700}}.small{{font-size:11px}}.label{{font-size:13px}}.axis{{stroke:#555;stroke-width:1.2}}.grid{{stroke:#bbb;stroke-width:1;stroke-dasharray:4 4}}</style><rect x="1" y="1" width="{width-2}" height="{height-2}" fill="white" stroke="#444" stroke-width="1.5"/><text x="28" y="36" class="title">PRESSURE DIAGRAMS — CALCULATED LOAD CASE</text><text x="28" y="56" class="small">All lateral pressure ordinates share one horizontal scale; depth uses the wall-height scale. Values in kPa.</text>''']
    panels = ((84.0,1,'Effective soil','#d33b3b'),(300.0,2,'Surcharge','#d33b3b'),(516.0,3,'Seismic increment','#a44782'),(732.0,4,'Hydrostatic water','#10a6b8'))
    for x0, index, label, color in panels:
        coords = ' '.join(f'{x0+v[index]*p_scale:.1f},{y_top+v[0]/d.wall_height*plot_h:.1f}' for v in pts)
        peak = max(v[index] for v in pts)
        parts.append(f'<text x="{x0:.1f}" y="84" class="label">{label}</text><line x1="{x0:.1f}" y1="{y_top:.1f}" x2="{x0:.1f}" y2="{y_top+plot_h:.1f}" class="axis"/><line x1="{x0:.1f}" y1="{y_top+plot_h:.1f}" x2="{x0+150:.1f}" y2="{y_top+plot_h:.1f}" class="axis"/><line x1="{x0:.1f}" y1="{y_top+plot_h/2:.1f}" x2="{x0+150:.1f}" y2="{y_top+plot_h/2:.1f}" class="grid"/><polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"/><text x="{x0:.1f}" y="{y_top+plot_h+22:.1f}" class="small">peak {peak:.1f} kPa</text>')
    trace = next((item.values for item in result.trace.items if item.step == 'Bearing capacity'), {})
    q_toe, q_heel = float(trace.get('qtoe_kPa', 0)), float(trace.get('qheel_kPa', 0))
    x1, x2, y0 = 963.0, 1115.0, 354.0
    q_scale = 150.0/max(1.0,abs(q_toe),abs(q_heel))
    parts.append(f'<text x="{x1:.1f}" y="84" class="label">Base contact</text><text x="{x1:.1f}" y="101" class="small">q [kPa]</text><line x1="{x1:.1f}" y1="{y0:.1f}" x2="{x2:.1f}" y2="{y0:.1f}" class="axis"/><path d="M {x1:.1f} {y0:.1f} L {x1:.1f} {y0-q_toe*q_scale:.1f} L {x2:.1f} {y0-q_heel*q_scale:.1f} L {x2:.1f} {y0:.1f}" stroke="#c44836" stroke-width="2" fill="none"/><text x="{x1:.1f}" y="{y0+50:.1f}" class="small">toe {q_toe:.1f}</text><text x="{x1:.1f}" y="{y0+68:.1f}" class="small">heel {q_heel:.1f}</text>')
    parts.append(f'<text x="28" y="421" class="small">Depth 0 at retained ground; wall base at {d.wall_height:.2f} m. Lateral diagrams show behind-wall pressure components; opposing front water and key resistance are in the action table.</text><text x="28" y="444" class="small">Pressure scale: {p_max:.1f} kPa = 145 drawing units. Base-contact scale is separate.</text></svg>')
    return _svg_document(parts)


# ---------- Streamlit user interface ----------
def _streamlit_rows(value: Any) -> list[dict[str, Any]]:
    """Accept Streamlit's DataFrame return without importing pandas directly."""
    return value.to_dict(orient="records") if hasattr(value, "to_dict") else list(value)


def _streamlit_soil_layers(rows: Any) -> tuple[SoilLayer, ...]:
    result = []
    for row in _streamlit_rows(rows):
        if float(row.get("thickness", 0)) > 0:
            result.append(SoilLayer(float(row["thickness"]), float(row["gamma_dry"]), float(row["gamma_sat"]), float(row["phi_deg"]), float(row.get("cohesion", 0)), float(row.get("interface_delta_deg", 0)), float(row.get("su_kpa", 0))))
    if not result: raise ValueError("Enter at least one soil layer with positive thickness.")
    return tuple(result)


def streamlit_app() -> None:
    """Professional Streamlit front end. Start with: streamlit run retaining_wall_engine.py"""
    try:
        import streamlit as st
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt, then start: streamlit run retaining_wall_engine.py") from exc

    st.set_page_config(page_title="RetainWorks | RC Wall", page_icon="▰", layout="wide", initial_sidebar_state="expanded")
    st.markdown("""<style>
    .stApp {background:#f5f7fb;color:#162033} .block-container {max-width:1440px;padding-top:1.4rem;padding-bottom:2.5rem}
    [data-testid="stSidebar"] {background:#101c32} [data-testid="stSidebar"] * {color:#eaf0fb}
    [data-testid="stMain"] [data-testid="stWidgetLabel"] p, [data-testid="stMain"] label {color:#34445f !important;font-weight:600}
    [data-testid="stMain"] [data-baseweb="input"] > div, [data-testid="stMain"] [data-baseweb="select"] > div {background:#fff !important;border-color:#cbd5e1 !important}
    [data-testid="stMain"] [data-baseweb="input"] input, [data-testid="stMain"] [data-baseweb="select"] * {color:#17233a !important}
    [data-testid="stMain"] [data-testid="stMetricLabel"] *, [data-testid="stMain"] [data-testid="stMetricValue"] * {color:#17233a !important}
    [data-testid="stTabs"] button [data-testid="stMarkdownContainer"] p {color:#34445f !important;opacity:1 !important;font-weight:600}
    [data-testid="stTabs"] button[aria-selected="true"] [data-testid="stMarkdownContainer"] p {color:#c94351 !important}
    .hero {background:linear-gradient(115deg,#11244a,#1d5f83);border-radius:14px;padding:22px 28px;color:white;margin-bottom:18px}
    .hero h1 {margin:0;font-size:1.6rem;letter-spacing:.1px}.hero p {margin:.35rem 0 0;opacity:.82}
    div[data-testid="stMetric"] {background:white;border:1px solid #e2e8f0;border-radius:10px;padding:12px 15px;box-shadow:0 1px 3px #14213d10}
    .section-note {color:#61708a;font-size:.88rem;margin-top:-.3rem;margin-bottom:1rem}
    .stButton>button {border-radius:8px;font-weight:600;min-height:2.6rem}
    [data-testid="stExpander"] {background:white;border:1px solid #e2e8f0;border-radius:8px}
    </style>""", unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("## RETAINWORKS")
        st.caption("Cantilever RC retaining-wall analysis")
        st.divider()
        st.text_input("Project name", "Concept retaining wall", key="project_name")
        st.text_input("Revision / reference", "REV-01", key="project_revision")
        st.selectbox("Design basis", ["SNI-first preliminary", "Project-specific review required"], key="design_basis")
        st.divider()
        run = st.button("Run calculation", type="primary", width="stretch")
        st.caption("All calculations are per metre length of wall.")
        st.caption("Review warnings before relying on any result.")

    st.markdown("""<div class="hero"><h1>Reinforced-Concrete Retaining Wall</h1><p>Transparent preliminary design • auditable calculation trace • SNI-first reference hierarchy</p></div>""", unsafe_allow_html=True)
    tab_g, tab_l, tab_r = st.tabs(["Geometry, soil layers & GWT", "Actions, seismic & design", "Results & audit"])

    with tab_g:
        st.subheader("Wall geometry")
        st.markdown("<p class='section-note'>Use one common project datum for elevations. Wall height is derived from retained-top minus base-top elevation.</p>", unsafe_allow_html=True)
        a,b,c = st.columns(3)
        with a:
            st.number_input("EL bottom of base slab [m datum]", value=99.40, step=.1, key="base_bottom_el")
            st.number_input("EL retained ground / backfill surface [m datum]", value=106.0, step=.1, key="retained_el")
        with b:
            st.number_input("EL exposed front ground / excavation surface [m datum]", value=102.0, step=.1, key="front_el")
            st.number_input("Base thickness [m]", min_value=.10, value=.60, step=.05, key="base_t")
        with c:
            h_display = st.session_state.get("retained_el",106.0)-(st.session_state.get("base_bottom_el",99.4)+st.session_state.get("base_t",.6))
            st.metric("Derived retained height above top base", f"{h_display:.2f} m")
            st.caption(f"EL top of base slab = {st.session_state.get('base_bottom_el',99.4)+st.session_state.get('base_t',.6):.2f} m datum")
        st.divider(); c1,c2,c3 = st.columns(3)
        with c1:
            st.number_input("Toe width [m]", min_value=0.0, value=1.20, step=.05, key="toe")
            st.number_input("Heel width [m]", min_value=0.0, value=3.20, step=.05, key="heel")
        with c2:
            st.number_input("Stem thickness at top [m]", min_value=.10, value=.30, step=.05, key="stem_top")
            st.number_input("Stem thickness at base [m]", min_value=.10, value=.65, step=.05, key="stem_base")
        with c3:
            st.selectbox("Retained-side terrain", [x.value for x in TerrainKind], format_func=lambda x: {"flat":"Flat", "infinite_slope":"Infinite slope", "flat_then_infinite_slope":"Flat length then infinite slope"}[x], key="terrain_kind")
            _is_flat_terrain = st.session_state.get("terrain_kind") == TerrainKind.FLAT.value
            st.number_input("Slope angle [deg]", min_value=0.0, max_value=50.0, value=8.0, step=.5, key="slope", disabled=_is_flat_terrain)
            if _is_flat_terrain:
                st.caption("Flat selected: the analysis uses β = 0°. Any previously entered slope is ignored.")
            st.number_input("Initial flat length L [m]", min_value=0.0, value=4.0, step=.25, key="flat_len", disabled=st.session_state.get("terrain_kind") != "flat_then_infinite_slope")
        st.divider(); st.subheader("Stiffener / shear key")
        k1,k2 = st.columns(2)
        with k1:
            st.selectbox("Stiffener type", ["None", "Counterfort – heel side", "Buttress – toe side"], key="stiff_type")
            if st.session_state.get("stiff_type") != "None":
                zmax=max(.1, h_display)
                q1,q2,q3 = st.columns(3)
                with q1: st.number_input("Spacing [m]", min_value=.25, value=3.0, step=.25, key="stiff_spacing"); st.number_input("Web thickness [m]", min_value=.10, value=.30, step=.05, key="stiff_t")
                with q2: st.number_input("Top elevation above base [m]", min_value=.05, value=float(zmax), max_value=float(zmax), step=.05, key="stiff_top"); st.number_input("Constant-depth down to [m]", min_value=0.0, value=min(3.0,float(zmax)), max_value=float(zmax), step=.05, key="stiff_const")
                with q3: st.number_input("Upper depth [m]", min_value=0.0, value=.30, step=.05, key="stiff_top_d"); st.number_input("Base depth [m]", min_value=0.0, value=1.20, step=.05, key="stiff_base_d")
        with k2:
            st.toggle("Include shear key", value=False, key="key_on")
            if st.session_state.get("key_on"):
                q1,q2,q3=st.columns(3)
                with q1: st.number_input("Key depth [m]", min_value=.05,value=.70,step=.05,key="key_d")
                with q2: st.number_input("Key thickness [m]",min_value=.05,value=.30,step=.05,key="key_t")
                with q3: st.number_input("Key x from toe [m]",min_value=0.0,value=1.0,step=.05,key="key_x")

    default_rear = [{"thickness":2.0,"gamma_dry":18.0,"gamma_sat":20.0,"phi_deg":32.0,"cohesion":0.0,"interface_delta_deg":0.0,"su_kpa":0.0},{"thickness":8.0,"gamma_dry":19.0,"gamma_sat":21.0,"phi_deg":30.0,"cohesion":0.0,"interface_delta_deg":0.0,"su_kpa":0.0}]
    default_front = [{"thickness":3.0,"gamma_dry":18.0,"gamma_sat":20.0,"phi_deg":30.0,"cohesion":0.0,"interface_delta_deg":0.0,"su_kpa":0.0}]
    with tab_g:
        st.subheader("Retained-side soil profile")
        st.markdown("<p class='section-note'>Layer thickness is measured downward from the retained-soil surface. Add, delete, or edit rows as needed.</p>", unsafe_allow_html=True)
        rear = st.data_editor(default_rear, num_rows="dynamic", hide_index=True, key="rear_soil", width="stretch", column_config={"thickness":"Thickness [m]","gamma_dry":"γ dry [kN/m³]","gamma_sat":"γ sat [kN/m³]","phi_deg":"φ′ [deg]","cohesion":"c′ [kPa]","interface_delta_deg":"δ [deg]","su_kpa":"Su [kPa]"})
        st.subheader("Front-side soil profile")
        st.markdown("<p class='section-note'>Leave equivalent to the retained profile only if the front ground is genuinely the same material sequence.</p>", unsafe_allow_html=True)
        front = st.data_editor(default_front, num_rows="dynamic", hide_index=True, key="front_soil", width="stretch", column_config={"thickness":"Thickness [m]","gamma_dry":"γ dry [kN/m³]","gamma_sat":"γ sat [kN/m³]","phi_deg":"φ′ [deg]","cohesion":"c′ [kPa]","interface_delta_deg":"δ [deg]","su_kpa":"Su [kPa]"})
        st.divider(); st.subheader("Groundwater and drainage")
        w1,w2,w3=st.columns(3)
        with w1: st.toggle("Water behind wall", value=True, key="gwt_back_on"); st.number_input("EL water behind [m datum]",value=103.5,step=.1,key="gwt_back_el",disabled=not st.session_state.get("gwt_back_on"))
        with w2: st.toggle("Water / pond in front", value=True, key="gwt_front_on"); st.number_input("EL water in front [m datum]",value=101.2,step=.1,key="gwt_front_el",disabled=not st.session_state.get("gwt_front_on"))
        with w3: st.number_input("Water unit weight [kN/m³]",min_value=9.0,max_value=10.5,value=9.81,step=.01,key="gamma_w")
        st.info("Hydrostatic pressures on both faces are included. Uplift, drainage flow, custom pore-pressure profiles, capillary action, and hydrodynamic water require a project-specific assessment.")
        try:
            _base_top=st.session_state.base_bottom_el+st.session_state.base_t
            _terrain_kind=TerrainKind(st.session_state.terrain_kind); _terrain_slope=0.0 if _terrain_kind is TerrainKind.FLAT else st.session_state.slope
            _preview=RetainingWallInput(wall_height=st.session_state.retained_el-_base_top,base_top_elevation=_base_top,retained_top_elevation=st.session_state.retained_el,front_soil_elevation=st.session_state.front_el,stem_thickness_top=st.session_state.stem_top,stem_thickness_base=st.session_state.stem_base,toe_width=st.session_state.toe,heel_width=st.session_state.heel,base_thickness=st.session_state.base_t,soils=_streamlit_soil_layers(rear),front_soils=_streamlit_soil_layers(front),terrain=Terrain(_terrain_kind,_terrain_slope,st.session_state.flat_len),groundwater=Groundwater(st.session_state.retained_el-st.session_state.gwt_back_el if st.session_state.gwt_back_on else None,st.session_state.front_el-st.session_state.gwt_front_el if st.session_state.gwt_front_on else None,st.session_state.gamma_w))
            st.subheader("Live geometry and layering illustration")
            st.caption("All elevations use the project datum: bottom base slab, top base slab, backfill, front/excavation, and water levels are shown explicitly.")
            st.markdown(_svg_wall_section(_preview),unsafe_allow_html=True)
        except Exception as exc:
            st.warning(f"Geometry preview will appear after the minimum geometry and soil inputs are valid: {exc}")

    with tab_l:
        st.subheader("Actions and stability criteria")
        l1,l2,l3=st.columns(3)
        with l1:
            st.number_input("Uniform surcharge [kPa]",min_value=0.0,value=12.0,step=1.0,key="surcharge")
            st.info("Base friction angle and base adhesion are identified automatically from the governing founding layer, the selected strength basis, and the interface factors below.")
        with l2:
            st.number_input("PGAm [g]",min_value=0.0,max_value=2.0,value=.20,step=.01,key="pga_m")
            st.number_input("PGAm reduction factor ηPGA",min_value=0.0,max_value=1.0,value=.60,step=.05,key="pga_red")
            st.number_input("Earthquake return period [years]",min_value=1.0,value=475.0,step=25.0,key="return_period")
            st.number_input("kv / kh",min_value=-1.0,max_value=1.0,value=0.0,step=.05,key="kvkh")
            st.caption("Derived kh = PGAm × ηPGA")
            st.toggle("Include reduced passive front resistance",value=True,key="passive_on")
            st.number_input("Passive reduction factor",min_value=0.0,max_value=1.0,value=.50,step=.05,key="passive_red",disabled=not st.session_state.get("passive_on"))
        with l3:
            st.info("Allowable bearing is calculated automatically from the governing founding layer and bearing FS.")
            st.number_input("Required sliding FS",min_value=.1,value=1.50,step=.05,key="fs_slide")
            st.number_input("Required overturning FS",min_value=.1,value=2.00,step=.05,key="fs_ot")
            st.number_input("Preliminary RC load factor",min_value=.1,value=1.60,step=.05,key="strength_factor")
        st.divider(); st.subheader("Concrete and global stability")
        d1,d2,d3=st.columns(3)
        with d1: st.number_input("f′c [MPa]",min_value=15.0,value=28.0,step=1.0,key="fc");st.number_input("fy [MPa]",min_value=250.0,value=420.0,step=10.0,key="fy")
        with d2: st.number_input("Concrete unit weight [kN/m³]",min_value=20.0,value=24.0,step=.5,key="conc_g");st.number_input("Cover [mm]",min_value=20.0,value=50.0,step=5.0,key="cover");st.number_input("Main bar diameter [mm]",min_value=10.0,value=16.0,step=2.0,key="bar_d")
        with d3: st.selectbox("Soil strength basis",["Drained effective stress", "Undrained total stress"],key="strength_mode_ui");st.number_input("Shear-strength reduction η",min_value=.05,max_value=1.0,value=1.0,step=.05,key="shear_red");st.number_input("Base interface factor",min_value=0.0,max_value=1.0,value=.67,step=.05,key="base_interface");st.number_input("Undrained adhesion factor α",min_value=0.0,max_value=1.0,value=.50,step=.05,key="alpha_adh")
        g1,g2,g3=st.columns(3)
        with g1: st.number_input("Bearing capacity FS",min_value=1.0,value=3.0,step=.25,key="bearing_fs");st.number_input("Foundation embedment Df [m]",min_value=0.0,value=0.0,step=.1,key="embedment")
        with g2: st.toggle("Include seismic hydrodynamic water",value=False,key="hydro_dynamic");st.toggle("Record external global-stability result",value=False,key="global_on")
        with g3: st.number_input("External global FS",min_value=.01,value=1.50,step=.05,key="global_fs",disabled=not st.session_state.get("global_on"));st.number_input("Required global FS",min_value=.1,value=1.50,step=.05,key="global_req")
        with st.expander("Seismic water and excess pore-water pressure: design basis"):
            st.markdown("**Free water:** when enabled, the engine applies the simplified Westergaard increment `Pwd = 7/12 · kh · γw · Hw²` at `0.4Hw` on each wetted face. **Excess pore-water pressure in soil:** this is not a universal fixed percentage. It must come from a liquefaction/cyclic-response assessment using site investigation and the design earthquake; use its depth-dependent pore-pressure/strength result in a project-specific model. Mononobe-Okabe is restricted here to drained analysis and is not used as a substitute for liquefaction analysis.")
        st.warning("Seismic analysis is pseudo-static. Verify project seismic criteria, hydrodynamic water, deformation/displacement, and code-specific load combinations separately.")

    if run:
        try:
            ss=st.session_state; base_top=ss.base_bottom_el+ss.base_t; h=ss.retained_el-base_top
            stiffener=None
            if ss.stiff_type != "None":
                stiffener=Stiffener("heel" if ss.stiff_type.startswith("Counterfort") else "toe",ss.stiff_spacing,ss.stiff_t,ss.stiff_top,ss.stiff_const,ss.stiff_top_d,ss.stiff_base_d)
            key=ShearKey(ss.key_d,ss.key_t,ss.key_x) if ss.key_on else None
            _terrain_kind=TerrainKind(ss.terrain_kind); _terrain_slope=0.0 if _terrain_kind is TerrainKind.FLAT else ss.slope
            inp=RetainingWallInput(wall_height=h,base_top_elevation=base_top,retained_top_elevation=ss.retained_el,front_soil_elevation=ss.front_el,stem_thickness_top=ss.stem_top,stem_thickness_base=ss.stem_base,toe_width=ss.toe,heel_width=ss.heel,base_thickness=ss.base_t,soils=_streamlit_soil_layers(rear),front_soils=_streamlit_soil_layers(front),terrain=Terrain(_terrain_kind,_terrain_slope,ss.flat_len),groundwater=Groundwater(ss.retained_el-ss.gwt_back_el if ss.gwt_back_on else None,ss.front_el-ss.gwt_front_el if ss.gwt_front_on else None,ss.gamma_w),surcharge_kpa=ss.surcharge,base_friction_angle_deg=None,adhesion_kpa=0.0,allowable_bearing_kpa=None,include_passive_front=ss.passive_on,passive_reduction=ss.passive_red,shear_key=key,stiffener=stiffener,concrete=Concrete(ss.fc,ss.fy,ss.conc_g,ss.cover,ss.bar_d),global_stability=GlobalStabilityInput(ss.global_fs if ss.global_on else None,ss.global_req,"user-supplied external result"),required_sliding_fs=ss.fs_slide,required_overturning_fs=ss.fs_ot,strength_load_factor=ss.strength_factor,strength_mode="drained" if ss.strength_mode_ui.startswith("Drained") else "undrained",shear_strength_reduction=ss.shear_red,base_interface_factor=ss.base_interface,undrained_adhesion_factor=ss.alpha_adh,foundation_embedment_m=ss.embedment,bearing_safety_factor=ss.bearing_fs,pga_m_g=ss.pga_m,pga_reduction_factor=ss.pga_red,return_period_years=ss.return_period,vertical_to_horizontal_ratio=ss.kvkh,include_hydrodynamic_water=ss.hydro_dynamic)
            st.session_state["rw_result"]=RetainingWallEngine().run(inp);st.session_state["rw_error"]=None
        except Exception as exc: st.session_state["rw_error"]=str(exc)

    with tab_r:
        result=st.session_state.get("rw_result"); error=st.session_state.get("rw_error")
        st.subheader("Calculation dashboard")
        if error: st.error(f"Input could not be analysed: {error}")
        elif result is None: st.info("Complete the inputs and select **Run calculation** in the sidebar.")
        else:
            checks={x.name:x for x in result.checks}; slide=checks["sliding"];ot=checks["overturning"]; bearing=checks.get("allowable bearing")
            m1,m2,m3,m4=st.columns(4);m1.metric("Overall status","PASS" if result.passed else "REVIEW");m2.metric("Sliding FS",f"{slide.ratio_or_fs:.2f}",f"Required ≥ {slide.required:.2f}");m3.metric("Overturning FS",f"{ot.ratio_or_fs:.2f}",f"Required ≥ {ot.required:.2f}");m4.metric("Max bearing",f"{bearing.demand:.1f} kPa" if bearing else "Not set",f"Allowable {bearing.capacity:.1f} kPa" if bearing else None)
            st.divider(); st.subheader("Geometry and working-stress diagrams")
            st.caption("Line drawing with one scale in both geometry axes, levels, soil layers, water, actions, and dimensions")
            st.markdown(_svg_wall_section(result.input, result), unsafe_allow_html=True)
            st.caption("Calculated pressure ordinates; equal horizontal pressure scale across components")
            st.markdown(_svg_stress_diagrams(result), unsafe_allow_html=True)
            st.caption("Indicative section layout of the calculated primary reinforcement")
            st.markdown(_svg_reinforcement_section(result.input, result.reinforcement), unsafe_allow_html=True)
            f1,f2,f3=st.columns(3)
            with f1: st.download_button("Download analysis section (SVG)",_svg_wall_section(result.input,result),file_name="wall_analysis_section.svg",mime="image/svg+xml",width="stretch")
            with f2: st.download_button("Download pressure diagrams (SVG)",_svg_stress_diagrams(result),file_name="wall_pressure_diagrams.svg",mime="image/svg+xml",width="stretch")
            with f3: st.download_button("Download reinforcement drawing (SVG)",_svg_reinforcement_section(result.input,result.reinforcement),file_name="wall_reinforcement.svg",mime="image/svg+xml",width="stretch")
            st.divider();st.subheader("External stability checks")
            check_rows=[{"Check":x.name,"Demand":round(x.demand,3),"Capacity":None if x.capacity is None else round(x.capacity,3),"FS / ratio":round(x.ratio_or_fs,3),"Required":x.required,"Status":"PASS" if x.ok is True else "FAIL" if x.ok is False else "EXTERNAL"} for x in result.checks];st.dataframe(check_rows,width="stretch",hide_index=True)
            c1,c2=st.columns(2)
            with c1: st.subheader("Actions and line of action");st.dataframe([{ "Action":x.name,"H [kN/m]":round(x.horizontal_kn,2),"V [kN/m]":round(x.vertical_kn,2),"x from toe [m]":round(x.x_m,2),"y above base [m]":round(x.y_m,2)} for x in result.forces],width="stretch",hide_index=True)
            with c2: st.subheader("Preliminary reinforcement");st.dataframe([{ "Component":x.component,"Mu [kNm/m]":round(x.mu_knm_per_m,1),"As req. [mm²/m]":round(x.as_required_mm2_per_m,0),"Provide":f"D{x.bar_diameter_mm:.0f} @ {x.spacing_mm} mm","Flexure":"PASS" if x.flexure_ok else "FAIL","Shear":"PASS" if x.shear_ok else "FAIL"} for x in result.reinforcement],width="stretch",hide_index=True)
            st.subheader("Warnings requiring engineering review")
            for warning in result.trace.warnings: st.warning(warning)
            st.subheader("Calculation trace and references")
            for item in result.trace.items:
                with st.expander(item.step):
                    st.code(item.formula,language=None);st.json(item.values);st.caption("References: " + " • ".join(item.references) + (" | " + item.note if item.note else ""))
            payload=json.dumps(result.as_dict(),indent=2,default=str);st.download_button("Download calculation trace (JSON)",payload,file_name="retaining_wall_calculation.json",mime="application/json",width="content")


def _is_streamlit_runtime() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except ImportError:
        return False


if __name__ == "__main__":
    if _is_streamlit_runtime():
        streamlit_app()
    else:
        # Command-line demonstration remains available when Streamlit is not used.
        _sample = RetainingWallInput(
            wall_height=6, base_top_elevation=100, retained_top_elevation=106, front_soil_elevation=102,
            stem_thickness_top=.30, stem_thickness_base=.65, toe_width=1.2, heel_width=3.2, base_thickness=.60,
            soils=(SoilLayer(2,18,20,32), SoilLayer(8,19,21,30)), front_soils=(SoilLayer(3,18,20,30),),
            terrain=Terrain(TerrainKind.FLAT_THEN_INFINITE_SLOPE,8,4), groundwater=Groundwater(2.5,.8), surcharge_kpa=12,
            kh=.12, include_passive_front=True, shear_key=ShearKey(.7,.3,1),
            stiffener=Stiffener("heel",3,.30,6,3,.30,1.2), allowable_bearing_kpa=250,
            global_stability=GlobalStabilityInput(factor_of_safety=1.6, method="Bishop external model"),
        )
        print(json.dumps(RetainingWallEngine().run(_sample).as_dict(), indent=2))
