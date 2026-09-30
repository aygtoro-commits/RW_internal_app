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
    "bearing": (SNI_8460, TEXTBOOK), "rc": (SNI_2847, ACI_318),
    "global": (SNI_8460, FHWA), "geo5": (GEO5,),
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
    cohesion: float = 0.0; interface_delta_deg: float = 0.0

    def __post_init__(self) -> None:
        if min(self.thickness, self.gamma_dry, self.gamma_sat) <= 0: raise ValueError("soil thickness/unit weights must be positive")
        if not 0 < self.phi_deg < 55: raise ValueError("phi_deg must be (0,55)")


@dataclass(frozen=True)
class Groundwater:
    behind_depth: float | None = None   # depth below retained soil top
    front_depth: float | None = None    # depth below front soil top
    gamma_water: float = 9.81

    def __post_init__(self) -> None:
        if any(x is not None and x < 0 for x in (self.behind_depth, self.front_depth)): raise ValueError("water depths must be >= 0")


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

    def __post_init__(self) -> None:
        if min(self.wall_height, self.stem_thickness_top, self.stem_thickness_base, self.base_thickness) <= 0 or min(self.toe_width, self.heel_width) < 0: raise ValueError("invalid wall dimensions")
        if not self.soils or sum(x.thickness for x in self.soils) + 1e-9 < self.wall_height: raise ValueError("retained-side soil layers must cover wall height")
        if self.retained_top_elevation is not None and abs(self.retained_top_elevation-self.base_top_elevation-self.wall_height) > 1e-6: raise ValueError("retained_top_elevation-base_top_elevation must equal wall_height")
        if not self.base_top_elevation <= self.front_soil_elevation <= self.base_top_elevation+self.wall_height: raise ValueError("front soil elevation must be between base and retained top")
        if self.front_soils and sum(x.thickness for x in self.front_soils) + 1e-9 < self.front_soil_height: raise ValueError("front-side layers must cover front soil height")
        if self.kh < 0 or not -.5 < self.kv < .5: raise ValueError("invalid pseudo-static coefficients")
        if self.shear_key and self.shear_key.x_from_toe > self.base_width: raise ValueError("shear key must lie within base")
        if self.stiffener:
            if self.stiffener.top_elevation > self.wall_height: raise ValueError("stiffener top exceeds wall")
            if max(self.stiffener.depth_top, self.stiffener.depth_base) > (self.heel_width if self.stiffener.side == "heel" else self.toe_width): raise ValueError("stiffener exceeds its projection")

    @property
    def base_width(self) -> float: return self.toe_width+self.stem_thickness_base+self.heel_width
    @property
    def front_soil_height(self) -> float: return self.front_soil_elevation-self.base_top_elevation
    @property
    def front_soil_profile(self) -> tuple[SoilLayer, ...]: return self.front_soils or self.soils


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
        if d.kh:
            wc = sum(x.vertical_kn for x in vertical if "self weight" in x.name or "stiffener" in x.name)
            forces.append(Force("seismic inertia of concrete wall", d.kh*wc, 0, 0, d.base_thickness+d.wall_height/2))
            trace.add("Wall pseudo-static inertia", "Fi=kh*W_concrete", {"kh":d.kh,"Fi_kN_per_m":d.kh*wc}, REF["seismic"], "Soil-wedge inertia is in Mononobe-Okabe.")
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
            gamma=max(.01,soil.gamma_sat-d.groundwater.gamma_water) if d.groundwater.behind_depth is not None and mid>d.groundwater.behind_depth else soil.gamma_dry
            svm=sv+gamma*dz/2; sv+=gamma*dz; ka=rankine_ka(soil.phi_deg,beta); pressure=max(0.,ka*svm-2*soil.cohesion*sqrt(ka)); y=h-mid
            ps+=pressure*dz; ms+=pressure*dz*y; pq+=ka*d.surcharge_kpa*dz; mq+=ka*d.surcharge_kpa*dz*y
            if d.kh: inc=max(0.,(mononobe_okabe_kae(soil.phi_deg,beta,soil.interface_delta_deg,d.kh,d.kv)-ka)*svm); pe+=inc*dz; me+=inc*dz*y
            z+=dz
        out=[Force("active effective-soil pressure",ps,0,0,ms/ps if ps else 0),Force("surcharge lateral pressure",pq,0,0,mq/pq if pq else 0)]
        t.add("Active earth pressure","sigma_h'=max(0,Ka*sigma_v'-2*c'*sqrt(Ka)); P=integral(sigma_h'dz)",{"Psoil_kN_per_m":ps,"Psurcharge_kN_per_m":pq,"terrain":d.terrain.kind.value},REF["active"])
        if d.kh: out.append(Force("seismic earth-pressure increment",pe,0,0,me/pe if pe else 0)); t.add("Seismic earth pressure","Delta sigma_h'=(Kae-Ka)*sigma_v'; Kae=Mononobe-Okabe",{"kh":d.kh,"kv":d.kv,"DeltaP_kN_per_m":pe},REF["seismic"]); t.warn("Hydrodynamic seismic-water force and displacement checks are outside this preliminary engine.")
        if d.groundwater.behind_depth is not None and d.groundwater.behind_depth<h:
            hw=h-d.groundwater.behind_depth; p=.5*d.groundwater.gamma_water*hw**2; out.append(Force("behind hydrostatic pressure",p,0,0,hw/3)); t.add("Behind hydrostatic pressure","Pw=0.5*gamma_w*hw^2",{"Pw_kN_per_m":p},REF["water"])
        if d.groundwater.front_depth is not None:
            hw=max(0.,d.front_soil_height-d.groundwater.front_depth)
            if hw: p=.5*d.groundwater.gamma_water*hw**2; out.append(Force("front hydrostatic resistance",-p,0,0,hw/3)); t.add("Front hydrostatic pressure","Pw=0.5*gamma_w*hw^2 (resisting)",{"Pw_kN_per_m":p},REF["water"])
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
        phi=d.base_friction_angle_deg if d.base_friction_angle_deg is not None else d.soils[-1].phi_deg; friction=n*tan(radians(phi))+d.adhesion_kpa*d.base_width; passive=0.
        if d.include_passive_front and d.front_soil_height:
            soil=d.front_soil_profile[0]; kp=(1+sin(radians(soil.phi_deg)))/(1-sin(radians(soil.phi_deg))); passive+=d.passive_reduction*.5*kp*soil.gamma_dry*d.front_soil_height**2
            if d.shear_key: soil=d.front_soil_profile[-1];kp=(1+sin(radians(soil.phi_deg)))/(1-sin(radians(soil.phi_deg)));passive+=d.passive_reduction*.5*kp*soil.gamma_dry*d.shear_key.depth**2
        resistance=friction+passive+front; fs=resistance/drive if drive else float("inf"); mo=sum(max(0,x.horizontal_kn)*x.y_m for x in hor); mr=sum(x.vertical_kn*x.x_m for x in ver)+sum(-min(0,x.horizontal_kn)*x.y_m for x in hor); fsot=mr/mo if mo else float("inf"); xr=(mr-mo)/n if n else 0.; e=xr-d.base_width/2; q=n/d.base_width; qtoe=q*(1-6*e/d.base_width); qheel=q*(1+6*e/d.base_width)
        t.add("Sliding stability","FS=(N*tan(delta_base)+c_a*B+Ppassive+Pfront-water)/Pdrive",{"FS_sliding":fs,"driving_kN_per_m":drive,"resistance_kN_per_m":resistance},REF["stability"]); t.add("Overturning and bearing","FSot=Mresist/Moverturn; q=qavg*(1 +/- 6e/B)",{"FS_overturning":fsot,"eccentricity_m":e,"qtoe_kPa":qtoe,"qheel_kPa":qheel},REF["bearing"])
        if not d.include_passive_front: t.warn("Passive front resistance is excluded by default; enable only when permanent cover/development are assured.")
        if d.include_passive_front and d.groundwater.front_depth is not None: t.warn("Front passive resistance uses gamma_dry; assess water/effective-stress effect before relying on it.")
        out=[Check("sliding",drive,resistance,fs,d.required_sliding_fs,fs>=d.required_sliding_fs,"FS"),Check("overturning",mo,mr,fsot,d.required_overturning_fs,fsot>=d.required_overturning_fs,"FS"),Check("resultant within middle third",abs(e),d.base_width/6,abs(e)/(d.base_width/6),1.,qtoe>=0 and qheel>=0,"m")]
        if d.allowable_bearing_kpa is not None: out.append(Check("allowable bearing",max(qtoe,qheel),d.allowable_bearing_kpa,max(qtoe,qheel)/d.allowable_bearing_kpa,1.,max(qtoe,qheel)<=d.allowable_bearing_kpa and min(qtoe,qheel)>=0,"kPa"))
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
        if x.step=="Overturning and bearing": values.update(qtoe_kpa=x.values["qtoe_kPa"],qheel_kpa=x.values["qheel_kPa"])
    if set(geo5_values)-set(values): raise ValueError(f"unsupported GEO5 comparison keys: {sorted(set(geo5_values)-set(values))}")
    out=[GEO5Comparison(k,values[k],v,values[k]-v,100*(values[k]-v)/v if v else None) for k,v in geo5_values.items()]
    result.trace.add("GEO5 comparison","difference=engine-GEO5; comparison only",{x.quantity:asdict(x) for x in out},REF["geo5"],"Match geometry, drainage, soil, pressure method, actions, factors and passive assumptions.")
    return out


# ---------- Streamlit user interface ----------
def _streamlit_rows(value: Any) -> list[dict[str, Any]]:
    """Accept Streamlit's DataFrame return without importing pandas directly."""
    return value.to_dict(orient="records") if hasattr(value, "to_dict") else list(value)


def _streamlit_soil_layers(rows: Any) -> tuple[SoilLayer, ...]:
    result = []
    for row in _streamlit_rows(rows):
        if float(row.get("thickness", 0)) > 0:
            result.append(SoilLayer(float(row["thickness"]), float(row["gamma_dry"]), float(row["gamma_sat"]), float(row["phi_deg"]), float(row.get("cohesion", 0)), float(row.get("interface_delta_deg", 0))))
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
    tab_g, tab_s, tab_l, tab_r = st.tabs(["Geometry", "Ground & water", "Actions", "Results & audit"])

    with tab_g:
        st.subheader("Wall geometry")
        st.markdown("<p class='section-note'>Use one common project datum for elevations. Wall height is derived from retained-top minus base-top elevation.</p>", unsafe_allow_html=True)
        a,b,c = st.columns(3)
        with a:
            st.number_input("Base-top elevation [m]", value=100.0, step=.1, key="base_el")
            st.number_input("Retained-soil top elevation [m]", value=106.0, step=.1, key="retained_el")
        with b:
            st.number_input("Front-soil top elevation [m]", value=102.0, step=.1, key="front_el")
            st.number_input("Base thickness [m]", min_value=.10, value=.60, step=.05, key="base_t")
        with c:
            h_display = st.session_state.get("retained_el",106.0)-st.session_state.get("base_el",100.0)
            st.metric("Derived retained height", f"{h_display:.2f} m")
            st.caption("Must be positive; the engine uses base-top as its internal vertical datum.")
        st.divider(); c1,c2,c3 = st.columns(3)
        with c1:
            st.number_input("Toe width [m]", min_value=0.0, value=1.20, step=.05, key="toe")
            st.number_input("Heel width [m]", min_value=0.0, value=3.20, step=.05, key="heel")
        with c2:
            st.number_input("Stem thickness at top [m]", min_value=.10, value=.30, step=.05, key="stem_top")
            st.number_input("Stem thickness at base [m]", min_value=.10, value=.65, step=.05, key="stem_base")
        with c3:
            st.selectbox("Retained-side terrain", [x.value for x in TerrainKind], format_func=lambda x: {"flat":"Flat", "infinite_slope":"Infinite slope", "flat_then_infinite_slope":"Flat length then infinite slope"}[x], key="terrain_kind")
            st.number_input("Slope angle [deg]", min_value=0.0, max_value=50.0, value=8.0, step=.5, key="slope")
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

    default_rear = [{"thickness":2.0,"gamma_dry":18.0,"gamma_sat":20.0,"phi_deg":32.0,"cohesion":0.0,"interface_delta_deg":0.0},{"thickness":8.0,"gamma_dry":19.0,"gamma_sat":21.0,"phi_deg":30.0,"cohesion":0.0,"interface_delta_deg":0.0}]
    default_front = [{"thickness":3.0,"gamma_dry":18.0,"gamma_sat":20.0,"phi_deg":30.0,"cohesion":0.0,"interface_delta_deg":0.0}]
    with tab_s:
        st.subheader("Retained-side soil profile")
        st.markdown("<p class='section-note'>Layer thickness is measured downward from the retained-soil surface. Add, delete, or edit rows as needed.</p>", unsafe_allow_html=True)
        rear = st.data_editor(default_rear, num_rows="dynamic", hide_index=True, key="rear_soil", width="stretch", column_config={"thickness":"Thickness [m]","gamma_dry":"γ dry [kN/m³]","gamma_sat":"γ sat [kN/m³]","phi_deg":"φ′ [deg]","cohesion":"c′ [kPa]","interface_delta_deg":"δ [deg]"})
        st.subheader("Front-side soil profile")
        st.markdown("<p class='section-note'>Leave equivalent to the retained profile only if the front ground is genuinely the same material sequence.</p>", unsafe_allow_html=True)
        front = st.data_editor(default_front, num_rows="dynamic", hide_index=True, key="front_soil", width="stretch", column_config={"thickness":"Thickness [m]","gamma_dry":"γ dry [kN/m³]","gamma_sat":"γ sat [kN/m³]","phi_deg":"φ′ [deg]","cohesion":"c′ [kPa]","interface_delta_deg":"δ [deg]"})
        st.divider(); st.subheader("Groundwater and drainage")
        w1,w2,w3=st.columns(3)
        with w1: st.toggle("Groundwater behind", value=True, key="gwt_back_on"); st.number_input("Depth below retained top [m]",min_value=0.0,value=2.5,step=.1,key="gwt_back",disabled=not st.session_state.get("gwt_back_on"))
        with w2: st.toggle("Groundwater in front", value=True, key="gwt_front_on"); st.number_input("Depth below front top [m]",min_value=0.0,value=.8,step=.1,key="gwt_front",disabled=not st.session_state.get("gwt_front_on"))
        with w3: st.number_input("Water unit weight [kN/m³]",min_value=9.0,max_value=10.5,value=9.81,step=.01,key="gamma_w")
        st.info("Hydrostatic pressures on both faces are included. Uplift, drainage flow, custom pore-pressure profiles, capillary action, and hydrodynamic water require a project-specific assessment.")

    with tab_l:
        st.subheader("Actions and stability criteria")
        l1,l2,l3=st.columns(3)
        with l1:
            st.number_input("Uniform surcharge [kPa]",min_value=0.0,value=12.0,step=1.0,key="surcharge")
            st.number_input("Base friction angle δbase [deg]",min_value=0.0,max_value=50.0,value=28.0,step=.5,key="base_phi")
            st.number_input("Base adhesion [kPa]",min_value=0.0,value=0.0,step=1.0,key="adhesion")
        with l2:
            st.number_input("Horizontal seismic coefficient kh",min_value=0.0,max_value=.6,value=.12,step=.01,key="kh")
            st.number_input("Vertical seismic coefficient kv",min_value=-.49,max_value=.49,value=0.0,step=.01,key="kv")
            st.toggle("Include reduced passive front resistance",value=True,key="passive_on")
            st.number_input("Passive reduction factor",min_value=0.0,max_value=1.0,value=.50,step=.05,key="passive_red",disabled=not st.session_state.get("passive_on"))
        with l3:
            st.number_input("Allowable bearing pressure [kPa]",min_value=1.0,value=250.0,step=10.0,key="q_allow")
            st.number_input("Required sliding FS",min_value=.1,value=1.50,step=.05,key="fs_slide")
            st.number_input("Required overturning FS",min_value=.1,value=2.00,step=.05,key="fs_ot")
            st.number_input("Preliminary RC load factor",min_value=.1,value=1.60,step=.05,key="strength_factor")
        st.divider(); st.subheader("Concrete and global stability")
        d1,d2,d3=st.columns(3)
        with d1: st.number_input("f′c [MPa]",min_value=15.0,value=28.0,step=1.0,key="fc");st.number_input("fy [MPa]",min_value=250.0,value=420.0,step=10.0,key="fy")
        with d2: st.number_input("Concrete unit weight [kN/m³]",min_value=20.0,value=24.0,step=.5,key="conc_g");st.number_input("Cover [mm]",min_value=20.0,value=50.0,step=5.0,key="cover");st.number_input("Main bar diameter [mm]",min_value=10.0,value=16.0,step=2.0,key="bar_d")
        with d3: st.toggle("Record external global-stability result",value=False,key="global_on");st.number_input("External global FS",min_value=.01,value=1.50,step=.05,key="global_fs",disabled=not st.session_state.get("global_on"));st.number_input("Required global FS",min_value=.1,value=1.50,step=.05,key="global_req")
        st.warning("Seismic analysis is pseudo-static. Verify project seismic criteria, hydrodynamic water, deformation/displacement, and code-specific load combinations separately.")

    if run:
        try:
            ss=st.session_state; h=ss.retained_el-ss.base_el
            stiffener=None
            if ss.stiff_type != "None":
                stiffener=Stiffener("heel" if ss.stiff_type.startswith("Counterfort") else "toe",ss.stiff_spacing,ss.stiff_t,ss.stiff_top,ss.stiff_const,ss.stiff_top_d,ss.stiff_base_d)
            key=ShearKey(ss.key_d,ss.key_t,ss.key_x) if ss.key_on else None
            inp=RetainingWallInput(wall_height=h,base_top_elevation=ss.base_el,retained_top_elevation=ss.retained_el,front_soil_elevation=ss.front_el,stem_thickness_top=ss.stem_top,stem_thickness_base=ss.stem_base,toe_width=ss.toe,heel_width=ss.heel,base_thickness=ss.base_t,soils=_streamlit_soil_layers(rear),front_soils=_streamlit_soil_layers(front),terrain=Terrain(TerrainKind(ss.terrain_kind),ss.slope,ss.flat_len),groundwater=Groundwater(ss.gwt_back if ss.gwt_back_on else None,ss.gwt_front if ss.gwt_front_on else None,ss.gamma_w),surcharge_kpa=ss.surcharge,base_friction_angle_deg=ss.base_phi,adhesion_kpa=ss.adhesion,allowable_bearing_kpa=ss.q_allow,kh=ss.kh,kv=ss.kv,include_passive_front=ss.passive_on,passive_reduction=ss.passive_red,shear_key=key,stiffener=stiffener,concrete=Concrete(ss.fc,ss.fy,ss.conc_g,ss.cover,ss.bar_d),global_stability=GlobalStabilityInput(ss.global_fs if ss.global_on else None,ss.global_req,"user-supplied external result"),required_sliding_fs=ss.fs_slide,required_overturning_fs=ss.fs_ot,strength_load_factor=ss.strength_factor)
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
