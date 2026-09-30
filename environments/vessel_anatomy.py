"""Anatomical vessel territories where thrombus actually lodges.

The legacy scenarios are three tubes and a fork. That is enough to test whether
a policy can steer, but not whether it can steer *anywhere real*: the geometry
carries no information about the calibre ratios, branch angles or tortuosity a
device meets in a named vessel, and the clot can sit anywhere rather than at the
sites where clots actually form.

This module holds one `Territory` per vascular region, each carrying the
morphometry of its named segments (diameters, lengths, branch angles, curvature)
and the sites within it where thrombus preferentially lodges. Geometry is built
from that description by `build_territory`, so the anatomy lives as data that can
be checked against a reference, not as coordinates buried in code.

Scale
-----
Every territory is specified in millimetres and then normalised into the unit
cube by `segments_to_vessel_tree`, which scales isotropically. Isotropic scaling
preserves the ratios that matter -- diameter ratios between segments, the ratio
of vessel diameter to device diameter, branch angles -- while letting the
simulator keep one length unit across territories that differ 20-fold in true
size (a calf vein and a main pulmonary artery).

The consequence is that `sim_radius / device_radius` is only faithful for one
territory at a time, since the device is a fixed size in sim units. Each
territory therefore records `reference_diameter_mm`, the calibre its scaling is
pinned to, so a reader can convert any sim length back to millimetres.

Thrombogenic geometry
---------------------
Where a clot lodges is not arbitrary; it follows the flow. The recurring motifs,
each represented by a `ClotSite`:

  * bifurcation apex -- flow divides, wall shear drops on the outer walls just
    distal to the apex, and an embolus travelling in the parent lodges where the
    lumen abruptly narrows. This is the dominant arterial embolic site.
  * stenosis -- a plaque both narrows the lumen and creates a recirculation zone
    immediately downstream, where platelets accumulate.
  * valve pocket -- in veins, the sinus behind a valve leaflet is a near-stagnant
    pocket. Most calf-vein thrombi start here.
  * extrinsic compression -- a vessel pinched by a crossing structure
    (May-Thurner), giving both stenosis and downstream stasis.
  * curvature outer wall -- on a tight bend the inner wall sees high shear and
    the outer wall low shear, so thrombus favours the outer wall.

Sources are cited per territory in `notes`, against the morphometry literature.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Segment:
    """One named vessel segment, in millimetres.

    `diameter_mm` is the lumen diameter proximally, `diameter_distal_mm` at its
    distal end (defaults to no taper). `length_mm` is measured along the
    centerline, so a tortuous segment is longer than the straight-line distance
    between its ends.

    `parent` names the segment this one arises from, and `angle_deg` is the
    take-off angle from the parent's distal tangent. `rotation_deg` rotates the
    take-off around the parent's axis, which is what separates two daughters
    that leave at the same angle in different planes.
    """

    name: str
    diameter_mm: float
    length_mm: float
    parent: str | None = None
    angle_deg: float = 0.0
    rotation_deg: float = 0.0
    # Fraction along the parent where this segment takes off. 1.0 is the
    # parent's distal end (a true bifurcation); a smaller value is a side
    # branch, which is how perforators and diagonals actually arise.
    origin: float = 1.0
    diameter_distal_mm: float | None = None
    # Peak lateral excursion of the centerline as a fraction of segment length.
    # 0 is straight; the ICA siphon is the extreme case in this collection.
    tortuosity: float = 0.0
    # Number of curvature reversals along the segment. 1 is a simple bow, 2 an
    # S-bend; the carotid siphon needs 2 to read as a siphon at all.
    bends: int = 1
    # Direction the segment heads in when it has no parent, or an override.
    direction: tuple[float, float, float] | None = None


@dataclass
class ClotSite:
    """Where thrombus lodges within a territory, and why.

    `segment` names the vessel, `position` is the fraction along it (0 proximal,
    1 distal), and `mechanism` records the reason so the geometry and the
    placement cannot silently disagree.

    `weight` is the relative likelihood used when sampling which site a given
    episode's clot occupies, reflecting how often the site is implicated
    clinically.
    """

    segment: str
    position: float
    mechanism: str
    weight: float = 1.0
    # Fraction of the local lumen the clot occludes when fully formed.
    occlusion: float = 0.8


@dataclass
class Stenosis:
    """A fixed narrowing, as found rather than as a random perturbation.

    `residual_fraction` is the residual lumen *diameter* as a fraction of the
    reference, so 0.35 is a 65% diameter stenosis. `length_mm` is the axial
    extent of the lesion.
    """

    segment: str
    position: float
    residual_fraction: float
    length_mm: float
    kind: str = "atherosclerotic"


@dataclass
class Territory:
    """One vascular region: its segments, its lesions, and its clot sites."""

    name: str
    description: str
    circulation: str                    # "arterial" or "venous"
    segments: list[Segment]
    clot_sites: list[ClotSite] = field(default_factory=list)
    stenoses: list[Stenosis] = field(default_factory=list)
    # The calibre the unit-cube scaling is pinned to; see the module docstring.
    reference_diameter_mm: float = 0.0
    # Flow direction. Venous territories drain toward the root, so the device
    # travels against the current where an arterial one travels with it.
    retrograde: bool = False
    notes: str = ""


# Territories are registered here. Populated by the anatomy tables below.
TERRITORIES: dict[str, Territory] = {}


def _register(terr: Territory) -> Territory:
    """Add a territory to the registry, checking it is internally consistent."""
    names = {s.name for s in terr.segments}
    if len(names) != len(terr.segments):
        raise ValueError(f"{terr.name}: duplicate segment names")
    roots = [s for s in terr.segments if s.parent is None]
    if len(roots) != 1:
        raise ValueError(f"{terr.name}: need exactly one root, got {len(roots)}")
    for s in terr.segments:
        if s.parent is not None and s.parent not in names:
            raise ValueError(f"{terr.name}: {s.name} names unknown parent {s.parent}")
    for site in terr.clot_sites:
        if site.segment not in names:
            raise ValueError(f"{terr.name}: clot site on unknown segment {site.segment}")
    for lesion in terr.stenoses:
        if lesion.segment not in names:
            raise ValueError(f"{terr.name}: stenosis on unknown segment {lesion.segment}")
    if not terr.clot_sites:
        raise ValueError(f"{terr.name}: no clot sites declared")
    TERRITORIES[terr.name] = terr
    return terr


# ============================================================ pulmonary embolism

# The saddle embolus is the reason this territory is modelled at all: a thrombus
# straddling the bifurcation of the pulmonary trunk, which is both the most
# recognisable PE and the one where mechanical intervention is used.
#
# Morphometry: main pulmonary artery diameter 24-30mm (CT reference upper limit
# 29mm), trunk length ~47mm before it divides. The right PA is markedly longer
# than the left and leaves at a shallower angle, which is why emboli travel
# preferentially into the right lung. Pulmonary branching does NOT follow
# Murray's cube law -- measured exponent is ~2.3 -- so the daughters are wider
# relative to the parent than a cube-law tree would make them.
_register(Territory(
    name="pulmonary_saddle",
    description="Pulmonary trunk bifurcation with a saddle embolus at the apex",
    circulation="arterial",   # arterial in topology; carries venous blood
    reference_diameter_mm=27.0,
    segments=[
        Segment("main_pa", diameter_mm=27.0, length_mm=47.0,
                diameter_distal_mm=26.0, tortuosity=0.04, bends=1,
                direction=(0.30, 0.0, 1.0)),
        # Right PA: longer, shallower take-off. Both features push emboli right.
        Segment("right_pa", diameter_mm=20.0, length_mm=60.0, parent="main_pa",
                angle_deg=68.0, rotation_deg=0.0, diameter_distal_mm=16.0,
                tortuosity=0.05, bends=1),
        Segment("left_pa", diameter_mm=19.0, length_mm=32.0, parent="main_pa",
                angle_deg=82.0, rotation_deg=180.0, diameter_distal_mm=15.0,
                tortuosity=0.07, bends=1),
        Segment("right_lower_lobar", diameter_mm=12.0, length_mm=35.0,
                parent="right_pa", angle_deg=35.0, rotation_deg=20.0,
                diameter_distal_mm=9.5, tortuosity=0.05),
        Segment("right_upper_lobar", diameter_mm=10.0, length_mm=22.0,
                parent="right_pa", angle_deg=75.0, rotation_deg=200.0,
                origin=0.35, diameter_distal_mm=8.0, tortuosity=0.04),
        Segment("left_lower_lobar", diameter_mm=11.5, length_mm=32.0,
                parent="left_pa", angle_deg=40.0, rotation_deg=15.0,
                diameter_distal_mm=9.0, tortuosity=0.06),
    ],
    clot_sites=[
        # The saddle itself: astride the apex, occluding both outflows.
        ClotSite("main_pa", 0.97, "bifurcation apex (saddle)", weight=3.0,
                 occlusion=0.85),
        # Lobar emboli are far more common than saddles overall, and the right
        # side is favoured by both the shallower take-off and the longer vessel.
        ClotSite("right_lower_lobar", 0.25, "lobar bifurcation, dependent lobe",
                 weight=2.5, occlusion=0.8),
        ClotSite("left_lower_lobar", 0.25, "lobar bifurcation, dependent lobe",
                 weight=1.6, occlusion=0.8),
        ClotSite("right_pa", 0.55, "main branch, flow-divider wall", weight=1.2),
    ],
    notes=(
        "Main PA 24-30mm (CT upper limit of normal 29mm); trunk ~47mm. Right PA "
        "longer and shallower than left, which biases embolus laterality. "
        "Pulmonary branching exponent ~2.3, not Murray's 3.0."
    ),
))


# =================================================================== coronary

# Left main bifurcation. The distal LM angle is bimodal on whether a ramus
# intermedius is present: 89+/-21 deg with, 75+/-21 deg without (n=300 CTA).
# LM diameter 3.5+/-0.8mm, length 10.5+/-5.3mm -- the spread is half the mean,
# which is why `variation` matters more here than anywhere else.
#
# Coronary bifurcations are fitted better by Huo-Kassab than by Murray; the
# Finet relation D_parent = 0.678*(D_main + D_side) fits worst of the four
# tested (RMS 0.456 against Huo-Kassab's 0.165 over 446 real bifurcations), so
# these diameters are taken from measurement rather than derived from a law.
_register(Territory(
    name="coronary_lm_bifurcation",
    description="Left main into LAD and circumflex, the classic PCI bifurcation",
    circulation="arterial",
    reference_diameter_mm=3.5,
    segments=[
        Segment("left_main", diameter_mm=3.5, length_mm=10.5,
                diameter_distal_mm=3.3, tortuosity=0.03,
                direction=(1.0, 0.15, -0.2)),
        Segment("lad_proximal", diameter_mm=3.0, length_mm=30.0,
                parent="left_main", angle_deg=38.0, rotation_deg=0.0,
                diameter_distal_mm=2.6, tortuosity=0.07, bends=1),
        Segment("lad_mid", diameter_mm=2.6, length_mm=40.0,
                parent="lad_proximal", angle_deg=14.0, rotation_deg=30.0,
                diameter_distal_mm=2.2, tortuosity=0.10, bends=2),
        Segment("circumflex", diameter_mm=2.8, length_mm=35.0,
                parent="left_main", angle_deg=44.0, rotation_deg=185.0,
                diameter_distal_mm=2.4, tortuosity=0.12, bends=1),
        # The first diagonal leaves the proximal LAD partway along, not at a
        # terminal split, hence origin < 1.
        Segment("first_diagonal", diameter_mm=2.0, length_mm=25.0,
                parent="lad_proximal", angle_deg=48.0, rotation_deg=250.0,
                origin=0.55, diameter_distal_mm=1.5, tortuosity=0.08),
        Segment("om1", diameter_mm=2.2, length_mm=28.0, parent="circumflex",
                angle_deg=52.0, rotation_deg=95.0, origin=0.45,
                diameter_distal_mm=1.6, tortuosity=0.09),
    ],
    stenoses=[
        # Proximal LAD is the single most common culprit lesion site.
        Stenosis("lad_proximal", 0.22, residual_fraction=0.40, length_mm=12.0),
    ],
    clot_sites=[
        ClotSite("lad_proximal", 0.30, "plaque rupture just distal to stenosis",
                 weight=3.0, occlusion=0.9),
        ClotSite("left_main", 0.95, "LM bifurcation apex", weight=1.0,
                 occlusion=0.85),
        ClotSite("lad_mid", 0.35, "mid-LAD after diagonal take-off", weight=2.0),
        ClotSite("circumflex", 0.30, "proximal LCx, flow divider", weight=1.5),
        ClotSite("om1", 0.20, "OM ostium", weight=0.8),
    ],
    notes=(
        "LM 3.5+/-0.8mm dia, 10.5+/-5.3mm long; distal LM angle 89+/-21 deg with "
        "ramus intermedius, 75+/-21 without (Medrano-Gracia 2016, n=300 CTA). "
        "Huo-Kassab fits coronary bifurcations better than Murray or Finet "
        "(Medrano-Gracia 2017, n=446)."
    ),
))

_register(Territory(
    name="coronary_rca",
    description="Right coronary artery with a proximal-to-mid culprit lesion",
    circulation="arterial",
    reference_diameter_mm=3.6,
    segments=[
        # The RCA runs in the atrioventricular groove, so its centerline is a
        # long C-curve rather than a straight run.
        Segment("rca_proximal", diameter_mm=3.6, length_mm=35.0,
                diameter_distal_mm=3.2, tortuosity=0.13, bends=1,
                direction=(0.8, -0.4, -0.4)),
        Segment("rca_mid", diameter_mm=3.2, length_mm=40.0,
                parent="rca_proximal", angle_deg=22.0, rotation_deg=20.0,
                diameter_distal_mm=2.8, tortuosity=0.14, bends=1),
        Segment("rca_distal", diameter_mm=2.8, length_mm=30.0, parent="rca_mid",
                angle_deg=26.0, rotation_deg=15.0, diameter_distal_mm=2.4,
                tortuosity=0.10, bends=1),
        Segment("pda", diameter_mm=2.0, length_mm=35.0, parent="rca_distal",
                angle_deg=48.0, rotation_deg=10.0, diameter_distal_mm=1.4,
                tortuosity=0.08),
        Segment("plb", diameter_mm=1.9, length_mm=28.0, parent="rca_distal",
                angle_deg=55.0, rotation_deg=170.0, diameter_distal_mm=1.4,
                tortuosity=0.09),
        Segment("acute_marginal", diameter_mm=1.6, length_mm=22.0,
                parent="rca_mid", angle_deg=62.0, rotation_deg=250.0,
                origin=0.5, diameter_distal_mm=1.2, tortuosity=0.07),
    ],
    stenoses=[
        Stenosis("rca_mid", 0.35, residual_fraction=0.38, length_mm=15.0),
    ],
    clot_sites=[
        ClotSite("rca_mid", 0.45, "plaque rupture, mid-RCA", weight=3.0,
                 occlusion=0.9),
        ClotSite("rca_proximal", 0.25, "proximal RCA lesion", weight=2.0,
                 occlusion=0.88),
        ClotSite("rca_distal", 0.90, "crux, before the PDA/PLB split", weight=1.2),
        ClotSite("pda", 0.20, "PDA ostium", weight=0.7),
    ],
    notes="RCA 3.6mm proximally, tapering distally; C-shaped course in the AV groove.",
))


# ================================================================ venous / DVT

# Iliocaval confluence. The left common iliac vein passes under the right common
# iliac artery, and where it is compressed there (May-Thurner) the combination of
# a fixed narrowing and downstream stasis makes it the dominant site of
# iliofemoral DVT -- which is why left-sided DVT outnumbers right. Confluence
# angles are reported around 34 and 49 degrees for the two limbs; that asymmetry
# is itself part of the left-sided predominance.
_register(Territory(
    name="iliac_may_thurner",
    description="Iliocaval confluence with left common iliac vein compression",
    circulation="venous",
    reference_diameter_mm=22.0,
    retrograde=True,
    segments=[
        Segment("ivc", diameter_mm=22.0, length_mm=70.0,
                diameter_distal_mm=20.0, tortuosity=0.03,
                direction=(0.0, 0.1, -1.0)),
        Segment("left_common_iliac", diameter_mm=13.0, length_mm=55.0,
                parent="ivc", angle_deg=49.0, rotation_deg=0.0,
                diameter_distal_mm=12.0, tortuosity=0.07, bends=1),
        Segment("right_common_iliac", diameter_mm=12.5, length_mm=45.0,
                parent="ivc", angle_deg=34.0, rotation_deg=180.0,
                diameter_distal_mm=11.5, tortuosity=0.05, bends=1),
        Segment("left_external_iliac", diameter_mm=11.5, length_mm=60.0,
                parent="left_common_iliac", angle_deg=18.0, rotation_deg=10.0,
                diameter_distal_mm=10.5, tortuosity=0.06),
        Segment("left_common_femoral", diameter_mm=10.5, length_mm=45.0,
                parent="left_external_iliac", angle_deg=14.0, rotation_deg=5.0,
                diameter_distal_mm=9.5, tortuosity=0.05),
        Segment("left_internal_iliac", diameter_mm=8.0, length_mm=35.0,
                parent="left_common_iliac", angle_deg=58.0, rotation_deg=120.0,
                origin=0.8, diameter_distal_mm=6.5, tortuosity=0.08),
    ],
    stenoses=[
        # Extrinsic compression, not plaque: the artery crossing above pins the
        # vein against the vertebral body.
        Stenosis("left_common_iliac", 0.15, residual_fraction=0.45,
                 length_mm=18.0, kind="extrinsic compression"),
    ],
    clot_sites=[
        ClotSite("left_common_iliac", 0.30, "stasis distal to the compression",
                 weight=3.5, occlusion=0.85),
        ClotSite("left_external_iliac", 0.35, "propagation from the iliac vein",
                 weight=2.0, occlusion=0.8),
        ClotSite("left_common_femoral", 0.40, "femoral propagation", weight=1.5),
        ClotSite("right_common_iliac", 0.30, "right-sided, less common",
                 weight=0.6),
    ],
    notes=(
        "Iliocaval confluence angles ~34 and ~49 deg; the left CIV crosses under "
        "the right CIA, giving May-Thurner compression and the left-sided DVT "
        "predominance."
    ),
))

# Popliteal and calf veins. Thrombi start in the valve sinuses, where the pocket
# behind a leaflet is near-stagnant. Note the reverse taper: tributaries are
# smaller than the vessel they drain into, so a device travelling from the
# popliteal outward meets a narrowing lumen, unlike in an arterial tree.
_register(Territory(
    name="popliteal_calf_dvt",
    description="Popliteal vein and calf tributaries; thrombus in valve sinuses",
    circulation="venous",
    reference_diameter_mm=9.0,
    retrograde=True,
    segments=[
        Segment("popliteal", diameter_mm=9.0, length_mm=60.0,
                diameter_distal_mm=8.0, tortuosity=0.05,
                direction=(0.1, 0.0, -1.0)),
        Segment("posterior_tibial", diameter_mm=5.0, length_mm=70.0,
                parent="popliteal", angle_deg=26.0, rotation_deg=0.0,
                diameter_distal_mm=3.8, tortuosity=0.08, bends=2),
        Segment("peroneal", diameter_mm=4.6, length_mm=65.0, parent="popliteal",
                angle_deg=32.0, rotation_deg=140.0, diameter_distal_mm=3.5,
                tortuosity=0.09, bends=2),
        Segment("anterior_tibial", diameter_mm=3.4, length_mm=60.0,
                parent="popliteal", angle_deg=38.0, rotation_deg=250.0,
                diameter_distal_mm=2.6, tortuosity=0.07, bends=1),
        Segment("soleal_sinus", diameter_mm=5.5, length_mm=35.0,
                parent="popliteal", angle_deg=64.0, rotation_deg=60.0,
                origin=0.45, diameter_distal_mm=4.0, tortuosity=0.12, bends=2),
        Segment("gastrocnemius_vein", diameter_mm=4.2, length_mm=30.0,
                parent="popliteal", angle_deg=58.0, rotation_deg=300.0,
                origin=0.25, diameter_distal_mm=3.2, tortuosity=0.10),
    ],
    clot_sites=[
        # The soleal sinuses are the classic site of origin for calf DVT.
        ClotSite("soleal_sinus", 0.45, "valve sinus stasis (soleal)", weight=3.5,
                 occlusion=0.85),
        ClotSite("posterior_tibial", 0.35, "valve pocket, posterior tibial",
                 weight=2.5, occlusion=0.8),
        ClotSite("peroneal", 0.35, "valve pocket, peroneal", weight=2.0),
        ClotSite("popliteal", 0.55, "propagation into the popliteal", weight=2.0,
                 occlusion=0.85),
        ClotSite("gastrocnemius_vein", 0.40, "valve sinus (gastrocnemius)",
                 weight=1.2),
    ],
    notes=(
        "Calf DVT originates in valve sinuses, most often soleal. Reverse taper: "
        "tributaries are smaller than the popliteal they drain into."
    ),
))


# ============================================================== cerebrovascular
#
# PROVENANCE. The territories above were built from morphometry retrieved for
# this work, and their `notes` name the studies. The cerebrovascular territories
# below were transcribed from standard neuroanatomy reference values instead:
# the literature search for this region did not complete. The numbers are
# conventional and self-consistent, but they are NOT freshly sourced, so treat
# them as textbook nominal rather than as measured means, and check them against
# a reference before quoting any of them.

# MCA M1 is the single most common large-vessel-occlusion target. M1 is short and
# horizontal, and the lenticulostriate perforators leave it almost at right
# angles -- which is why a clot extending proximally along M1 infarcts the basal
# ganglia even when the cortical branches reopen.
_register(Territory(
    name="mca_m1_lvo",
    description="MCA M1 occlusion with the M2 bifurcation and perforators",
    circulation="arterial",
    reference_diameter_mm=3.0,
    segments=[
        Segment("m1", diameter_mm=3.0, length_mm=18.0, diameter_distal_mm=2.7,
                tortuosity=0.05, bends=1, direction=(1.0, 0.10, 0.05)),
        Segment("m2_superior", diameter_mm=2.2, length_mm=30.0, parent="m1",
                angle_deg=52.0, rotation_deg=0.0, diameter_distal_mm=1.7,
                tortuosity=0.11, bends=2),
        Segment("m2_inferior", diameter_mm=2.1, length_mm=30.0, parent="m1",
                angle_deg=58.0, rotation_deg=180.0, diameter_distal_mm=1.6,
                tortuosity=0.12, bends=2),
        # Perforators leave M1 nearly perpendicular and are an order of magnitude
        # smaller. They are not a device target -- no catheter enters them -- but
        # they carry flow away from M1, and that calibre step is part of why a
        # clot arrests where it does.
        Segment("lenticulostriate_1", diameter_mm=0.9, length_mm=14.0,
                parent="m1", angle_deg=84.0, rotation_deg=95.0, origin=0.45,
                diameter_distal_mm=0.7, tortuosity=0.10),
        Segment("lenticulostriate_2", diameter_mm=0.8, length_mm=13.0,
                parent="m1", angle_deg=88.0, rotation_deg=115.0, origin=0.70,
                diameter_distal_mm=0.6, tortuosity=0.10),
        Segment("m3_branch", diameter_mm=1.6, length_mm=25.0,
                parent="m2_superior", angle_deg=44.0, rotation_deg=40.0,
                diameter_distal_mm=1.2, tortuosity=0.13, bends=2),
    ],
    clot_sites=[
        ClotSite("m1", 0.55, "M1 trunk, proximal to the bifurcation",
                 weight=3.5, occlusion=0.9),
        ClotSite("m1", 0.95, "M1 bifurcation apex", weight=2.5, occlusion=0.9),
        ClotSite("m2_superior", 0.20, "M2 superior division ostium", weight=1.8,
                 occlusion=0.85),
        ClotSite("m2_inferior", 0.20, "M2 inferior division ostium", weight=1.5,
                 occlusion=0.85),
    ],
    notes=(
        "TEXTBOOK NOMINAL, not freshly sourced. M1 ~3mm and ~18mm long; M2 "
        "divisions ~2mm; lenticulostriate perforators <1mm leaving M1 near "
        "perpendicular."
    ),
))

# The carotid siphon is the passage that makes neurovascular access hard: the
# cavernous ICA reverses curvature twice within a couple of centimetres. It is
# modelled here as a navigation problem rather than a clot target -- the clot
# sites are distal, which is the clinical situation: you must get *through* the
# siphon to reach the lesion.
_register(Territory(
    name="ica_siphon",
    description="Carotid siphon: the S-bend a device must track to reach the MCA",
    circulation="arterial",
    reference_diameter_mm=4.5,
    segments=[
        Segment("ica_cervical", diameter_mm=4.5, length_mm=60.0,
                diameter_distal_mm=4.2, tortuosity=0.06, bends=1,
                direction=(0.1, 0.0, 1.0)),
        Segment("ica_petrous", diameter_mm=4.0, length_mm=25.0,
                parent="ica_cervical", angle_deg=48.0, rotation_deg=0.0,
                diameter_distal_mm=3.8, tortuosity=0.14, bends=1),
        # The siphon proper: two curvature reversals in ~22mm.
        Segment("ica_cavernous", diameter_mm=3.8, length_mm=22.0,
                parent="ica_petrous", angle_deg=42.0, rotation_deg=170.0,
                diameter_distal_mm=3.6, tortuosity=0.30, bends=2),
        Segment("ica_clinoid", diameter_mm=3.6, length_mm=8.0,
                parent="ica_cavernous", angle_deg=38.0, rotation_deg=20.0,
                diameter_distal_mm=3.4, tortuosity=0.10, bends=1),
        Segment("ica_communicating", diameter_mm=3.4, length_mm=12.0,
                parent="ica_clinoid", angle_deg=22.0, rotation_deg=10.0,
                diameter_distal_mm=3.2, tortuosity=0.08),
        Segment("ophthalmic", diameter_mm=1.3, length_mm=18.0,
                parent="ica_clinoid", angle_deg=72.0, rotation_deg=250.0,
                origin=0.4, diameter_distal_mm=1.0, tortuosity=0.09),
    ],
    clot_sites=[
        ClotSite("ica_communicating", 0.70, "distal ICA, beyond the siphon",
                 weight=3.0, occlusion=0.9),
        ClotSite("ica_cavernous", 0.55, "thrombus within the siphon bend",
                 weight=1.5, occlusion=0.85),
        ClotSite("ica_petrous", 0.50, "petrous segment", weight=1.0),
    ],
    notes=(
        "TEXTBOOK NOMINAL, not freshly sourced. Cavernous ICA reverses curvature "
        "twice over ~22mm; this territory exists to make the siphon a navigation "
        "obstacle, with the clot placed beyond it."
    ),
))

# ICA terminus, the carotid T. A clot at the T occludes MCA and ACA territory at
# once and carries the worst prognosis of the LVO patterns, because there is no
# collateral route past it.
_register(Territory(
    name="ica_terminus_t",
    description="Carotid T occlusion: ICA terminus into M1 and A1",
    circulation="arterial",
    reference_diameter_mm=3.6,
    segments=[
        Segment("ica_terminal", diameter_mm=3.6, length_mm=15.0,
                diameter_distal_mm=3.4, tortuosity=0.08, bends=1,
                direction=(0.6, 0.0, 1.0)),
        Segment("m1_seg", diameter_mm=2.9, length_mm=18.0,
                parent="ica_terminal", angle_deg=62.0, rotation_deg=0.0,
                diameter_distal_mm=2.6, tortuosity=0.06, bends=1),
        Segment("a1_seg", diameter_mm=2.2, length_mm=14.0,
                parent="ica_terminal", angle_deg=68.0, rotation_deg=175.0,
                diameter_distal_mm=2.0, tortuosity=0.09, bends=1),
        Segment("m2_sup", diameter_mm=2.1, length_mm=26.0, parent="m1_seg",
                angle_deg=54.0, rotation_deg=15.0, diameter_distal_mm=1.6,
                tortuosity=0.11, bends=2),
        Segment("m2_inf", diameter_mm=2.0, length_mm=26.0, parent="m1_seg",
                angle_deg=58.0, rotation_deg=195.0, diameter_distal_mm=1.5,
                tortuosity=0.12, bends=2),
        Segment("a2_seg", diameter_mm=2.0, length_mm=28.0, parent="a1_seg",
                angle_deg=76.0, rotation_deg=10.0, diameter_distal_mm=1.6,
                tortuosity=0.12, bends=2),
    ],
    clot_sites=[
        ClotSite("ica_terminal", 0.92, "carotid T apex, occluding both outflows",
                 weight=3.5, occlusion=0.92),
        ClotSite("m1_seg", 0.30, "M1 just beyond the T", weight=2.0,
                 occlusion=0.9),
        ClotSite("a1_seg", 0.35, "A1 beyond the T", weight=1.2, occlusion=0.85),
        ClotSite("m1_seg", 0.90, "M1 bifurcation", weight=1.5),
    ],
    notes=(
        "TEXTBOOK NOMINAL, not freshly sourced. ICA terminus ~3.5mm splitting "
        "into M1 ~3mm and A1 ~2.2mm; T occlusion has no collateral bypass."
    ),
))

# Common carotid bifurcation. Plaque forms on the outer wall of the carotid bulb,
# opposite the flow divider: the bulb's expansion makes a recirculation zone
# there where wall shear is low and oscillatory. This is the classic low-shear
# atherogenic site.
_register(Territory(
    name="carotid_bifurcation",
    description="Carotid bifurcation with bulb plaque at the low-shear wall",
    circulation="arterial",
    reference_diameter_mm=7.0,
    segments=[
        Segment("cca", diameter_mm=7.0, length_mm=60.0, diameter_distal_mm=6.8,
                tortuosity=0.04, bends=1, direction=(0.05, 0.0, 1.0)),
        # The bulb is wider than the CCA it comes from -- one of the few places
        # in the arterial tree where the daughter dilates -- and that expansion
        # is what creates the recirculation zone.
        Segment("ica_bulb", diameter_mm=7.4, length_mm=20.0, parent="cca",
                angle_deg=18.0, rotation_deg=0.0, diameter_distal_mm=5.0,
                tortuosity=0.05, bends=1),
        Segment("ica_distal_cervical", diameter_mm=4.8, length_mm=55.0,
                parent="ica_bulb", angle_deg=12.0, rotation_deg=10.0,
                diameter_distal_mm=4.4, tortuosity=0.07, bends=1),
        Segment("eca", diameter_mm=4.2, length_mm=45.0, parent="cca",
                angle_deg=34.0, rotation_deg=180.0, diameter_distal_mm=3.4,
                tortuosity=0.08, bends=1),
        Segment("superior_thyroid", diameter_mm=1.8, length_mm=25.0,
                parent="eca", angle_deg=68.0, rotation_deg=250.0, origin=0.2,
                diameter_distal_mm=1.3, tortuosity=0.10),
        Segment("facial_artery", diameter_mm=2.2, length_mm=30.0, parent="eca",
                angle_deg=62.0, rotation_deg=80.0, origin=0.55,
                diameter_distal_mm=1.6, tortuosity=0.12, bends=2),
    ],
    stenoses=[
        # Bulb plaque, on the wall opposite the flow divider.
        Stenosis("ica_bulb", 0.45, residual_fraction=0.35, length_mm=15.0),
    ],
    clot_sites=[
        ClotSite("ica_bulb", 0.55, "thrombus on bulb plaque, low-shear wall",
                 weight=3.5, occlusion=0.85),
        ClotSite("ica_distal_cervical", 0.25, "propagation distal to the plaque",
                 weight=1.8, occlusion=0.8),
        ClotSite("cca", 0.90, "bifurcation apex", weight=1.0),
        ClotSite("eca", 0.20, "ECA origin", weight=0.6),
    ],
    notes=(
        "TEXTBOOK NOMINAL, not freshly sourced. CCA ~7mm, ICA bulb dilates to "
        "~7.4mm before narrowing to ~5mm; ECA ~4mm at ~34 deg. Plaque forms on "
        "the outer bulb wall opposite the flow divider (low, oscillatory shear)."
    ),
))

# Vertebrobasilar. Basilar occlusion is uncommon but has the highest mortality of
# the LVO patterns. The two vertebrals are usually asymmetric, one dominant, and
# they fuse into the basilar at a shallow angle.
_register(Territory(
    name="basilar_vertebral",
    description="Vertebrobasilar junction and basilar trunk occlusion",
    circulation="arterial",
    reference_diameter_mm=3.4,
    segments=[
        # Modelled from the dominant vertebral upward. The confluence is a merge,
        # which a tree cannot represent directly, so the non-dominant vertebral
        # is attached as a tributary rather than a true anastomosis.
        Segment("vertebral_dominant", diameter_mm=3.4, length_mm=40.0,
                diameter_distal_mm=3.2, tortuosity=0.12, bends=2,
                direction=(0.1, 0.0, 1.0)),
        Segment("basilar", diameter_mm=3.4, length_mm=30.0,
                parent="vertebral_dominant", angle_deg=16.0, rotation_deg=0.0,
                diameter_distal_mm=3.1, tortuosity=0.07, bends=1),
        Segment("vertebral_minor", diameter_mm=2.4, length_mm=35.0,
                parent="vertebral_dominant", angle_deg=28.0, rotation_deg=180.0,
                origin=0.85, diameter_distal_mm=2.2, tortuosity=0.13, bends=2),
        Segment("pca_p1_left", diameter_mm=2.3, length_mm=12.0, parent="basilar",
                angle_deg=68.0, rotation_deg=0.0, diameter_distal_mm=2.1,
                tortuosity=0.08),
        Segment("pca_p1_right", diameter_mm=2.2, length_mm=12.0, parent="basilar",
                angle_deg=72.0, rotation_deg=180.0, diameter_distal_mm=2.0,
                tortuosity=0.08),
        Segment("aica", diameter_mm=1.4, length_mm=22.0, parent="basilar",
                angle_deg=78.0, rotation_deg=95.0, origin=0.35,
                diameter_distal_mm=1.1, tortuosity=0.12, bends=2),
    ],
    clot_sites=[
        ClotSite("basilar", 0.45, "mid-basilar occlusion", weight=3.5,
                 occlusion=0.9),
        ClotSite("basilar", 0.92, "basilar apex, occluding both P1s", weight=2.0,
                 occlusion=0.9),
        ClotSite("vertebral_dominant", 0.75, "distal vertebral, V4", weight=1.5,
                 occlusion=0.85),
        ClotSite("pca_p1_left", 0.25, "P1 occlusion", weight=0.8),
    ],
    notes=(
        "TEXTBOOK NOMINAL, not freshly sourced. Basilar ~3.4mm and ~30mm; "
        "vertebrals usually asymmetric. The vertebral confluence is a merge, "
        "approximated here as a tributary."
    ),
))

# Cerebral venous sinus thrombosis. The sinuses are triangular dural channels,
# not round tubes, and they widen posteriorly, so a device travels from a narrow
# sigmoid into a wider transverse sinus. Thrombosis most often involves the
# superior sagittal and transverse sinuses.
_register(Territory(
    name="cerebral_venous_sinus",
    description="Dural venous sinuses; thrombosis of the transverse/sagittal",
    circulation="venous",
    reference_diameter_mm=9.0,
    retrograde=True,
    segments=[
        # Access is retrograde from the jugular, so the root is the sigmoid.
        Segment("sigmoid_sinus", diameter_mm=8.0, length_mm=45.0,
                diameter_distal_mm=8.5, tortuosity=0.18, bends=2,
                direction=(0.2, 0.0, 1.0)),
        Segment("transverse_sinus", diameter_mm=8.5, length_mm=60.0,
                parent="sigmoid_sinus", angle_deg=52.0, rotation_deg=0.0,
                diameter_distal_mm=7.5, tortuosity=0.06, bends=1),
        Segment("torcula", diameter_mm=9.0, length_mm=15.0,
                parent="transverse_sinus", angle_deg=20.0, rotation_deg=10.0,
                diameter_distal_mm=8.5, tortuosity=0.03),
        Segment("superior_sagittal", diameter_mm=8.0, length_mm=80.0,
                parent="torcula", angle_deg=48.0, rotation_deg=0.0,
                diameter_distal_mm=4.5, tortuosity=0.05, bends=1),
        Segment("straight_sinus", diameter_mm=5.0, length_mm=40.0,
                parent="torcula", angle_deg=56.0, rotation_deg=180.0,
                diameter_distal_mm=4.0, tortuosity=0.06),
        Segment("contralateral_transverse", diameter_mm=6.5, length_mm=55.0,
                parent="torcula", angle_deg=64.0, rotation_deg=95.0,
                diameter_distal_mm=5.5, tortuosity=0.07, bends=1),
    ],
    clot_sites=[
        ClotSite("superior_sagittal", 0.45, "SSS thrombosis", weight=3.0,
                 occlusion=0.85),
        ClotSite("transverse_sinus", 0.45, "transverse sinus thrombosis",
                 weight=3.0, occlusion=0.85),
        ClotSite("sigmoid_sinus", 0.60, "sigmoid extension", weight=1.8),
        ClotSite("torcula", 0.50, "confluence of sinuses", weight=1.2,
                 occlusion=0.88),
    ],
    notes=(
        "TEXTBOOK NOMINAL, not freshly sourced. Sinuses are triangular dural "
        "channels, approximated as round here; they widen posteriorly. SSS and "
        "transverse are the usual thrombosis sites."
    ),
))


# ================================================= peripheral / visceral arterial

# Superior mesenteric artery. Acute mesenteric ischaemia from SMA embolism
# characteristically spares the proximal few centimetres: the embolus travels
# down the SMA and lodges just distal to the origin of the middle colic artery,
# where the vessel narrows. That "spared proximal jejunum" pattern is the
# radiological signature, and it is a geometric consequence of where the calibre
# step is.
_register(Territory(
    name="sma_embolism",
    description="Superior mesenteric artery embolism distal to the middle colic",
    circulation="arterial",
    reference_diameter_mm=7.0,
    segments=[
        # The SMA leaves the aorta at an acute downward angle, which is itself
        # why emboli enter it preferentially over the more perpendicular renals.
        Segment("sma_origin", diameter_mm=7.0, length_mm=25.0,
                diameter_distal_mm=6.2, tortuosity=0.05,
                direction=(0.35, 0.0, -1.0)),
        Segment("sma_mid", diameter_mm=6.2, length_mm=45.0, parent="sma_origin",
                angle_deg=18.0, rotation_deg=10.0, diameter_distal_mm=5.0,
                tortuosity=0.09, bends=1),
        Segment("sma_distal", diameter_mm=5.0, length_mm=50.0, parent="sma_mid",
                angle_deg=15.0, rotation_deg=15.0, diameter_distal_mm=3.6,
                tortuosity=0.10, bends=2),
        Segment("middle_colic", diameter_mm=3.0, length_mm=40.0,
                parent="sma_mid", angle_deg=55.0, rotation_deg=90.0,
                origin=0.30, diameter_distal_mm=2.2, tortuosity=0.08),
        Segment("jejunal_1", diameter_mm=2.8, length_mm=45.0, parent="sma_mid",
                angle_deg=62.0, rotation_deg=260.0, origin=0.65,
                diameter_distal_mm=2.0, tortuosity=0.11, bends=2),
        Segment("ileocolic", diameter_mm=3.2, length_mm=42.0,
                parent="sma_distal", angle_deg=48.0, rotation_deg=80.0,
                diameter_distal_mm=2.4, tortuosity=0.09),
    ],
    clot_sites=[
        # Just distal to the middle colic origin: the classic embolic lodgement.
        ClotSite("sma_mid", 0.40, "distal to middle colic origin, calibre step",
                 weight=3.5, occlusion=0.9),
        ClotSite("sma_distal", 0.25, "distal SMA embolus", weight=2.0,
                 occlusion=0.85),
        ClotSite("sma_origin", 0.15, "ostial thrombosis on plaque", weight=1.2,
                 occlusion=0.9),
        ClotSite("ileocolic", 0.20, "branch ostium", weight=0.8),
    ],
    notes=(
        "SMA 6-8mm at origin, acute caudal take-off from the aorta. Embolic "
        "occlusion typically lodges just distal to the middle colic origin, "
        "sparing the proximal jejunum."
    ),
))

# Femoropopliteal arterial segment. The superficial femoral artery in the
# adductor canal is the classic site of peripheral arterial occlusion: it is
# tethered where it passes through the adductor hiatus, and both the fixation
# and the calibre step there promote thrombosis on existing plaque.
_register(Territory(
    name="femoropopliteal_pad",
    description="Superficial femoral artery through the adductor canal",
    circulation="arterial",
    reference_diameter_mm=8.0,
    segments=[
        Segment("common_femoral", diameter_mm=8.0, length_mm=40.0,
                diameter_distal_mm=7.4, tortuosity=0.03,
                direction=(0.05, 0.0, -1.0)),
        Segment("superficial_femoral", diameter_mm=6.0, length_mm=90.0,
                parent="common_femoral", angle_deg=12.0, rotation_deg=0.0,
                diameter_distal_mm=5.2, tortuosity=0.06, bends=1),
        # Profunda leaves posterolaterally at a wide angle and is the collateral
        # that keeps the limb alive when the SFA occludes.
        Segment("profunda_femoris", diameter_mm=5.0, length_mm=70.0,
                parent="common_femoral", angle_deg=42.0, rotation_deg=200.0,
                origin=0.85, diameter_distal_mm=3.6, tortuosity=0.08),
        Segment("popliteal_artery", diameter_mm=5.2, length_mm=60.0,
                parent="superficial_femoral", angle_deg=14.0, rotation_deg=15.0,
                diameter_distal_mm=4.6, tortuosity=0.07, bends=1),
        Segment("anterior_tibial_artery", diameter_mm=3.0, length_mm=70.0,
                parent="popliteal_artery", angle_deg=34.0, rotation_deg=30.0,
                diameter_distal_mm=2.2, tortuosity=0.06),
        Segment("tibioperoneal_trunk", diameter_mm=3.6, length_mm=35.0,
                parent="popliteal_artery", angle_deg=26.0, rotation_deg=190.0,
                diameter_distal_mm=3.0, tortuosity=0.05),
    ],
    stenoses=[
        # Adductor hiatus, at the distal SFA.
        Stenosis("superficial_femoral", 0.75, residual_fraction=0.35,
                 length_mm=30.0),
    ],
    clot_sites=[
        ClotSite("superficial_femoral", 0.80, "adductor canal, on plaque",
                 weight=3.5, occlusion=0.9),
        ClotSite("popliteal_artery", 0.30, "popliteal embolus", weight=2.0,
                 occlusion=0.88),
        ClotSite("superficial_femoral", 0.35, "mid-SFA thrombosis", weight=1.5),
        ClotSite("common_femoral", 0.85, "femoral bifurcation embolus",
                 weight=1.2, occlusion=0.85),
    ],
    notes=(
        "SFA 5-6mm, tethered at the adductor hiatus where occlusion "
        "preferentially occurs; profunda femoris is the collateral route."
    ),
))

# Renal artery. Short, leaves the aorta near-perpendicular, and the ostium is
# the site of both atherosclerotic disease and embolic lodgement.
_register(Territory(
    name="renal_artery",
    description="Renal artery with ostial disease and segmental branches",
    circulation="arterial",
    reference_diameter_mm=5.5,
    segments=[
        Segment("renal_main", diameter_mm=5.5, length_mm=40.0,
                diameter_distal_mm=4.8, tortuosity=0.04,
                direction=(1.0, 0.15, -0.1)),
        Segment("anterior_division", diameter_mm=3.8, length_mm=30.0,
                parent="renal_main", angle_deg=32.0, rotation_deg=0.0,
                diameter_distal_mm=3.0, tortuosity=0.07),
        Segment("posterior_division", diameter_mm=3.2, length_mm=28.0,
                parent="renal_main", angle_deg=40.0, rotation_deg=180.0,
                diameter_distal_mm=2.6, tortuosity=0.07),
        Segment("upper_segmental", diameter_mm=2.6, length_mm=25.0,
                parent="anterior_division", angle_deg=45.0, rotation_deg=60.0,
                diameter_distal_mm=2.0, tortuosity=0.08),
        Segment("lower_segmental", diameter_mm=2.5, length_mm=25.0,
                parent="anterior_division", angle_deg=48.0, rotation_deg=250.0,
                diameter_distal_mm=1.9, tortuosity=0.08),
        Segment("posterior_segmental", diameter_mm=2.3, length_mm=22.0,
                parent="posterior_division", angle_deg=42.0, rotation_deg=140.0,
                diameter_distal_mm=1.8, tortuosity=0.07),
    ],
    stenoses=[
        Stenosis("renal_main", 0.10, residual_fraction=0.40, length_mm=12.0),
    ],
    clot_sites=[
        ClotSite("renal_main", 0.18, "ostial, on atherosclerotic plaque",
                 weight=3.0, occlusion=0.9),
        ClotSite("renal_main", 0.85, "pre-hilar bifurcation embolus", weight=2.0,
                 occlusion=0.85),
        ClotSite("anterior_division", 0.25, "segmental embolus, infarct wedge",
                 weight=1.5),
        ClotSite("posterior_division", 0.25, "segmental embolus", weight=1.0),
    ],
    notes="Renal artery 5-6mm, short and near-perpendicular; ostial plaque common.",
))


# --------------------------------------------------------------- geometry build


def _frame(direction) -> tuple:
    """Orthonormal frame with its first axis along `direction`."""
    import numpy as np

    d = np.asarray(direction, dtype=np.float64)
    n = np.linalg.norm(d)
    d = d / n if n > 1e-12 else np.array([1.0, 0.0, 0.0])
    seed = np.array([0.0, 0.0, 1.0])
    if abs(float(d @ seed)) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    u = np.cross(d, seed)
    u /= max(np.linalg.norm(u), 1e-12)
    v = np.cross(d, u)
    return d, u, v


def _curve(start, direction, length: float, tortuosity: float, bends: int,
           rng, jitter: float = 0.0):
    """Centerline for one segment: a bowed or S-bent curve of the right length.

    The curve is built as control points offset perpendicular to the chord, then
    splined. `bends` sets how many curvature reversals: 1 bows one way, 2 gives
    the S-shape a carotid siphon needs. The chord is shortened so the *arclength*
    matches `length_mm`, since a tortuous vessel's centerline is longer than the
    straight-line distance between its ends -- which is exactly what the
    morphometry tables report.
    """
    import numpy as np

    from environments.vessel_tree_generator import catmull_rom_spline

    d, u, v = _frame(direction)
    start = np.asarray(start, dtype=np.float64)

    if tortuosity < 1e-6:
        end = start + length * d
        return np.stack([start, end]).astype(np.float32)

    # Chord shortening: for a sinusoidal centerline of amplitude a over chord L,
    # arclength ~= L * (1 + (pi*a*bends/L)^2 / 4). Invert that to first order so
    # the sampled curve has the length the anatomy specifies.
    amp = tortuosity * length
    chord = length / (1.0 + (np.pi * amp * bends / max(length, 1e-9)) ** 2 / 4.0)

    n_ctrl = 2 * max(int(bends), 1) + 1
    phase0 = float(rng.uniform(0.0, 2.0 * np.pi)) if jitter > 0 else 0.0
    ctrl = []
    for i in range(n_ctrl + 1):
        t = i / n_ctrl
        # Sine envelope: zero at both ends, so the segment leaves and arrives
        # along its nominal direction rather than kinking at the junction.
        off = amp * np.sin(np.pi * bends * t) * np.cos(phase0)
        off_v = amp * np.sin(np.pi * bends * t) * np.sin(phase0)
        p = start + chord * t * d + off * u + off_v * v
        if jitter > 0 and 0 < i < n_ctrl:
            p = p + rng.normal(0.0, jitter * length, 3)
        ctrl.append(p)
    return catmull_rom_spline(np.asarray(ctrl, dtype=np.float32),
                              n_samples=max(12, 4 * n_ctrl))


def build_territory(name: str, rng=None, variation: float = 1.0,
                    min_radius: float = 0.0045, stations_per_mm: float = 0.9):
    """Build a `VesselTree` for one anatomical territory.

    Args:
        name: key into TERRITORIES.
        rng: generator for the per-episode anatomical variation.
        variation: scales the sampled deviation from the mean anatomy. 0 builds
            the textbook mean; 1 uses the reported standard deviations. Every
            territory's numbers come with real spread (LM length 10.5+/-5.3mm),
            so training on the mean alone would fit one patient.
        stations_per_mm: centerline resolution. Fixed per millimetre rather than
            per segment, so a 117mm right pulmonary artery gets proportionally
            more stations than a 20mm M1 and the station spacing -- which sets
            geodesic edge weights and lookahead offsets -- is comparable across
            territories.

    Returns:
        A fully constructed `VesselTree` normalised into the unit cube.
    """
    import numpy as np

    from environments.vessel_tree_generator import (
        VesselSegment,
        segments_to_vessel_tree,
    )

    try:
        terr = TERRITORIES[name]
    except KeyError:
        raise ValueError(
            f"unknown territory {name!r}; have {sorted(TERRITORIES)}"
        ) from None

    rng = rng if rng is not None else np.random.default_rng()
    var = float(max(variation, 0.0))

    def wobble(value: float, frac: float) -> float:
        """Sample a value with `frac` relative spread, clipped positive."""
        if var <= 0.0 or frac <= 0.0:
            return value
        return float(max(value * (1.0 + frac * var * rng.normal()), 1e-6))

    # Lay out segments parent-first so a child can read its parent's geometry.
    order: list[Segment] = []
    seen: set[str] = set()
    by_name = {s.name: s for s in terr.segments}

    def visit(seg: Segment) -> None:
        if seg.name in seen:
            return
        if seg.parent is not None:
            parent = by_name.get(seg.parent)
            if parent is None:
                raise ValueError(
                    f"{terr.name}: segment {seg.name!r} names unknown parent "
                    f"{seg.parent!r}"
                )
            visit(parent)
        seen.add(seg.name)
        order.append(seg)

    for seg in terr.segments:
        visit(seg)

    curves: dict[str, np.ndarray] = {}
    index: dict[str, int] = {}
    built: list[VesselSegment] = []

    for seg in order:
        length = wobble(seg.length_mm, 0.18)
        d_prox = wobble(seg.diameter_mm, 0.12)
        d_dist = (d_prox if seg.diameter_distal_mm is None
                  else wobble(seg.diameter_distal_mm, 0.12))

        if seg.parent is None:
            start = np.zeros(3)
            direction = np.asarray(seg.direction or (1.0, 0.0, 0.0), float)
            parent_idx = None
            generation = 0
        else:
            pcurve = curves[seg.parent]
            # Take-off point: `origin` along the parent, so a side branch leaves
            # mid-segment rather than only at a terminal bifurcation.
            k = int(round(np.clip(seg.origin, 0.0, 1.0) * (pcurve.shape[0] - 1)))
            start = pcurve[k].astype(np.float64)
            tangent = (pcurve[min(k + 1, pcurve.shape[0] - 1)]
                       - pcurve[max(k - 1, 0)]).astype(np.float64)
            d, u, v = _frame(tangent)
            angle = np.radians(wobble(seg.angle_deg, 0.0)
                               + (var * 8.0 * rng.normal() if var > 0 else 0.0))
            roll = np.radians(seg.rotation_deg)
            direction = (np.cos(angle) * d
                         + np.sin(angle) * (np.cos(roll) * u + np.sin(roll) * v))
            parent_idx = index[seg.parent]
            generation = built[parent_idx].generation + 1
            if seg.direction is not None:
                direction = np.asarray(seg.direction, float)

        curve = _curve(start, direction, length, seg.tortuosity,
                       seg.bends, rng, jitter=0.02 * var)
        curves[seg.name] = curve
        index[seg.name] = len(built)
        built.append(VesselSegment(
            start=curve[0].astype(np.float32),
            end=curve[-1].astype(np.float32),
            radius_prox=0.5 * d_prox,
            radius_dist=0.5 * d_dist,
            parent_idx=parent_idx,
            generation=generation,
            tortuosity=seg.tortuosity,
            curve=curve,
            n_stations=max(6, int(round(length * stations_per_mm))),
        ))

    tree = segments_to_vessel_tree(
        built, scenario=name, min_radius=min_radius, fit_unit_cube=True,
    )
    # Record what the anatomy meant, so a consumer can convert back to mm and so
    # clot placement can target named sites instead of arbitrary stations.
    tree.territory = terr
    tree.segment_index = dict(index)
    tree.mm_per_unit = _mm_per_unit(terr, tree)
    # Segment inputs above are in millimetres. Preserve their actual scale for
    # explicit-unit consumers; leave the historical nominal-calibre field
    # unchanged so archived workers/checkpoints keep their existing semantics.
    tree.physical_mm_per_unit = tree.source_units_per_unit
    # Stenoses run after normalisation, so they would otherwise undercut the
    # `min_radius` floor that `segments_to_vessel_tree` already applied: a 60%
    # diameter stenosis on an already-small distal vessel left lumina at 1.0x
    # the device radius, which no device can pass. Re-imposing the floor keeps
    # every lesion navigable while preserving its position and extent.
    _apply_stenoses(tree, terr, rng, var, min_radius=min_radius)
    return tree


def _mm_per_unit(terr: Territory, tree) -> float:
    """Millimetres per simulator length unit, from the reference calibre."""
    ref = terr.reference_diameter_mm
    if ref <= 0.0:
        return 0.0
    import numpy as np

    return float(ref / max(2.0 * float(np.max(tree.radii)), 1e-9))


def _apply_stenoses(tree, terr: Territory, rng, var: float,
                    min_radius: float = 0.0) -> None:
    """Narrow the lumen where the territory says a lesion sits.

    Applied after normalisation so `residual_fraction` is a fraction of the
    local calibre, matching how a diameter stenosis is reported clinically.
    """
    if not terr.stenoses:
        return
    import numpy as np

    for lesion in terr.stenoses:
        bid = tree.segment_index.get(lesion.segment)
        if bid is None:
            continue
        br = tree.branches[bid]
        idx = np.arange(br.start, br.stop + 1)
        if idx.size < 3:
            continue
        centre = br.start + np.clip(lesion.position, 0.0, 1.0) * (br.size - 1)
        # Station spacing from the geometry itself. Taking it as the branch's
        # end-to-end arclength over its station count is wrong on a branch that
        # is not the inlet: `arclength` is distance from the tree inlet along the
        # graph, so the difference across a branch is only its own length when
        # the path to it happens to be monotone. Measuring adjacent stations
        # directly is exact either way.
        pts = tree.points[br.start: br.stop + 1]
        steps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        per_station = float(steps.mean()) if steps.size else 1.0
        mm_per_unit = tree.mm_per_unit or 1.0
        length_stations = (lesion.length_mm / max(mm_per_unit, 1e-9)
                           / max(per_station, 1e-9))
        # `length_mm` is the visible extent of the narrowing, as a lesion is
        # measured clinically -- not a Gaussian sigma. A Gaussian is within 10%
        # of baseline out to about +/-2.15 sigma, so the sigma that produces a
        # lesion of the stated length is length/(2*2.15). Treating length as
        # sigma directly made a declared 12mm lesion measure 30mm.
        width = max(length_stations / 4.3, 1.0)
        residual = float(np.clip(
            lesion.residual_fraction * (1.0 + 0.10 * var * rng.normal()),
            0.08, 0.98,
        ))
        bump = np.exp(-0.5 * np.square((idx - centre) / width))
        narrowed = tree.radii[idx] * (1.0 - (1.0 - residual) * bump)
        tree.radii[idx] = np.maximum(narrowed, min_radius)
