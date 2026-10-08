ACTIVE_SV_THICKNESS_MM = 0.001  # 1 um Geant4 active sensitive volumes

# Packages are a silica-filled epoxy mold compound and boards are glass-epoxy
# FR-4, both defined in the Geant4 bridge. These are generic estimates, not
# specific parts.
PACKAGE_MATERIAL = "MoldCompound"
BOARD_MATERIAL = "FR4"

# Back-end-of-line metal stack on the transistor surface, as slabs from the
# silicon outward: (layer, material, thickness in um). A generic ~9.5 um
# multi-level copper process; each metal slab holds the metal of its levels at
# a typical fill (~40% Cu, ~15% W contacts), the rest of each level is oxide.
BEOL_LAYERS_UM = (
    ("Contact", "G4_W", 0.08),
    ("ContactOxide", "G4_SILICON_DIOXIDE", 0.42),
    ("LocalCu", "G4_Cu", 0.5),
    ("LocalOxide", "G4_SILICON_DIOXIDE", 2.5),
    ("IntermediateCu", "G4_Cu", 0.7),
    ("IntermediateOxide", "G4_SILICON_DIOXIDE", 1.3),
    ("TopCu", "G4_Cu", 1.2),
    ("Passivation", "G4_SILICON_DIOXIDE", 2.8),
)
BEOL_THICKNESS_MM = sum(layer[2] for layer in BEOL_LAYERS_UM) * 1.0e-3

# Flip-chip interconnect between the metal stack and the package substrate:
# solder bumps as an equivalent tin sheet (~25% of ~80 um bumps), then underfill.
BUMP_THICKNESS_MM = 0.020
UNDERFILL_THICKNESS_MM = 0.050


def _device_component(name, material, center_mm, size_mm, parent="", score=False):
    return {
        "name": name,
        "material": material,
        "parent": parent,
        "center_mm": center_mm,
        "size_mm": size_mm,
        "score": score,
    }


def _die(name, center_xy, die_size_mm, die_bottom_z, active_size_mm, package, flip=False):
    """Die, its 1 um SV at the transistor surface, and the metal stack on it.

    Face-up (wire-bonded) dies have the transistor surface on top; flip-chip
    dies face the board, so the SV and metal stack are on the die bottom.
    """
    x, y = center_xy
    die_name = f"{name}Die"
    die_top_z = die_bottom_z + die_size_mm[2]
    surface_z, direction = (die_bottom_z, -1.0) if flip else (die_top_z, 1.0)
    components = [
        _device_component(
            die_name,
            "G4_Si",
            [x, y, die_bottom_z + 0.5 * die_size_mm[2]],
            die_size_mm,
            package,
        ),
        _device_component(
            f"{name}Active",
            "G4_Si",
            [x, y, surface_z - direction * 0.5 * ACTIVE_SV_THICKNESS_MM],
            [*active_size_mm[:2], ACTIVE_SV_THICKNESS_MM],
            die_name,
            score=True,
        ),
    ]
    z = surface_z
    for layer, material, thickness_um in BEOL_LAYERS_UM:
        thickness = thickness_um * 1.0e-3
        components.append(
            _device_component(
                f"{name}BEOL{layer}",
                material,
                [x, y, z + direction * 0.5 * thickness],
                [die_size_mm[0], die_size_mm[1], thickness],
                package,
            )
        )
        z += direction * thickness
    return components


def _packaged_ic(name, center_mm, package_size_mm, die_size_mm, active_size_mm, flip=False):
    package_name = f"{name}Package"
    center_x, center_y, center_z = center_mm
    package_bottom = center_z - 0.5 * package_size_mm[2]
    die_bottom = package_bottom + 0.25
    components = [
        _device_component(package_name, PACKAGE_MATERIAL, center_mm, package_size_mm),
        *_die(
            name,
            (center_x, center_y),
            die_size_mm,
            die_bottom,
            active_size_mm,
            package_name,
            flip,
        ),
    ]
    if flip:
        # bumps and underfill under the metal stack, on an FR-4 package substrate
        bump_top = die_bottom - BEOL_THICKNESS_MM
        underfill_top = bump_top - BUMP_THICKNESS_MM
        substrate_top = underfill_top - UNDERFILL_THICKNESS_MM
        for part, material, top, thickness in (
            ("Bumps", "G4_Sn", bump_top, BUMP_THICKNESS_MM),
            ("Underfill", PACKAGE_MATERIAL, underfill_top, UNDERFILL_THICKNESS_MM),
        ):
            components.append(
                _device_component(
                    f"{name}{part}",
                    material,
                    [center_x, center_y, top - 0.5 * thickness],
                    [die_size_mm[0], die_size_mm[1], thickness],
                    package_name,
                )
            )
        components.append(
            _device_component(
                f"{name}Substrate",
                BOARD_MATERIAL,
                [center_x, center_y, 0.5 * (package_bottom + substrate_top)],
                [package_size_mm[0], package_size_mm[1], substrate_top - package_bottom],
                package_name,
            )
        )
    return components


def _eps_control_ic(name, center_mm, package_size_mm, die_size_mm, active_size_mm):
    package_name = f"{name}Package"
    center_x, center_y, _ = center_mm
    return [
        _device_component(package_name, PACKAGE_MATERIAL, center_mm, package_size_mm),
        *_die(name, (center_x, center_y), die_size_mm, 0.15, active_size_mm, package_name),
    ]


def build_detector_sizes_mm():
    return {
        "obc": (36.0, 27.0, 6.0),
        "eps": (25.0, 65.0, 2.3),
        "adcs": (45.0, 45.0, 18.0),
        "comms": (11.0, 9.7, 2.7),
    }


def build_device_components():
    return {
        "obc": [
            _device_component(
                "OBC_Board",
                BOARD_MATERIAL,
                [0.0, 0.0, -2.65],
                [34.0, 25.0, 0.50],
            ),
            _device_component(
                "OBC_GroundPlane",
                "G4_Cu",
                [0.0, 0.0, -2.35],
                [32.0, 23.0, 0.05],
            ),
            # Board-level OBC model: processor, volatile memory, nonvolatile
            # memory, and support logic are independent SEE-sensitive targets.
            # The processor and FPGA are flip-chip, the memories wire-bonded.
            *_packaged_ic(
                "OBC_Processor",
                [-7.0, -4.0, -1.45],
                [12.0, 12.0, 1.60],
                [8.0, 8.0, 0.35],
                [6.5, 6.5],
                flip=True,
            ),
            *_packaged_ic(
                "OBC_SRAM",
                [7.0, -5.0, -1.65],
                [8.0, 6.0, 1.20],
                [5.5, 4.0, 0.30],
                [4.5, 3.0],
            ),
            *_packaged_ic(
                "OBC_Flash",
                [7.0, 5.0, -1.65],
                [8.0, 6.0, 1.20],
                [5.5, 4.0, 0.30],
                [4.5, 3.0],
            ),
            *_packaged_ic(
                "OBC_FPGA",
                [-7.0, 7.5, -1.55],
                [9.0, 8.0, 1.40],
                [6.0, 5.0, 0.30],
                [5.0, 4.0],
                flip=True,
            ),
        ],
        "eps": [
            # EPS electronics board inside the coarse MCDC EPS handoff volume;
            # the copper planes provide passive shielding and routing material.
            _device_component(
                "EPS_Board",
                BOARD_MATERIAL,
                [0.0, 0.0, -0.45],
                [24.0, 63.0, 0.70],
            ),
            _device_component(
                "EPS_GroundPlane",
                "G4_Cu",
                [0.0, 0.0, -0.075],
                [23.0, 62.0, 0.05],
            ),
            # Scored silicon active layers represent battery-management logic
            # whose upset could disable or miscommand the battery system.
            *_eps_control_ic(
                "EPS_Protection",
                [-6.0, -20.0, 0.35],
                [7.0, 7.0, 0.70],
                [4.5, 4.5, 0.30],
                [3.5, 3.5],
            ),
            *_eps_control_ic(
                "EPS_Charger",
                [6.0, -20.0, 0.35],
                [7.0, 7.0, 0.70],
                [4.5, 4.5, 0.30],
                [3.5, 3.5],
            ),
            *_eps_control_ic(
                "EPS_FuelGauge",
                [-5.0, 16.0, 0.35],
                [6.0, 6.0, 0.70],
                [3.5, 3.5, 0.30],
                [2.5, 2.5],
            ),
            *_eps_control_ic(
                "EPS_Controller",
                [5.0, 16.0, 0.35],
                [8.0, 8.0, 0.70],
                [5.0, 5.0, 0.30],
                [4.0, 4.0],
            ),
            # A small unscored pouch-cell coupon gives local battery material
            # context without making bulk cell edep the EPS SEE metric.
            _device_component(
                "EPS_BatteryCouponCathode",
                "LiCoO2",
                [0.0, 29.0, 0.15],
                [22.0, 5.0, 0.25],
            ),
            _device_component(
                "EPS_BatteryCouponAnode",
                "G4_GRAPHITE",
                [0.0, 29.0, 0.45],
                [22.0, 5.0, 0.25],
            ),
        ],
        "adcs": [
            _device_component(
                "ADCS_HousingBottom",
                "G4_Al",
                [0.0, 0.0, -7.5],
                [42.0, 42.0, 1.0],
            ),
            _device_component(
                "ADCS_HousingTop", "G4_Al", [0.0, 0.0, 7.5], [42.0, 42.0, 1.0]
            ),
            _device_component(
                "ADCS_HousingLeft", "G4_Al", [-20.5, 0.0, 0.0], [1.0, 42.0, 14.0]
            ),
            _device_component(
                "ADCS_HousingRight", "G4_Al", [20.5, 0.0, 0.0], [1.0, 42.0, 14.0]
            ),
            _device_component(
                "ADCS_HousingFront",
                "G4_Al",
                [0.0, -20.5, 0.0],
                [40.0, 1.0, 14.0],
            ),
            _device_component(
                "ADCS_HousingBack",
                "G4_Al",
                [0.0, 20.5, 0.0],
                [40.0, 1.0, 14.0],
            ),
            _device_component(
                "ADCS_Substrate",
                "G4_ALUMINUM_OXIDE",
                [0.0, 0.0, -5.5],
                [36.0, 32.0, 1.0],
            ),
            _device_component(
                "ADCS_GroundPlane",
                "G4_Cu",
                [0.0, 0.0, -4.9],
                [36.0, 32.0, 0.2],
            ),
            _device_component(
                "ADCS_ControllerPackage",
                PACKAGE_MATERIAL,
                [-10.0, 0.0, -3.7],
                [12.0, 12.0, 2.0],
            ),
            *_die(
                "ADCS_Controller",
                (-10.0, 0.0),
                [8.0, 8.0, 0.5],
                -4.6,
                [6.0, 6.0],
                "ADCS_ControllerPackage",
            ),
            _device_component(
                "ADCS_MEMSPackage",
                PACKAGE_MATERIAL,
                [4.0, 0.0, -3.2],
                [10.0, 10.0, 3.0],
            ),
            _device_component(
                "ADCS_MEMSCavity",
                "G4_Galactic",
                [4.0, 0.0, -2.7],
                [8.0, 8.0, 1.6],
                "ADCS_MEMSPackage",
            ),
            *_die(
                "ADCS_MEMS",
                (4.0, 0.0),
                [6.0, 6.0, 0.5],
                -4.05,
                [4.0, 4.0],
                "ADCS_MEMSPackage",
            ),
            _device_component(
                "ADCS_MagnetometerPackage",
                PACKAGE_MATERIAL,
                [13.0, 0.0, -3.8],
                [6.0, 6.0, 1.8],
            ),
            *_die(
                "ADCS_Magnetometer",
                (13.0, 0.0),
                [3.0, 3.0, 0.5],
                -4.6,
                [2.0, 2.0],
                "ADCS_MagnetometerPackage",
            ),
            *_packaged_ic(
                "ADCS_ActuatorDriver",
                [0.0, 12.0, -3.7],
                [10.0, 8.0, 1.4],
                [6.0, 4.0, 0.30],
                [5.0, 3.0],
            ),
        ],
        "comms": [
            _device_component(
                "Comms_Package", PACKAGE_MATERIAL, [0.0, 0.0, 0.0], [10.5, 9.2, 2.4]
            ),
            _device_component(
                "Comms_Solder",
                "G4_Sn",
                [0.0, 0.0, -1.16],
                [8.0, 7.0, 0.06],
                "Comms_Package",
            ),
            _device_component(
                "Comms_ExposedPad",
                "G4_Cu",
                [0.0, 0.0, -1.08],
                [8.0, 7.0, 0.08],
                "Comms_Package",
            ),
            _device_component(
                "Comms_LeadLeft",
                "G4_Cu",
                [-4.75, 0.0, -0.95],
                [0.5, 8.7, 0.1],
                "Comms_Package",
            ),
            _device_component(
                "Comms_LeadRight",
                "G4_Cu",
                [4.75, 0.0, -0.95],
                [0.5, 8.7, 0.1],
                "Comms_Package",
            ),
            _device_component(
                "Comms_LeadFront",
                "G4_Cu",
                [0.0, -4.1, -0.95],
                [9.0, 0.5, 0.1],
                "Comms_Package",
            ),
            _device_component(
                "Comms_LeadBack",
                "G4_Cu",
                [0.0, 4.1, -0.95],
                [9.0, 0.5, 0.1],
                "Comms_Package",
            ),
            *_die(
                "Comms_RF",
                (-2.0, 0.0),
                [3.0, 3.2, 0.35],
                -0.825,
                [2.3, 2.4],
                "Comms_Package",
            ),
            *_die(
                "Comms_Baseband",
                (1.8, -1.8),
                [3.0, 2.4, 0.35],
                -0.825,
                [2.3, 1.8],
                "Comms_Package",
            ),
            *_die(
                "Comms_Control",
                (1.8, 1.8),
                [2.6, 2.2, 0.35],
                -0.825,
                [2.0, 1.6],
                "Comms_Package",
            ),
        ],
    }
