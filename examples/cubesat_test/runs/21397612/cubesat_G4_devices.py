def _device_component(name, material, center_mm, size_mm, parent="", score=False):
    return {
        "name": name,
        "material": material,
        "parent": parent,
        "center_mm": center_mm,
        "size_mm": size_mm,
        "score": score,
    }


def _packaged_ic(name, center_mm, package_size_mm, die_size_mm, active_size_mm):
    package_name = f"{name}Package"
    die_name = f"{name}Die"
    center_x, center_y, center_z = center_mm
    package_bottom = center_z - 0.5 * package_size_mm[2]
    die_center_z = package_bottom + 0.25 + 0.5 * die_size_mm[2]
    active_center_z = die_center_z + 0.5 * die_size_mm[2] - 0.5 * active_size_mm[2] - 0.01
    return [
        _device_component(package_name, "G4_BAKELITE", center_mm, package_size_mm),
        _device_component(
            die_name,
            "G4_Si",
            [center_x, center_y, die_center_z],
            die_size_mm,
            package_name,
        ),
        _device_component(
            f"{name}Active",
            "G4_Si",
            [center_x, center_y, active_center_z],
            active_size_mm,
            die_name,
            score=True,
        ),
    ]


def _eps_control_ic(name, center_mm, package_size_mm, die_size_mm, active_size_mm):
    package_name = f"{name}Package"
    die_name = f"{name}Die"
    center_x, center_y, _ = center_mm
    return [
        _device_component(package_name, "G4_BAKELITE", center_mm, package_size_mm),
        _device_component(
            die_name,
            "G4_Si",
            [center_x, center_y, 0.30],
            die_size_mm,
            package_name,
        ),
        _device_component(
            f"{name}Active",
            "G4_Si",
            [center_x, center_y, 0.415],
            active_size_mm,
            die_name,
            score=True,
        ),
    ]


def build_detector_sizes_mm():
    return {
        "obc": (36.0, 27.0, 6.0),
        "eps": (25.0, 65.0, 2.3),
        "adcs": (45.0, 45.0, 18.0),
        "comms": (11.0, 9.7, 2.7),
    }


def build_device_components():
    # G4_BAKELITE is a built-in Geant4 resin proxy for package epoxy. It is more
    # realistic than polyethylene for molded packages, but it is not a filled
    # commercial mold compound.
    return {
        "obc": [
            _device_component(
                "OBC_Board",
                "G4_BAKELITE",
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
            *_packaged_ic(
                "OBC_Processor",
                [-7.0, -4.0, -1.45],
                [12.0, 12.0, 1.60],
                [8.0, 8.0, 0.35],
                [6.5, 6.5, 0.05],
            ),
            *_packaged_ic(
                "OBC_SRAM",
                [7.0, -5.0, -1.65],
                [8.0, 6.0, 1.20],
                [5.5, 4.0, 0.30],
                [4.5, 3.0, 0.05],
            ),
            *_packaged_ic(
                "OBC_Flash",
                [7.0, 5.0, -1.65],
                [8.0, 6.0, 1.20],
                [5.5, 4.0, 0.30],
                [4.5, 3.0, 0.05],
            ),
            *_packaged_ic(
                "OBC_FPGA",
                [-7.0, 7.5, -1.55],
                [9.0, 8.0, 1.40],
                [6.0, 5.0, 0.30],
                [5.0, 4.0, 0.05],
            ),
        ],
        "eps": [
            # EPS electronics board inside the coarse MCDC EPS handoff volume.
            # G4_BAKELITE is used as an FR-4/package proxy; the copper planes
            # provide passive shielding and routing material.
            _device_component(
                "EPS_Board",
                "G4_BAKELITE",
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
                [3.5, 3.5, 0.05],
            ),
            *_eps_control_ic(
                "EPS_Charger",
                [6.0, -20.0, 0.35],
                [7.0, 7.0, 0.70],
                [4.5, 4.5, 0.30],
                [3.5, 3.5, 0.05],
            ),
            *_eps_control_ic(
                "EPS_FuelGauge",
                [-5.0, 16.0, 0.35],
                [6.0, 6.0, 0.70],
                [3.5, 3.5, 0.30],
                [2.5, 2.5, 0.05],
            ),
            *_eps_control_ic(
                "EPS_Controller",
                [5.0, 16.0, 0.35],
                [8.0, 8.0, 0.70],
                [5.0, 5.0, 0.30],
                [4.0, 4.0, 0.05],
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
                "G4_BAKELITE",
                [-10.0, 0.0, -3.7],
                [12.0, 12.0, 2.0],
            ),
            _device_component(
                "ADCS_ControllerDie",
                "G4_Si",
                [-10.0, 0.0, -4.35],
                [8.0, 8.0, 0.5],
                "ADCS_ControllerPackage",
            ),
            _device_component(
                "ADCS_ControllerActive",
                "G4_Si",
                [-10.0, 0.0, -4.135],
                [6.0, 6.0, 0.05],
                "ADCS_ControllerDie",
                True,
            ),
            _device_component(
                "ADCS_MEMSPackage",
                "G4_BAKELITE",
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
            _device_component(
                "ADCS_MEMSDie",
                "G4_Si",
                [4.0, 0.0, -3.8],
                [6.0, 6.0, 0.5],
                "ADCS_MEMSPackage",
            ),
            _device_component(
                "ADCS_MEMSActive",
                "G4_Si",
                [4.0, 0.0, -3.585],
                [4.0, 4.0, 0.05],
                "ADCS_MEMSDie",
                True,
            ),
            _device_component(
                "ADCS_MagnetometerPackage",
                "G4_BAKELITE",
                [13.0, 0.0, -3.8],
                [6.0, 6.0, 1.8],
            ),
            _device_component(
                "ADCS_MagnetometerDie",
                "G4_Si",
                [13.0, 0.0, -4.35],
                [3.0, 3.0, 0.5],
                "ADCS_MagnetometerPackage",
            ),
            _device_component(
                "ADCS_MagnetometerActive",
                "G4_Si",
                [13.0, 0.0, -4.16],
                [2.0, 2.0, 0.10],
                "ADCS_MagnetometerDie",
                True,
            ),
            *_packaged_ic(
                "ADCS_ActuatorDriver",
                [0.0, 12.0, -3.7],
                [10.0, 8.0, 1.4],
                [6.0, 4.0, 0.30],
                [5.0, 3.0, 0.05],
            ),
        ],
        "comms": [
            _device_component(
                "Comms_Package", "G4_BAKELITE", [0.0, 0.0, 0.0], [10.5, 9.2, 2.4]
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
            _device_component(
                "Comms_RFDie",
                "G4_Si",
                [-2.0, 0.0, -0.65],
                [3.0, 3.2, 0.35],
                "Comms_Package",
            ),
            _device_component(
                "Comms_RFActive",
                "G4_Si",
                [-2.0, 0.0, -0.505],
                [2.3, 2.4, 0.05],
                "Comms_RFDie",
                True,
            ),
            _device_component(
                "Comms_BasebandDie",
                "G4_Si",
                [1.8, -1.8, -0.65],
                [3.0, 2.4, 0.35],
                "Comms_Package",
            ),
            _device_component(
                "Comms_BasebandActive",
                "G4_Si",
                [1.8, -1.8, -0.505],
                [2.3, 1.8, 0.05],
                "Comms_BasebandDie",
                True,
            ),
            _device_component(
                "Comms_ControlDie",
                "G4_Si",
                [1.8, 1.8, -0.65],
                [2.6, 2.2, 0.35],
                "Comms_Package",
            ),
            _device_component(
                "Comms_ControlActive",
                "G4_Si",
                [1.8, 1.8, -0.505],
                [2.0, 1.6, 0.05],
                "Comms_ControlDie",
                True,
            ),
        ],
    }
