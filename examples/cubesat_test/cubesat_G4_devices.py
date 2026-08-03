def _device_component(name, material, center_mm, size_mm, parent="", score=False):
    return {
        "name": name,
        "material": material,
        "parent": parent,
        "center_mm": center_mm,
        "size_mm": size_mm,
        "score": score,
    }


def build_device_components():
    # G4_BAKELITE is a built-in Geant4 resin proxy for package epoxy. It is more
    # realistic than polyethylene for molded packages, but it is not a filled
    # commercial mold compound.
    return {
        "obc": [
            _device_component(
                "OBC_Package", "G4_BAKELITE", [0.0, 0.0, 0.0], [34.0, 25.0, 5.4]
            ),
            _device_component(
                "OBC_Solder",
                "G4_Sn",
                [0.0, 0.0, -2.55],
                [32.0, 23.0, 0.10],
                "OBC_Package",
            ),
            _device_component(
                "OBC_Interposer",
                "G4_Cu",
                [0.0, 0.0, -2.15],
                [32.0, 23.0, 0.25],
                "OBC_Package",
            ),
            _device_component(
                "OBC_MainDie",
                "G4_Si",
                [-6.0, 0.0, -1.75],
                [12.0, 12.0, 0.50],
                "OBC_Package",
            ),
            _device_component(
                "OBC_MainActive",
                "G4_Si",
                [-6.0, 0.0, -1.525],
                [10.0, 10.0, 0.05],
                "OBC_MainDie",
                True,
            ),
            _device_component(
                "OBC_MemoryDie",
                "G4_Si",
                [8.0, 0.0, -1.70],
                [8.0, 5.0, 0.40],
                "OBC_Package",
            ),
            _device_component(
                "OBC_MemoryActive",
                "G4_Si",
                [8.0, 0.0, -1.525],
                [6.0, 3.0, 0.05],
                "OBC_MemoryDie",
                True,
            ),
        ],
        "eps": [
            # EPS electronics package. This represents a power-management or
            # controller IC inside the coarse MCDC EPS sensitive-volume region;
            # it is structural/package material and is not scored directly.
            _device_component(
                "EPS_PMICPackage",
                "G4_BAKELITE",
                [0.0, 0.0, 0.0],
                [12.0, 12.0, 2.0],
            ),
            # EPS silicon die. This is the device substrate inside the package;
            # the active layer below is the SEE-sensitive scoring volume.
            _device_component(
                "EPS_PMICDie",
                "G4_Si",
                [0.0, 0.0, -0.45],
                [8.0, 8.0, 0.50],
                "EPS_PMICPackage",
            ),
            # EPS sensitive volume. This is the only scored EPS component and
            # should be compared with the OBC/ADCS/Comms active silicon layers,
            # not with bulk battery electrode or separator dose.
            _device_component(
                "EPS_PMICActive",
                "G4_Si",
                [0.0, 0.0, -0.235],
                [6.0, 6.0, 0.05],
                "EPS_PMICDie",
                score=True,
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
                [-10.0, 0.0, -4.125],
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
                [4.0, 0.0, -3.575],
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
                [13.0, 0.0, -4.15],
                [2.0, 2.0, 0.10],
                "ADCS_MagnetometerDie",
                True,
            ),
        ],
        "comms": [
            _device_component(
                "Comms_Package", "G4_BAKELITE", [0.0, 0.0, 0.0], [10.5, 9.2, 2.4]
            ),
            _device_component(
                "Comms_Solder",
                "G4_Sn",
                [0.0, 0.0, -1.15],
                [8.0, 7.0, 0.1],
                "Comms_Package",
            ),
            _device_component(
                "Comms_ExposedPad",
                "G4_Cu",
                [0.0, 0.0, -1.0],
                [8.0, 7.0, 0.2],
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
                [0.0, 0.0, -0.65],
                [5.0, 4.0, 0.4],
                "Comms_Package",
            ),
            _device_component(
                "Comms_RFActive",
                "G4_Si",
                [0.0, 0.0, -0.475],
                [4.0, 3.0, 0.05],
                "Comms_RFDie",
                True,
            ),
        ],
    }
