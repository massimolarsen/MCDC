#ifndef MCDC_DIODE_DETECTOR_CONSTRUCTION_HH
#define MCDC_DIODE_DETECTOR_CONSTRUCTION_HH

#include "G4VUserDetectorConstruction.hh"

class G4LogicalVolume;
class G4VPhysicalVolume;

class DetectorConstruction : public G4VUserDetectorConstruction
{
  public:
    DetectorConstruction() = default;
    ~DetectorConstruction() override = default;

    G4VPhysicalVolume* Construct() override;

    G4LogicalVolume* GetSensitiveVolume() const { return fSensitiveVolume; }

  private:
    G4LogicalVolume* fSensitiveVolume = nullptr;
};

#endif
