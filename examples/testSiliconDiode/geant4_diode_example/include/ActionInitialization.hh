#ifndef MCDC_DIODE_ACTION_INITIALIZATION_HH
#define MCDC_DIODE_ACTION_INITIALIZATION_HH

#include "G4VUserActionInitialization.hh"

class DetectorConstruction;

class ActionInitialization : public G4VUserActionInitialization
{
  public:
    explicit ActionInitialization(DetectorConstruction* detector);
    ~ActionInitialization() override = default;

    void Build() const override;

  private:
    DetectorConstruction* fDetector = nullptr;
};

#endif
