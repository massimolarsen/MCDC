#ifndef MCDC_DIODE_STEPPING_ACTION_HH
#define MCDC_DIODE_STEPPING_ACTION_HH

#include "G4UserSteppingAction.hh"

class DetectorConstruction;
class EventAction;

class SteppingAction : public G4UserSteppingAction
{
  public:
    SteppingAction(const DetectorConstruction* detector, EventAction* eventAction);
    ~SteppingAction() override = default;

    void UserSteppingAction(const G4Step* step) override;

  private:
    const DetectorConstruction* fDetector = nullptr;
    EventAction* fEventAction = nullptr;
};

#endif
