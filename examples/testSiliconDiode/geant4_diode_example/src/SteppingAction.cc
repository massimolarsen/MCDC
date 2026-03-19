#include "SteppingAction.hh"

#include "DetectorConstruction.hh"
#include "EventAction.hh"

#include "G4LogicalVolume.hh"
#include "G4Step.hh"

SteppingAction::SteppingAction(const DetectorConstruction* detector, EventAction* eventAction)
  : fDetector(detector)
  , fEventAction(eventAction)
{
}

void SteppingAction::UserSteppingAction(const G4Step* step)
{
  auto* preVolume = step->GetPreStepPoint()->GetTouchableHandle()->GetVolume();
  if (preVolume == nullptr) {
    return;
  }

  auto* logical = preVolume->GetLogicalVolume();
  if (logical != fDetector->GetSensitiveVolume()) {
    return;
  }

  const auto edep = step->GetTotalEnergyDeposit();
  if (edep > 0.0) {
    fEventAction->AddSensitiveEdep(edep);
  }
}
