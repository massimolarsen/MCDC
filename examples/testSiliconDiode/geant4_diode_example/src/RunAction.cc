#include "RunAction.hh"

#include "G4Run.hh"
#include "G4SystemOfUnits.hh"

#include <iomanip>

void RunAction::BeginOfRunAction(const G4Run*)
{
  fTotalSensitiveEdep = 0.0;
}

void RunAction::EndOfRunAction(const G4Run* run)
{
  const auto nEvents = run->GetNumberOfEvent();
  if (nEvents == 0) {
    return;
  }

  const auto meanEdep = fTotalSensitiveEdep / static_cast<double>(nEvents);
  G4cout << "\n[diode] events: " << nEvents << G4endl;
  G4cout << "[diode] total sensitive edep: " << std::setprecision(8)
         << fTotalSensitiveEdep / MeV << " MeV" << G4endl;
  G4cout << "[diode] mean sensitive edep / event: " << meanEdep / MeV << " MeV"
         << G4endl;
}

void RunAction::AddSensitiveEdep(double edep)
{
  fTotalSensitiveEdep += edep;
}
