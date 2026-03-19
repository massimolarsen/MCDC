#include "EventAction.hh"

#include "RunAction.hh"

#include "G4Event.hh"

EventAction::EventAction(RunAction* runAction)
  : fRunAction(runAction)
{
}

void EventAction::BeginOfEventAction(const G4Event*)
{
  fSensitiveEdep = 0.0;
}

void EventAction::EndOfEventAction(const G4Event*)
{
  fRunAction->AddSensitiveEdep(fSensitiveEdep);
}

void EventAction::AddSensitiveEdep(double edep)
{
  fSensitiveEdep += edep;
}
