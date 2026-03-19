#include "ActionInitialization.hh"

#include "DetectorConstruction.hh"
#include "EventAction.hh"
#include "PrimaryGeneratorAction.hh"
#include "RunAction.hh"
#include "SteppingAction.hh"

ActionInitialization::ActionInitialization(DetectorConstruction* detector)
  : fDetector(detector)
{
}

void ActionInitialization::Build() const
{
  SetUserAction(new PrimaryGeneratorAction());

  auto* runAction = new RunAction();
  SetUserAction(runAction);

  auto* eventAction = new EventAction(runAction);
  SetUserAction(eventAction);

  SetUserAction(new SteppingAction(fDetector, eventAction));
}
