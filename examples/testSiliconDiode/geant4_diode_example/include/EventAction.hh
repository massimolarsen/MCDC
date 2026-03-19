#ifndef MCDC_DIODE_EVENT_ACTION_HH
#define MCDC_DIODE_EVENT_ACTION_HH

#include "G4UserEventAction.hh"

class RunAction;

class EventAction : public G4UserEventAction
{
  public:
    explicit EventAction(RunAction* runAction);
    ~EventAction() override = default;

    void BeginOfEventAction(const G4Event*) override;
    void EndOfEventAction(const G4Event*) override;

    void AddSensitiveEdep(double edep);

  private:
    RunAction* fRunAction = nullptr;
    double fSensitiveEdep = 0.0;
};

#endif
