#ifndef MCDC_DIODE_RUN_ACTION_HH
#define MCDC_DIODE_RUN_ACTION_HH

#include "G4UserRunAction.hh"

class G4Run;

class RunAction : public G4UserRunAction
{
  public:
    RunAction() = default;
    ~RunAction() override = default;

    void BeginOfRunAction(const G4Run* run) override;
    void EndOfRunAction(const G4Run* run) override;

    void AddSensitiveEdep(double edep);

  private:
    double fTotalSensitiveEdep = 0.0;
};

#endif
