#ifndef MCDC_DIODE_PRIMARY_GENERATOR_ACTION_HH
#define MCDC_DIODE_PRIMARY_GENERATOR_ACTION_HH

#include "G4VUserPrimaryGeneratorAction.hh"

class G4Event;
class G4GeneralParticleSource;

class PrimaryGeneratorAction : public G4VUserPrimaryGeneratorAction
{
  public:
    PrimaryGeneratorAction();
    ~PrimaryGeneratorAction() override;

    void GeneratePrimaries(G4Event* event) override;

  private:
    G4GeneralParticleSource* fGps = nullptr;
};

#endif
