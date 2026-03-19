#include "PrimaryGeneratorAction.hh"

#include "G4Event.hh"
#include "G4GeneralParticleSource.hh"
#include "G4ParticleTable.hh"
#include "G4SystemOfUnits.hh"

PrimaryGeneratorAction::PrimaryGeneratorAction()
{
  fGps = new G4GeneralParticleSource();
  auto* neutron = G4ParticleTable::GetParticleTable()->FindParticle("neutron");
  fGps->SetParticleDefinition(neutron);

  auto* posDist = fGps->GetCurrentSource()->GetPosDist();
  posDist->SetPosDisType("Plane");
  posDist->SetPosDisShape("Circle");
  posDist->SetRadius(4.0 * mm);
  posDist->SetCentreCoords(G4ThreeVector(0.0, 0.0, 3.5 * mm));

  auto* angDist = fGps->GetCurrentSource()->GetAngDist();
  angDist->SetParticleMomentumDirection(G4ThreeVector(0.0, 0.0, -1.0));

  auto* eneDist = fGps->GetCurrentSource()->GetEneDist();
  eneDist->SetMonoEnergy(1.0 * MeV);
}

PrimaryGeneratorAction::~PrimaryGeneratorAction()
{
  delete fGps;
}

void PrimaryGeneratorAction::GeneratePrimaries(G4Event* event)
{
  fGps->GeneratePrimaryVertex(event);
}
