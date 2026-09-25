// Electron slab benchmark (Geant4): pencil beam -> Al6061 slab -> Si detector.
// Same geometry as slab_mcdc.py. Usage: slab_e <E_MeV> <T_cm> <N> <nthreads> <seed>
// Electrons only (gammas killed, as MC/DC transports electrons only), option4 EM, 1 um cut.
#include <atomic>
#include <string>
#include <vector>
#include "G4EmCalculator.hh"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "G4Box.hh"
#include "G4Electron.hh"
#include "G4EmStandardPhysics_option4.hh"
#include "G4Gamma.hh"
#include "G4LogicalVolume.hh"
#include "G4Material.hh"
#include "G4NistManager.hh"
#include "G4PVPlacement.hh"
#include "G4ParticleGun.hh"
#include "G4RunManagerFactory.hh"
#include "G4Step.hh"
#include "G4SystemOfUnits.hh"
#include "G4Track.hh"
#include "G4UserStackingAction.hh"
#include "G4UserSteppingAction.hh"
#include "G4VModularPhysicsList.hh"
#include "G4VUserActionInitialization.hh"
#include "G4VUserDetectorConstruction.hh"
#include "G4VUserPrimaryGeneratorAction.hh"
#include "G4Event.hh"
#include "Randomize.hh"

static double gE = 1.0, gT = 0.3;
static const int NB = 12;
static const double kEdges[NB + 1] = {0, 1e-3, 1e-2, 5e-2, 0.1, 0.25, 0.5, 1, 2, 4, 6, 8, 12};  // MeV
static std::atomic<long> gDetAll{0}, gDetPrim{0}, gBack{0}, gTrans{0}, gDetBin[NB];

static int ebin(double e) { for (int i = 0; i < NB; ++i) if (e < kEdges[i + 1]) return i; return NB - 1; }

static G4Material* MakeMat(const char* name, std::vector<std::pair<const char*, double>> comp) {
  // comp: element symbol -> atom density [atoms/barn-cm]; converts to mass fractions + density
  auto* nist = G4NistManager::Instance();
  double rho = 0;  // g/cm3
  std::vector<std::pair<G4Element*, double>> m;
  for (auto& c : comp) {
    auto* el = nist->FindOrBuildElement(c.first);
    double mass = c.second * 1e24 * (el->GetA() / (g / mole)) / 6.02214076e23;  // g/cm3
    m.push_back({el, mass}); rho += mass;
  }
  auto* mat = new G4Material(name, rho * g / cm3, (int)m.size());
  for (auto& x : m) mat->AddElement(x.first, x.second / rho);
  return mat;
}

class Det : public G4VUserDetectorConstruction {
 public:
  G4VPhysicalVolume* Construct() override {
    auto* nist = G4NistManager::Instance();
    auto* vac = nist->FindOrBuildMaterial("G4_Galactic");
    // Materials built from the SAME atom densities (atoms/b-cm) used in slab_mcdc.py
    auto* al = MakeMat("Al6061", {{"Al", 0.059057360465045}, {"Mg", 0.00081349602416576},
                                  {"Si", 0.00046494828605733}});
    auto* si = MakeMat("Silicon", {{"Si", 0.050132618436459}});
    const double G = 0.7, DX = 1.8, H = 2.25, xmax = gT + G + DX + 1.0;
    auto* lw = new G4LogicalVolume(new G4Box("W", 0.5 * (xmax + 1.0) * cm, 11 * cm, 11 * cm), vac, "W");
    auto* pw = new G4PVPlacement(nullptr, {}, lw, "W", nullptr, false, 0);
    const double xc = 0.5 * (xmax - 1.0);  // world centre in deck frame
    auto* ls = new G4LogicalVolume(new G4Box("slab", 0.5 * gT * cm, 10 * cm, 10 * cm), al, "slab");
    new G4PVPlacement(nullptr, G4ThreeVector((0.5 * gT - xc) * cm, 0, 0), ls, "slab", lw, false, 0);
    auto* ld = new G4LogicalVolume(new G4Box("det", 0.5 * DX * cm, H * cm, H * cm), si, "det");
    new G4PVPlacement(nullptr, G4ThreeVector((gT + G + 0.5 * DX - xc) * cm, 0, 0), ld, "det", lw, false, 0);
    xc_ = xc;
    return pw;
  }
  static double xc_;
};
double Det::xc_ = 0;

class Gen : public G4VUserPrimaryGeneratorAction {
 public:
  void GeneratePrimaries(G4Event* ev) override {
    gun_.SetParticleDefinition(G4Electron::Definition());
    gun_.SetParticlePosition(G4ThreeVector((-0.5 - Det::xc_) * cm, 0, 0));
    gun_.SetParticleMomentumDirection(G4ThreeVector(1, 0, 0));
    gun_.SetParticleEnergy(gE * MeV);
    gun_.GeneratePrimaryVertex(ev);
  }
 private:
  G4ParticleGun gun_{1};
};

class Stack : public G4UserStackingAction {
 public:
  G4ClassificationOfNewTrack ClassifyNewTrack(const G4Track* t) override {
    return t->GetDefinition() == G4Gamma::Definition() ? fKill : fUrgent;
  }
};

class Step : public G4UserSteppingAction {
 public:
  void UserSteppingAction(const G4Step* s) override {
    auto* post = s->GetPostStepPoint();
    if (post->GetStepStatus() != fGeomBoundary) return;
    if (s->GetTrack()->GetDefinition() != G4Electron::Definition()) return;
    auto* pre = s->GetPreStepPoint()->GetPhysicalVolume();
    auto* nxt = post->GetPhysicalVolume();
    const auto& pn = pre->GetName();
    const std::string nn = nxt ? nxt->GetName() : "";
    if (pn == "slab" && nn != "slab") {  // leaving slab
      if (post->GetMomentumDirection().x() > 0) gTrans++; else gBack++;
    }
    if (nn == "det" && pn != "det") {
      gDetAll++;
      if (s->GetTrack()->GetTrackID() == 1) gDetPrim++;
      gDetBin[ebin(post->GetKineticEnergy() / MeV)]++;
    }
  }
};

class Act : public G4VUserActionInitialization {
 public:
  void Build() const override { SetUserAction(new Gen); SetUserAction(new Stack); SetUserAction(new Step); }
};

class Phys : public G4VModularPhysicsList {
 public:
  Phys() { RegisterPhysics(new G4EmStandardPhysics_option4()); SetDefaultCutValue(1 * um); }
};

int main(int argc, char** argv) {
  bool calc = argc >= 2 && std::string(argv[1]) == "calc";
  if (calc) { static char a1[] = "1", a2[] = "0.3", a3[] = "0", a4[] = "1", a5[] = "1";
              static char* av[] = {argv[0], a1, a2, a3, a4, a5}; argv = av; argc = 6; }
  if (argc < 6) { std::fprintf(stderr, "usage: slab_e E_MeV T_cm N nthreads seed | slab_e calc\n"); return 1; }
  gE = std::atof(argv[1]); gT = std::atof(argv[2]);
  long n = std::atol(argv[3]); int nthr = std::atoi(argv[4]);
  for (auto& b : gDetBin) b = 0;
  G4Random::setTheSeed(std::atol(argv[5]));
  auto* rm = G4RunManagerFactory::CreateRunManager(G4RunManagerType::Tasking);
  rm->SetNumberOfThreads(nthr);
  rm->SetUserInitialization(new Det);
  auto* ph = new Phys; ph->SetVerboseLevel(0);
  rm->SetUserInitialization(ph);
  rm->SetUserInitialization(new Act);
  rm->Initialize();
  if (calc) {
    rm->BeamOn(0);
    G4EmCalculator ec;
    for (const char* mn : {"Al6061", "Silicon"}) {
      auto* mat = G4Material::GetMaterial(mn);
      std::printf("MATERIAL %s density=%.6f g/cm3\n", mn, mat->GetDensity() / (g / cm3));
      for (size_t i = 0; i < mat->GetNumberOfElements(); ++i)
        std::printf("  EL %s N=%.9e atoms/b-cm\n", mat->GetElement(i)->GetSymbol().c_str(),
                    mat->GetVecNbOfAtomsPerVolume()[i] * cm3 * 1e-24);
      for (int k = 0; k <= 60; ++k) {
        double e = 1e-3 * std::pow(10.0, k * 4.0 / 60.0);  // 1 keV .. 10 MeV
        double scol = ec.ComputeElectronicDEDX(e * MeV, "e-", mn, 1e9 * MeV) / (MeV / cm);
        double srad = ec.ComputeDEDX(e * MeV, "e-", "eBrem", mn, 1e9 * MeV) / (MeV / cm);
        double xmsc = ec.ComputeCrossSectionPerVolume(e * MeV, "e-", "msc", mn) * cm;
        std::printf("CALC %s %.6e %.6e %.6e %.6e\n", mn, e, scol, srad, xmsc);
      }
    }
    delete rm;
    return 0;
  }
  rm->BeamOn(n);
  std::printf("RESULT E_MeV=%g T_cm=%g N=%ld det_all=%ld det_primary=%ld slab_back=%ld slab_trans=%ld\n",
              gE, gT, n, gDetAll.load(), gDetPrim.load(), gBack.load(), gTrans.load());
  std::printf("RESULT_BINS");
  for (int i = 0; i < NB; ++i) std::printf(" %ld", gDetBin[i].load());
  std::printf("\n");
  delete rm;
  return 0;
}
