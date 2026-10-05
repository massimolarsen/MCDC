// Tabulate Geant4 proton stopping powers for elements, in the two-column
// format the MC/DC proton generator reads from $MCDC_PSTAR_LIB:
//   <symbol>.txt : kinetic energy [MeV], total stopping power [MeV cm2/g]
//
// Usage: geant4_dedx OUTPUT_DIR SYMBOL [SYMBOL ...]
//
// Uses the physics list the MC/DC-Geant4 bridge runs (QGSP_BIC), so MC/DC
// slows protons with the same dE/dx Geant4 applies after the handoff.

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "G4Box.hh"
#include "G4EmCalculator.hh"
#include "G4LogicalVolume.hh"
#include "G4NistManager.hh"
#include "G4PVPlacement.hh"
#include "G4PhysListFactory.hh"
#include "G4Proton.hh"
#include "G4RunManager.hh"
#include "G4SystemOfUnits.hh"
#include "G4VUserDetectorConstruction.hh"
#include "G4Version.hh"

namespace
{

// one small box per element so every material has a production couple
class ElementBoxes : public G4VUserDetectorConstruction
{
  public:
    explicit ElementBoxes(std::vector<G4Material*> materials)
        : materials_(std::move(materials))
    {
    }

    G4VPhysicalVolume* Construct() override
    {
      auto* nist = G4NistManager::Instance();
      auto* world_solid = new G4Box("world", 1.0 * m, 1.0 * m, 1.0 * m);
      auto* world_logical =
        new G4LogicalVolume(world_solid, nist->FindOrBuildMaterial("G4_Galactic"), "world");
      auto* world = new G4PVPlacement(nullptr, {}, world_logical, "world", nullptr, false, 0);

      auto* box_solid = new G4Box("box", 1.0 * cm, 1.0 * cm, 1.0 * cm);
      for (std::size_t i = 0; i < materials_.size(); ++i) {
        auto* logical = new G4LogicalVolume(box_solid, materials_[i], materials_[i]->GetName());
        const G4ThreeVector position(-0.9 * m + 3.0 * cm * static_cast<double>(i), 0.0, 0.0);
        new G4PVPlacement(nullptr, position, logical, materials_[i]->GetName(), world_logical,
                          false, 0);
      }
      return world;
    }

  private:
    std::vector<G4Material*> materials_;
};

}  // namespace

int main(int argc, char** argv)
{
  if (argc < 3) {
    std::cerr << "Usage: geant4_dedx OUTPUT_DIR SYMBOL [SYMBOL ...]\n";
    return 1;
  }
  const std::string output_dir = argv[1];
  const std::string physics_list = "QGSP_BIC";

  // elemental NIST materials, e.g. G4_Al
  auto* nist = G4NistManager::Instance();
  std::vector<std::string> symbols;
  std::vector<G4Material*> materials;
  for (int i = 2; i < argc; ++i) {
    symbols.emplace_back(argv[i]);
    G4Material* material = nist->FindOrBuildMaterial("G4_" + symbols.back());
    if (material == nullptr) {
      std::cerr << "No Geant4 NIST material G4_" << symbols.back() << '\n';
      return 1;
    }
    materials.push_back(material);
  }

  auto run_manager = std::make_unique<G4RunManager>();
  run_manager->SetUserInitialization(new ElementBoxes(materials));
  G4PhysListFactory factory;
  run_manager->SetUserInitialization(factory.GetReferencePhysList(physics_list));
  run_manager->Initialize();

  // EM models and tables are built at the start of the first run
  run_manager->BeamOn(0);

  // 1 keV to 10 GeV, 20 points per decade
  std::vector<double> energies_mev;
  for (int i = 0; i <= 140; ++i) {
    energies_mev.push_back(1.0e-3 * std::pow(10.0, i / 20.0));
  }

  G4EmCalculator calculator;
  const G4ParticleDefinition* proton = G4Proton::Definition();
  for (std::size_t m_idx = 0; m_idx < materials.size(); ++m_idx) {
    const G4Material* material = materials[m_idx];
    const double density_g_cm3 = material->GetDensity() / (g / cm3);
    const std::string path = output_dir + "/" + symbols[m_idx] + ".txt";
    std::ofstream out(path);
    out << "# Geant4 " << G4Version << " " << physics_list << " proton total dE/dx in "
        << material->GetName() << ": T [MeV], S/rho [MeV cm2/g]\n";
    out << std::setprecision(6) << std::scientific;
    for (double energy_mev : energies_mev) {
      const double dedx = calculator.ComputeTotalDEDX(energy_mev * MeV, proton, material);
      out << energy_mev << ' ' << (dedx / (MeV / cm)) / density_g_cm3 << '\n';
    }
    std::cout << "wrote " << path << '\n';
  }
  return 0;
}
