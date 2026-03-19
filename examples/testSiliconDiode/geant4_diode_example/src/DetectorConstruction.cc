#include "DetectorConstruction.hh"

#include "G4Box.hh"
#include "G4Colour.hh"
#include "G4LogicalVolume.hh"
#include "G4Material.hh"
#include "G4NistManager.hh"
#include "G4PVPlacement.hh"
#include "G4SystemOfUnits.hh"
#include "G4Tubs.hh"
#include "G4VisAttributes.hh"

G4VPhysicalVolume* DetectorConstruction::Construct()
{
  auto* nist = G4NistManager::Instance();
  auto* vacuum = nist->FindOrBuildMaterial("G4_Galactic");
  auto* silicon = nist->FindOrBuildMaterial("G4_Si");
  auto* aluminum = nist->FindOrBuildMaterial("G4_Al");

  const G4double worldHalfLength = 6.0 * cm;
  const G4double diodeRadius = 0.4 * cm;
  const G4double diodeHalfLength = 0.3 * cm;
  const G4double beolHalfThickness = 0.25 * um;
  const G4double svHalfLength = diodeHalfLength - 1.0 * um;
  const G4double svRadius = diodeRadius - 1.0 * um;
  const G4double sourceRadius = 4.0 * mm;
  const G4double sourceHalfThickness = 1.0 * um;
  const G4double sourceZ = 3.5 * mm;

  auto* worldSolid =
    new G4Box("World", worldHalfLength, worldHalfLength, worldHalfLength);
  auto* worldLogical = new G4LogicalVolume(worldSolid, vacuum, "World");
  auto* worldPhysical = new G4PVPlacement(
    nullptr, G4ThreeVector(), worldLogical, "World", nullptr, false, 0, true);

  auto* diodeSolid =
    new G4Tubs("DiodeBulk", 0.0, diodeRadius, diodeHalfLength, 0.0, 360.0 * deg);
  auto* diodeLogical = new G4LogicalVolume(diodeSolid, silicon, "DiodeBulk");
  new G4PVPlacement(nullptr,
                    G4ThreeVector(),
                    diodeLogical,
                    "DiodeBulk",
                    worldLogical,
                    false,
                    0,
                    true);

  auto* svSolid = new G4Tubs("SensitiveVolume",
                             0.0,
                             svRadius,
                             svHalfLength,
                             0.0,
                             360.0 * deg);
  fSensitiveVolume = new G4LogicalVolume(svSolid, silicon, "SensitiveVolume");
  new G4PVPlacement(nullptr,
                    G4ThreeVector(),
                    fSensitiveVolume,
                    "SensitiveVolume",
                    diodeLogical,
                    false,
                    0,
                    true);

  auto* beolSolid = new G4Tubs("BEOL", 0.0, diodeRadius, beolHalfThickness, 0.0, 360.0 * deg);
  auto* beolLogical = new G4LogicalVolume(beolSolid, aluminum, "BEOL");
  new G4PVPlacement(nullptr,
                    G4ThreeVector(0.0, 0.0, diodeHalfLength + beolHalfThickness),
                    beolLogical,
                    "BEOL",
                    worldLogical,
                    false,
                    0,
                    true);

  auto* sourcePlaneSolid = new G4Tubs(
    "SourcePlane", 0.0, sourceRadius, sourceHalfThickness, 0.0, 360.0 * deg);
  auto* sourcePlaneLogical =
    new G4LogicalVolume(sourcePlaneSolid, vacuum, "SourcePlane");
  new G4PVPlacement(nullptr,
                    G4ThreeVector(0.0, 0.0, sourceZ),
                    sourcePlaneLogical,
                    "SourcePlane",
                    worldLogical,
                    false,
                    0,
                    true);

  worldLogical->SetVisAttributes(G4VisAttributes::GetInvisible());
  auto diodeVis = G4VisAttributes(G4Colour(0.7, 0.7, 0.7, 0.2));
  diodeVis.SetForceSolid(true);
  diodeLogical->SetVisAttributes(diodeVis);

  auto sensitiveVis = G4VisAttributes(G4Colour(0.1, 0.8, 0.2, 0.6));
  sensitiveVis.SetForceSolid(true);
  fSensitiveVolume->SetVisAttributes(sensitiveVis);

  auto beolVis = G4VisAttributes(G4Colour(0.8, 0.4, 0.1, 0.7));
  beolVis.SetForceSolid(true);
  beolLogical->SetVisAttributes(beolVis);

  auto sourcePlaneVis = G4VisAttributes(G4Colour(0.9, 0.1, 0.1, 0.9));
  sourcePlaneVis.SetForceWireframe(true);
  sourcePlaneVis.SetForceAuxEdgeVisible(true);
  sourcePlaneLogical->SetVisAttributes(sourcePlaneVis);

  return worldPhysical;
}
