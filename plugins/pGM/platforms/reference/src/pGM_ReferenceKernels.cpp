/* -------------------------------------------------------------------------- *
 *                               OpenMM_pGM                                   *
 * -------------------------------------------------------------------------- *
 This part is the main driver of pGM force field
 * -------------------------------------------------------------------------- */

#include "pGM_ReferenceKernels.h"
//#include "pGM_ReferenceTorsionTorsionForce.h"
//#include "pGM_ReferenceWcaDispersionForce.h"
//#include "pGM_ReferenceGeneralizedKirkwoodForce.h"
//#include "openmm/internal/pGM_TorsionTorsionForceImpl.h"
//#include "openmm/internal/pGM_WcaDispersionForceImpl.h"
#include "ReferencePlatform.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/pGM_MultipoleForce.h"
//#include "openmm/HippoNonbondedForce.h"
#include "openmm/internal/pGM_MultipoleForceImpl.h"
//#include "openmm/internal/pGM_VdwForceImpl.h"
//#include "openmm/internal/pGM_GeneralizedKirkwoodForceImpl.h"
#include "openmm/NonbondedForce.h"
#include "openmm/internal/NonbondedForceImpl.h"
//#include "SimTKReference/pGM_ReferenceHippoNonbondedForce.h"

#include <cmath>
#ifdef _MSC_VER
#include <windows.h>
#endif

using namespace OpenMM;
using namespace std;

static vector<Vec3>& extractPositions(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *data->positions;
}

static vector<Vec3>& extractVelocities(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *data->velocities;
}

static vector<Vec3>& extractForces(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *data->forces;
}

static Vec3& extractBoxSize(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *data->periodicBoxSize;
}

static Vec3* extractBoxVectors(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return data->periodicBoxVectors;
}


/* -------------------------------------------------------------------------- *
 *                             pGM_Multipole                                *
 * -------------------------------------------------------------------------- */

ReferenceCalc_pGM_MultipoleForceKernel::ReferenceCalc_pGM_MultipoleForceKernel(const std::string& name, const Platform& platform, const System& system) :
         Calc_pGM_MultipoleForceKernel(name, platform), system(system), numMultipoles(0), mutualInducedMaxIterations(60), mutualInducedTargetEpsilon(1.0e-03),
                                                         usePme(true),alphaEwald(0.0), cutoffDistance(1.0) {  

}

ReferenceCalc_pGM_MultipoleForceKernel::~ReferenceCalc_pGM_MultipoleForceKernel() {
}

void ReferenceCalc_pGM_MultipoleForceKernel::initialize(const System& system, const pGM_MultipoleForce& force) {

    numMultipoles   = force.getNumMultipoles();

    charges.resize(numMultipoles);
    dipoles.resize(3*numMultipoles);
    polarity.resize(numMultipoles);
    multipoleAtomCovalentInfo.resize(numMultipoles);

    double totalCharge   = 0.0;
    int dipoleIndex = 0;
    for (int ii = 0; ii < numMultipoles; ii++) {

        // multipoles
        double charge, betaD , polarityD;
        std::vector<double> dipolesD;
        std::vector<int> covalentAtoms;
        force.getMultipoleParameters(ii, charge, dipolesD, betaD, polarityD);

        totalCharge                       += charge;
        

        charges[ii]                        = charge;
        polarity[ii]                       = polarityD;

        int n = sizeof(dipolesD);
        for (int jj=0; jj < n; jj++){
            dipoles[dipoleIndex++] = dipolesD[jj];
        }
        // covalent info

        std::vector< std::vector<int> > covalentLists;
        force.getCovalentMaps(ii, covalentLists);
        multipoleAtomCovalentInfo[ii] = covalentLists;

    }


    mutualInducedMaxIterations = force.getMutualInducedMaxIterations();
    mutualInducedTargetEpsilon = force.getMutualInducedTargetEpsilon();


    // PME

    nonbondedMethod  = force.getNonbondedMethod();
    if (nonbondedMethod == pGM_MultipoleForce::PME) {
        usePme     = true;
        pmeGridDimension.resize(3);
        force.getPMEParameters(alphaEwald, pmeGridDimension[0], pmeGridDimension[1], pmeGridDimension[2]);
        cutoffDistance = force.getCutoffDistance();
        if (pmeGridDimension[0] == 0 || alphaEwald == 0.0) {
            NonbondedForce nb;
            nb.setEwaldErrorTolerance(force.getEwaldErrorTolerance());
            nb.setCutoffDistance(force.getCutoffDistance());
            int gridSizeX, gridSizeY, gridSizeZ;
            NonbondedForceImpl::calcPMEParameters(system, nb, alphaEwald, gridSizeX, gridSizeY, gridSizeZ, false);
            pmeGridDimension[0] = gridSizeX;
            pmeGridDimension[1] = gridSizeY;
            pmeGridDimension[2] = gridSizeZ;
        }    
    } else {
        usePme = false;
    }
    return;
}

pGM_ReferenceMultipoleForce* ReferenceCalc_pGM_MultipoleForceKernel::setup_pGM_ReferenceMultipoleForce(ContextImpl& context)
{

    // pgm_ReferenceMultipoleForce is set to _pGM_ReferenceGeneralizedKirkwoodForce if _pGM_GeneralizedKirkwoodForce is present
    // pgm_ReferenceMultipoleForce is set to _pGM_ReferencePmeMultipoleForce if 'usePme' is set
    // pgm_ReferenceMultipoleForce is set to _pGM_ReferenceMultipoleForce otherwise


    pGM_ReferenceMultipoleForce* pgm_ReferenceMultipoleForce = NULL;
    /*
    if (usePme) {
        pGM_ReferencePmeMultipoleForce* pgm_ReferencePmeMultipoleForce = new pGM_ReferencePmeMultipoleForce();
        pgm_ReferencePmeMultipoleForce->setAlphaEwald(alphaEwald);
        pgm_ReferencePmeMultipoleForce->setCutoffDistance(cutoffDistance);
        pgm_ReferencePmeMultipoleForce->setPmeGridDimensions(pmeGridDimension);
        Vec3* boxVectors = extractBoxVectors(context);
        double minAllowedSize = 1.999999*cutoffDistance;
        if (boxVectors[0][0] < minAllowedSize || boxVectors[1][1] < minAllowedSize || boxVectors[2][2] < minAllowedSize) {
            throw OpenMMException("The periodic box size has decreased to less than twice the nonbonded cutoff.");
        }
        pgm_ReferencePmeMultipoleForce->setPeriodicBoxSize(boxVectors);
        pgm_ReferenceMultipoleForce = static_cast<pGM_ReferenceMultipoleForce*>(pgm_ReferencePmeMultipoleForce);

    } else if (useIPS) {
        pGM_ReferenceIPSMultipoleForce* pgm_ReferenceIPSMultipoleForce = new pGM_ReferenceIPSMultipoleForce();
        pgm_ReferenceIPSMultipoleForce->setCutoffDistance(cutoffDistance);
        pgm_ReferenceMultipoleForce = static_cast<pGM_ReferenceMultipoleForce*>(pgm_ReferenceIPSMultipoleForce);
    } else {
         throw OpenMMException("Currently, pGM force field only supports PME and IPS method, in the future, other methods will be implemented.");
    }
*/
    pgm_ReferenceMultipoleForce->setPolarizationType(pGM_ReferenceMultipoleForce::Mutual);//to be removed?
    pgm_ReferenceMultipoleForce->setMutualInducedDipoleTargetEpsilon(mutualInducedTargetEpsilon);
    pgm_ReferenceMultipoleForce->setMaximumMutualInducedDipoleIterations(mutualInducedMaxIterations);


    return pgm_ReferenceMultipoleForce;

}

double ReferenceCalc_pGM_MultipoleForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {

    pGM_ReferenceMultipoleForce* pGM_ReferenceMultipoleForce = setup_pGM_ReferenceMultipoleForce(context);

    vector<Vec3>& posData = extractPositions(context);
    vector<Vec3>& forceData = extractForces(context);
    double energy = pGM_ReferenceMultipoleForce->calculateForceAndEnergy(posData, charges, dipoles, beta, polarity, 
                                                                           multipoleAtomCovalentInfo, forceData);

    delete pGM_ReferenceMultipoleForce;

    return static_cast<double>(energy);
}

void ReferenceCalc_pGM_MultipoleForceKernel::getInducedDipoles(ContextImpl& context, vector<Vec3>& outputDipoles) {
    int numParticles = context.getSystem().getNumParticles();
    outputDipoles.resize(numParticles);

    // Create an _pGM_ReferenceMultipoleForce to do the calculation.
    
    pGM_ReferenceMultipoleForce* pgm_ReferenceMultipoleForce = setup_pGM_ReferenceMultipoleForce(context);
    vector<Vec3>& posData = extractPositions(context);
    
    // Retrieve the induced dipoles.
    
    vector<Vec3> inducedDipoles;
    pgm_ReferenceMultipoleForce->calculateInducedDipoles(posData, charges, dipoles, beta, polarity, multipoleAtomCovalentInfo, inducedDipoles);
    for (int i = 0; i < numParticles; i++)
        outputDipoles[i] = inducedDipoles[i];
    delete pgm_ReferenceMultipoleForce;
}

void ReferenceCalc_pGM_MultipoleForceKernel::getLabFramePermanentDipoles(ContextImpl& context, vector<Vec3>& outputDipoles) {
    int numParticles = context.getSystem().getNumParticles();
    outputDipoles.resize(numParticles);

    // Create an _pGM_ReferenceMultipoleForce to do the calculation.
    
    pGM_ReferenceMultipoleForce* pgm_ReferenceMultipoleForce = setup_pGM_ReferenceMultipoleForce(context);
    vector<Vec3>& posData = extractPositions(context);
    
    // Retrieve the permanent dipoles in the lab frame.
    
    vector<Vec3> labFramePermanentDipoles;
    pgm_ReferenceMultipoleForce->calculateLabFramePermanentDipoles(posData, charges, dipoles, beta, polarity, multipoleAtomCovalentInfo, labFramePermanentDipoles);
    for (int i = 0; i < numParticles; i++)
        outputDipoles[i] = labFramePermanentDipoles[i];
    delete pgm_ReferenceMultipoleForce;
}


void ReferenceCalc_pGM_MultipoleForceKernel::getTotalDipoles(ContextImpl& context, vector<Vec3>& outputDipoles) {
    int numParticles = context.getSystem().getNumParticles();
    outputDipoles.resize(numParticles);

    // Create an _pGM_ReferenceMultipoleForce to do the calculation.
    
    pGM_ReferenceMultipoleForce* pgm_ReferenceMultipoleForce = setup_pGM_ReferenceMultipoleForce(context);
    vector<Vec3>& posData = extractPositions(context);
    
    // Retrieve the permanent dipoles in the lab frame.
    
    vector<Vec3> totalDipoles;
    pgm_ReferenceMultipoleForce->calculateTotalDipoles(posData, charges, dipoles, beta, polarity, multipoleAtomCovalentInfo, totalDipoles);

    for (int i = 0; i < numParticles; i++)
        outputDipoles[i] = totalDipoles[i];
    delete pgm_ReferenceMultipoleForce;
}


/*
void ReferenceCalc_pGM_MultipoleForceKernel::getElectrostaticPotential(ContextImpl& context, const std::vector< Vec3 >& inputGrid,
                                                                        std::vector< double >& outputElectrostaticPotential) {

    pGM_ReferenceMultipoleForce* pgm_ReferenceMultipoleForce = setup_pGM_ReferenceMultipoleForce(context);
    vector<Vec3>& posData                                     = extractPositions(context);
    vector<Vec3> grid(inputGrid.size());
    vector<double> potential(inputGrid.size());
    for (unsigned int ii = 0; ii < inputGrid.size(); ii++) {
        grid[ii] = inputGrid[ii];
    }
    pgm_ReferenceMultipoleForce->calculateElectrostaticPotential(posData, charges, dipoles, beta, polarity,
                                                                   multipoleAtomCovalentInfo, grid, potential);

    outputElectrostaticPotential.resize(inputGrid.size());
    for (unsigned int ii = 0; ii < inputGrid.size(); ii++) {
        outputElectrostaticPotential[ii] = potential[ii];
    }

    delete pgm_ReferenceMultipoleForce;
}
*/
void ReferenceCalc_pGM_MultipoleForceKernel::getSystemMultipoleMoments(ContextImpl& context, std::vector< double >& outputMultipoleMoments) {

    // retrieve masses

    const System& system             = context.getSystem();
    vector<double> masses;
    for (int i = 0; i <  system.getNumParticles(); ++i) {
        masses.push_back(system.getParticleMass(i));
    }    

    pGM_ReferenceMultipoleForce* pgm_ReferenceMultipoleForce = setup_pGM_ReferenceMultipoleForce(context);
    vector<Vec3>& posData                                     = extractPositions(context);
    pgm_ReferenceMultipoleForce->calculateSystemMultipoleMoments(masses, posData, charges, dipoles, beta, polarity,
                                                                         multipoleAtomCovalentInfo, outputMultipoleMoments);

    delete pgm_ReferenceMultipoleForce;
}

void ReferenceCalc_pGM_MultipoleForceKernel::copyParametersToContext(ContextImpl& context, const pGM_MultipoleForce& force) {
    if (numMultipoles != force.getNumMultipoles())
        throw OpenMMException("updateParametersInContext: The number of multipoles has changed");

    // Record the values.

    int dipoleIndex = 0;
    int quadrupoleIndex = 0;
    for (int i = 0; i < numMultipoles; ++i) {
        int axisType, multipoleAtomZ, multipoleAtomX, multipoleAtomY;
        double charge, polarityD, betaD;
        std::vector<double> dipolesD;

        force.getMultipoleParameters(i, charge, dipolesD, betaD, polarityD);

        charges[i] = charge;

        polarity[i] = polarityD;
        dipoles[dipoleIndex++] = dipolesD[0];
        dipoles[dipoleIndex++] = dipolesD[1];
        dipoles[dipoleIndex++] = dipolesD[2];
    }
}

void ReferenceCalc_pGM_MultipoleForceKernel::getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    if (!usePme)
        throw OpenMMException("getPMEParametersInContext: This Context is not using PME");
    alpha = alphaEwald;
    nx = pmeGridDimension[0];
    ny = pmeGridDimension[1];
    nz = pmeGridDimension[2];
}
