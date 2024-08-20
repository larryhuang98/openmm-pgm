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

void ReferenceCalc_pGM_MultipoleForceKernel::initialize(const System& system, const _pGM_MultipoleForce& force) {

    numMultipoles   = force.getNumMultipoles();

    charges.resize(numMultipoles);
    dipoles.resize(3*numMultipoles);
    polarity.resize(numMultipoles);
    multipoleAtomCovalentInfo.resize(numMultipoles);

    int dipoleIndex      = 0;
    int quadrupoleIndex  = 0;
    double totalCharge   = 0.0;
    for (int ii = 0; ii < numMultipoles; ii++) {

        // multipoles

        int axisType, multipoleAtomZ, multipoleAtomX, multipoleAtomY;
        double charge, tholeD, dampingFactorD, polarityD;
        std::vector<double> dipolesD;
        std::vector<double> quadrupolesD;
        force.getMultipoleParameters(ii, charge, dipolesD, quadrupolesD, axisType, multipoleAtomZ, multipoleAtomX, multipoleAtomY,
                                     tholeD, dampingFactorD, polarityD);

        totalCharge                       += charge;
        axisTypes[ii]                      = axisType;
        multipoleAtomZs[ii]                = multipoleAtomZ;
        multipoleAtomXs[ii]                = multipoleAtomX;
        multipoleAtomYs[ii]                = multipoleAtomY;

        charges[ii]                        = charge;
        tholes[ii]                         = tholeD;
        dampingFactors[ii]                 = dampingFactorD;
        polarity[ii]                       = polarityD;

        dipoles[dipoleIndex++]             = dipolesD[0];
        dipoles[dipoleIndex++]             = dipolesD[1];
        dipoles[dipoleIndex++]             = dipolesD[2];
        

        // covalent info

        std::vector< std::vector<int> > covalentLists;
        force.getCovalentMaps(ii, covalentLists);
        multipoleAtomCovalentInfo[ii] = covalentLists;

    }

    polarizationType = force.getPolarizationType();
    if (polarizationType == pGM_MultipoleForce::Mutual) {
        mutualInducedMaxIterations = force.getMutualInducedMaxIterations();
        mutualInducedTargetEpsilon = force.getMutualInducedTargetEpsilon();
    } else  {
        throw OPENMMException();
    }

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

    // amoebaReferenceMultipoleForce is set to _pGM_ReferenceGeneralizedKirkwoodForce if _pGM_GeneralizedKirkwoodForce is present
    // amoebaReferenceMultipoleForce is set to _pGM_ReferencePmeMultipoleForce if 'usePme' is set
    // amoebaReferenceMultipoleForce is set to _pGM_ReferenceMultipoleForce otherwise


    pGM_ReferenceMultipoleForce* amoebaReferenceMultipoleForce = NULL;
    if (usePme) {

        pGM_ReferencePmeMultipoleForce* amoebaReferencePmeMultipoleForce = new pGM_ReferencePmeMultipoleForce();
        amoebaReferencePmeMultipoleForce->setAlphaEwald(alphaEwald);
        amoebaReferencePmeMultipoleForce->setCutoffDistance(cutoffDistance);
        amoebaReferencePmeMultipoleForce->setPmeGridDimensions(pmeGridDimension);
        Vec3* boxVectors = extractBoxVectors(context);
        double minAllowedSize = 1.999999*cutoffDistance;
        if (boxVectors[0][0] < minAllowedSize || boxVectors[1][1] < minAllowedSize || boxVectors[2][2] < minAllowedSize) {
            throw OpenMMException("The periodic box size has decreased to less than twice the nonbonded cutoff.");
        }
        amoebaReferencePmeMultipoleForce->setPeriodicBoxSize(boxVectors);
        amoebaReferenceMultipoleForce = static_cast<pGM_ReferenceMultipoleForce*>(amoebaReferencePmeMultipoleForce);

    } else {
         throw OpenMMException("Currently, pGM force field only supports PME method, in the future, other methods will be implemented.");
    }

    // set polarization type

    if (polarizationType == pGM_MultipoleForce::Mutual) {
        amoebaReferenceMultipoleForce->setPolarizationType(pGM_ReferenceMultipoleForce::Mutual);
        amoebaReferenceMultipoleForce->setMutualInducedDipoleTargetEpsilon(mutualInducedTargetEpsilon);
        amoebaReferenceMultipoleForce->setMaximumMutualInducedDipoleIterations(mutualInducedMaxIterations);
    } else {
        throw OpenMMException("Currently, pGM force field only supports mutual induced dipole.");
    }

    return amoebaReferenceMultipoleForce;

}

double ReferenceCalc_pGM_MultipoleForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {

    pGM_ReferenceMultipoleForce* amoebaReferenceMultipoleForce = setup_pGM_ReferenceMultipoleForce(context);

    vector<Vec3>& posData = extractPositions(context);
    vector<Vec3>& forceData = extractForces(context);
    double energy = amoebaReferenceMultipoleForce->calculateForceAndEnergy(posData, charges, dipoles, quadrupoles, tholes,
                                                                           dampingFactors, polarity, axisTypes, 
                                                                           multipoleAtomZs, multipoleAtomXs, multipoleAtomYs,
                                                                           multipoleAtomCovalentInfo, forceData);

    delete amoebaReferenceMultipoleForce;

    return static_cast<double>(energy);
}

void ReferenceCalc_pGM_MultipoleForceKernel::getInducedDipoles(ContextImpl& context, vector<Vec3>& outputDipoles) {
    int numParticles = context.getSystem().getNumParticles();
    outputDipoles.resize(numParticles);

    // Create an _pGM_ReferenceMultipoleForce to do the calculation.
    
    pGM_ReferenceMultipoleForce* amoebaReferenceMultipoleForce = setup_pGM_ReferenceMultipoleForce(context);
    vector<Vec3>& posData = extractPositions(context);
    
    // Retrieve the induced dipoles.
    
    vector<Vec3> inducedDipoles;
    amoebaReferenceMultipoleForce->calculateInducedDipoles(posData, charges, dipoles, quadrupoles, tholes,
            dampingFactors, polarity, axisTypes, multipoleAtomZs, multipoleAtomXs, multipoleAtomYs, multipoleAtomCovalentInfo, inducedDipoles);
    for (int i = 0; i < numParticles; i++)
        outputDipoles[i] = inducedDipoles[i];
    delete amoebaReferenceMultipoleForce;
}

void ReferenceCalc_pGM_MultipoleForceKernel::getLabFramePermanentDipoles(ContextImpl& context, vector<Vec3>& outputDipoles) {
    int numParticles = context.getSystem().getNumParticles();
    outputDipoles.resize(numParticles);

    // Create an _pGM_ReferenceMultipoleForce to do the calculation.
    
    pGM_ReferenceMultipoleForce* amoebaReferenceMultipoleForce = setup_pGM_ReferenceMultipoleForce(context);
    vector<Vec3>& posData = extractPositions(context);
    
    // Retrieve the permanent dipoles in the lab frame.
    
    vector<Vec3> labFramePermanentDipoles;
    amoebaReferenceMultipoleForce->calculateLabFramePermanentDipoles(posData, charges, dipoles, quadrupoles, tholes, 
            dampingFactors, polarity, axisTypes, multipoleAtomZs, multipoleAtomXs, multipoleAtomYs, multipoleAtomCovalentInfo, labFramePermanentDipoles);
    for (int i = 0; i < numParticles; i++)
        outputDipoles[i] = labFramePermanentDipoles[i];
    delete amoebaReferenceMultipoleForce;
}


void ReferenceCalc_pGM_MultipoleForceKernel::getTotalDipoles(ContextImpl& context, vector<Vec3>& outputDipoles) {
    int numParticles = context.getSystem().getNumParticles();
    outputDipoles.resize(numParticles);

    // Create an _pGM_ReferenceMultipoleForce to do the calculation.
    
    pGM_ReferenceMultipoleForce* amoebaReferenceMultipoleForce = setup_pGM_ReferenceMultipoleForce(context);
    vector<Vec3>& posData = extractPositions(context);
    
    // Retrieve the permanent dipoles in the lab frame.
    
    vector<Vec3> totalDipoles;
    amoebaReferenceMultipoleForce->calculateTotalDipoles(posData, charges, dipoles, quadrupoles, tholes,
            dampingFactors, polarity, axisTypes, multipoleAtomZs, multipoleAtomXs, multipoleAtomYs, multipoleAtomCovalentInfo, totalDipoles);

    for (int i = 0; i < numParticles; i++)
        outputDipoles[i] = totalDipoles[i];
    delete amoebaReferenceMultipoleForce;
}



void ReferenceCalc_pGM_MultipoleForceKernel::getElectrostaticPotential(ContextImpl& context, const std::vector< Vec3 >& inputGrid,
                                                                        std::vector< double >& outputElectrostaticPotential) {

    pGM_ReferenceMultipoleForce* amoebaReferenceMultipoleForce = setup_pGM_ReferenceMultipoleForce(context);
    vector<Vec3>& posData                                     = extractPositions(context);
    vector<Vec3> grid(inputGrid.size());
    vector<double> potential(inputGrid.size());
    for (unsigned int ii = 0; ii < inputGrid.size(); ii++) {
        grid[ii] = inputGrid[ii];
    }
    amoebaReferenceMultipoleForce->calculateElectrostaticPotential(posData, charges, dipoles, quadrupoles, tholes,
                                                                   dampingFactors, polarity, axisTypes, 
                                                                   multipoleAtomZs, multipoleAtomXs, multipoleAtomYs,
                                                                   multipoleAtomCovalentInfo, grid, potential);

    outputElectrostaticPotential.resize(inputGrid.size());
    for (unsigned int ii = 0; ii < inputGrid.size(); ii++) {
        outputElectrostaticPotential[ii] = potential[ii];
    }

    delete amoebaReferenceMultipoleForce;
}

void ReferenceCalc_pGM_MultipoleForceKernel::getSystemMultipoleMoments(ContextImpl& context, std::vector< double >& outputMultipoleMoments) {

    // retrieve masses

    const System& system             = context.getSystem();
    vector<double> masses;
    for (int i = 0; i <  system.getNumParticles(); ++i) {
        masses.push_back(system.getParticleMass(i));
    }    

    pGM_ReferenceMultipoleForce* amoebaReferenceMultipoleForce = setup_pGM_ReferenceMultipoleForce(context);
    vector<Vec3>& posData                                     = extractPositions(context);
    amoebaReferenceMultipoleForce->calculate_pGM_SystemMultipoleMoments(masses, posData, charges, dipoles, quadrupoles, tholes,
                                                                         dampingFactors, polarity, axisTypes, 
                                                                         multipoleAtomZs, multipoleAtomXs, multipoleAtomYs,
                                                                         multipoleAtomCovalentInfo, outputMultipoleMoments);

    delete amoebaReferenceMultipoleForce;
}

void ReferenceCalc_pGM_MultipoleForceKernel::copyParametersToContext(ContextImpl& context, const _pGM_MultipoleForce& force) {
    if (numMultipoles != force.getNumMultipoles())
        throw OpenMMException("updateParametersInContext: The number of multipoles has changed");

    // Record the values.

    int dipoleIndex = 0;
    int quadrupoleIndex = 0;
    for (int i = 0; i < numMultipoles; ++i) {
        int axisType, multipoleAtomZ, multipoleAtomX, multipoleAtomY;
        double charge, tholeD, dampingFactorD, polarityD;
        std::vector<double> dipolesD;
        std::vector<double> quadrupolesD;
        force.getMultipoleParameters(i, charge, dipolesD, quadrupolesD, axisType, multipoleAtomZ, multipoleAtomX, multipoleAtomY, tholeD, dampingFactorD, polarityD);
        axisTypes[i] = axisType;
        multipoleAtomZs[i] = multipoleAtomZ;
        multipoleAtomXs[i] = multipoleAtomX;
        multipoleAtomYs[i] = multipoleAtomY;
        charges[i] = charge;
        tholes[i] = tholeD;
        dampingFactors[i] = dampingFactorD;
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
