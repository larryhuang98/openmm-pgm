/* -------------------------------------------------------------------------- *
 *                               OpenMMpGM                                 *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2008 Stanford University and the Authors.           *
 * Authors:                                                                   *
 * Contributors:                                                              *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

#include "openmm/internal/ContextImpl.h"
#include "openmm/internal/pGM_MultipoleForceImpl.h"
#include "openmm/internal/Messages.h"
#include "openmm/pGM_Kernels.h"
#include <stdio.h>

using namespace OpenMM;

using std::vector;

bool pGM_MultipoleForceImpl::initializedCovalentDegrees = false;
int pGM_MultipoleForceImpl::CovalentDegrees[]           = { 1,2,3,4,0,1,2,3};

pGM_MultipoleForceImpl::pGM_MultipoleForceImpl(const pGM_MultipoleForce& owner) : owner(owner) {
}

pGM_MultipoleForceImpl::~pGM_MultipoleForceImpl() {
}

void pGM_MultipoleForceImpl::initialize(ContextImpl& context) {

    const System& system = context.getSystem();
    int numParticles = system.getNumParticles();
    if (owner.getNumMultipoles() != numParticles)
        throw OpenMMException("pGM_MultipoleForce must have exactly as many particles as the System it belongs to.");

    // check cutoff < 0.5*boxSize

    if (owner.getNonbondedMethod() == pGM_MultipoleForce::PME) {
        Vec3 boxVectors[3];
        system.getDefaultPeriodicBoxVectors(boxVectors[0], boxVectors[1], boxVectors[2]);
        double cutoff = owner.getCutoffDistance();
        if (cutoff > 0.5*boxVectors[0][0] || cutoff > 0.5*boxVectors[1][1] || cutoff > 0.5*boxVectors[2][2])
            throw OpenMMException("pGM_MultipoleForce: "+Messages::cutoffTooLarge);
    }

    
    for (int ii = 0; ii < numParticles; ii++) {

        int axisType, multipoleAtomZ, multipoleAtomX, multipoleAtomY;
        double charge, polarity ;
        std::vector<double> molecularDipole;


        owner.getMultipoleParameters(ii, charge, molecularDipole, molecularQuadrupole, axisType, multipoleAtomZ, multipoleAtomX, multipoleAtomY,
                                     polarity);


       // only 'Z-then-X', 'Bisector', Z-Bisect, ThreeFold  currently handled

        if (axisType != pGM_MultipoleForce::ZThenX     && axisType != pGM_MultipoleForce::Bisector &&
            axisType != pGM_MultipoleForce::ZBisect    && axisType != pGM_MultipoleForce::ThreeFold &&
            axisType != pGM_MultipoleForce::ZOnly      && axisType != pGM_MultipoleForce::NoAxisType) {
             std::stringstream buffer;
             buffer << "pGM_MultipoleForce: axis type=" << axisType;
             buffer << " not currently handled - only axisTypes[ ";
             buffer << pGM_MultipoleForce::ZThenX   << ", " << pGM_MultipoleForce::Bisector  << ", ";
             buffer << pGM_MultipoleForce::ZBisect  << ", " << pGM_MultipoleForce::ThreeFold << ", ";
             buffer << pGM_MultipoleForce::NoAxisType;
             buffer << "] (ZThenX, Bisector, Z-Bisect, ThreeFold, NoAxisType) currently handled .";
             throw OpenMMException(buffer.str());
        }
        if (axisType != pGM_MultipoleForce::NoAxisType && (multipoleAtomZ < 0 || multipoleAtomZ >= numParticles)) {
            std::stringstream buffer;
            buffer << "pGM_MultipoleForce: invalid z axis particle: " << multipoleAtomZ;
            throw OpenMMException(buffer.str());
        }
        if (axisType != pGM_MultipoleForce::NoAxisType && axisType != pGM_MultipoleForce::ZOnly &&
                (multipoleAtomX < 0 || multipoleAtomX >= numParticles)) {
            std::stringstream buffer;
            buffer << "pGM_MultipoleForce: invalid x axis particle: " << multipoleAtomX;
            throw OpenMMException(buffer.str());
        }
        if ((axisType == pGM_MultipoleForce::ZBisect || axisType == pGM_MultipoleForce::ThreeFold) &&
                (multipoleAtomY < 0 || multipoleAtomY >= numParticles)) {
            std::stringstream buffer;
            buffer << "pGM_MultipoleForce: invalid y axis particle: " << multipoleAtomY;
            throw OpenMMException(buffer.str());
        }
    }
    kernel = context.getPlatform().createKernel(CalcpGM_MultipoleForceKernel::Name(), context);
    kernel.getAs<CalcpGM_MultipoleForceKernel>().initialize(context.getSystem(), owner);
}

double pGM_MultipoleForceImpl::calcForcesAndEnergy(ContextImpl& context, bool includeForces, bool includeEnergy, int groups) {
    if ((groups&(1<<owner.getForceGroup())) != 0)
        return kernel.getAs<CalcpGM_MultipoleForceKernel>().execute(context, includeForces, includeEnergy);
    return 0.0;
}

std::vector<std::string> pGM_MultipoleForceImpl::getKernelNames() {
    std::vector<std::string> names;
    names.push_back(CalcpGM_MultipoleForceKernel::Name());
    return names;
}

const int* pGM_MultipoleForceImpl::getCovalentDegrees() {
    if (!initializedCovalentDegrees) {
        initializedCovalentDegrees                                      = true;
        CovalentDegrees[pGM_MultipoleForce::Covalent12]               = 1;
        CovalentDegrees[pGM_MultipoleForce::Covalent13]               = 2;
        CovalentDegrees[pGM_MultipoleForce::Covalent14]               = 3;
        CovalentDegrees[pGM_MultipoleForce::Covalent15]               = 4;
        CovalentDegrees[pGM_MultipoleForce::PolarizationCovalent11]   = 0;
        CovalentDegrees[pGM_MultipoleForce::PolarizationCovalent12]   = 1;
        CovalentDegrees[pGM_MultipoleForce::PolarizationCovalent13]   = 2;
        CovalentDegrees[pGM_MultipoleForce::PolarizationCovalent14]   = 3;
    }
    return CovalentDegrees;
}

void pGM_MultipoleForceImpl::getCovalentRange(const pGM_MultipoleForce& force, int atomIndex, const std::vector<pGM_MultipoleForce::CovalentType>& lists,
                                                int* minCovalentIndex, int* maxCovalentIndex) {

    *minCovalentIndex =  999999999;
    *maxCovalentIndex = -999999999;
    for (unsigned int kk = 0; kk < lists.size(); kk++) {
        pGM_MultipoleForce::CovalentType jj = lists[kk];
        std::vector<int> covalentList;
        force.getCovalentMap(atomIndex, jj, covalentList);
        for (unsigned int ii = 0; ii < covalentList.size(); ii++) {
            if (*minCovalentIndex > covalentList[ii]) {
               *minCovalentIndex = covalentList[ii];
            }
            if (*maxCovalentIndex < covalentList[ii]) {
               *maxCovalentIndex = covalentList[ii];
            }
        }
    }
    return;
}

void pGM_MultipoleForceImpl::getCovalentDegree(const pGM_MultipoleForce& force, std::vector<int>& covalentDegree) {
    covalentDegree.resize(pGM_MultipoleForce::CovalentEnd);
    const int* CovalentDegrees = pGM_MultipoleForceImpl::getCovalentDegrees();
    for (unsigned int kk = 0; kk < pGM_MultipoleForce::CovalentEnd; kk++) {
        covalentDegree[kk] = CovalentDegrees[kk];
    }
    return;
}

void pGM_MultipoleForceImpl::getLabFramePermanentDipoles(ContextImpl& context, vector<Vec3>& dipoles) {
    kernel.getAs<CalcpGM_MultipoleForceKernel>().getLabFramePermanentDipoles(context, dipoles);
}

void pGM_MultipoleForceImpl::getInducedDipoles(ContextImpl& context, vector<Vec3>& dipoles) {
    kernel.getAs<CalcpGM_MultipoleForceKernel>().getInducedDipoles(context, dipoles);
}

void pGM_MultipoleForceImpl::getTotalDipoles(ContextImpl& context, vector<Vec3>& dipoles) {
    kernel.getAs<CalcpGM_MultipoleForceKernel>().getTotalDipoles(context, dipoles);
}

void pGM_MultipoleForceImpl::getElectrostaticPotential(ContextImpl& context, const std::vector< Vec3 >& inputGrid,
                                                         std::vector< double >& outputElectrostaticPotential) {
    kernel.getAs<CalcpGM_MultipoleForceKernel>().getElectrostaticPotential(context, inputGrid, outputElectrostaticPotential);
}

void pGM_MultipoleForceImpl::getSystemMultipoleMoments(ContextImpl& context, std::vector< double >& outputMultipoleMoments) {
    kernel.getAs<CalcpGM_MultipoleForceKernel>().getSystemMultipoleMoments(context, outputMultipoleMoments);
}

void pGM_MultipoleForceImpl::updateParametersInContext(ContextImpl& context) {
    kernel.getAs<CalcpGM_MultipoleForceKernel>().copyParametersToContext(context, owner);
    context.systemChanged();
}

void pGM_MultipoleForceImpl::getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    kernel.getAs<CalcpGM_MultipoleForceKernel>().getPMEParameters(alpha, nx, ny, nz);
}
