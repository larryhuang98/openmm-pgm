/* -------------------------------------------------------------------------- *
 *                                OpenMM_pGM                                  *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2008-2016 Stanford University and the Authors.      *
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

#include "openmm/Force.h"
#include "openmm/OpenMMException.h"
#include "openmm/pGM_MultipoleForce.h"
#include "openmm/internal/pGM_MultipoleForceImpl.h"
#include "SimTKOpenMMRealType.h"
#include <stdio.h>

using namespace OpenMM;
using std::string;
using std::vector;

pGM_MultipoleForce::pGM_MultipoleForce() : nonbondedMethod(PME), polarizationType(Mutual), pmeBSplineOrder(5), cutoffDistance(1.0), ewaldErrorTol(1e-4), mutualInducedMaxIterations(60),
                                               mutualInducedTargetEpsilon(1e-5), scalingDistanceCutoff(100.0), electricConstant(ONE_4PI_EPS0), alpha(1.0), nx(50), ny(50), nz(50) {
    

}

pGM_MultipoleForce::NonbondedMethod pGM_MultipoleForce::getNonbondedMethod() const {
    return nonbondedMethod;
}

void pGM_MultipoleForce::setNonbondedMethod(pGM_MultipoleForce::NonbondedMethod method) {
    if (method < 0 || method > 1)
        throw OpenMMException("pGM_MultipoleForce: Illegal value for nonbonded method, currently pGM model only supports PME method");
    nonbondedMethod = method;
}

pGM_MultipoleForce::PolarizationType pGM_MultipoleForce::getPolarizationType() const {
    return polarizationType;
}

void pGM_MultipoleForce::setPolarizationType(pGM_MultipoleForce::PolarizationType type) {
    if (type < 0 || type > 1)
        throw OpenMMException("pGM_MultipoleForce: Illegal value for PolarizationType, currently pGM model only supports Mutual Induced Dipole");
    polarizationType = type;
}


double pGM_MultipoleForce::getCutoffDistance() const {
    return cutoffDistance;
}

void pGM_MultipoleForce::setCutoffDistance(double distance) {
    cutoffDistance = distance;
}

void pGM_MultipoleForce::getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    alpha = this->alpha;
    nx = this->nx;
    ny = this->ny;
    nz = this->nz;
}

void pGM_MultipoleForce::setPMEParameters(double alpha, int nx, int ny, int nz) {
    this->alpha = alpha;
    this->nx = nx;
    this->ny = ny;
    this->nz = nz;
}

double pGM_MultipoleForce::getAEwald() const { 
    return alpha; 
} 
 
void pGM_MultipoleForce::setAEwald(double inputAewald) { 
    alpha = inputAewald; 
} 
 
int pGM_MultipoleForce::getPmeBSplineOrder() const { 
    return pmeBSplineOrder; 
} 
 
void pGM_MultipoleForce::getPmeGridDimensions(std::vector<int>& gridDimension) const { 
    if (gridDimension.size() < 3)
        gridDimension.resize(3);
    gridDimension[0] = nx;
    gridDimension[1] = ny;
    gridDimension[2] = nz;
} 
 
void pGM_MultipoleForce::setPmeGridDimensions(const std::vector<int>& gridDimension) {
    nx = gridDimension[0];
    ny = gridDimension[1];
    nz = gridDimension[2];
}

void pGM_MultipoleForce::getPMEParametersInContext(const Context& context, double& alpha, int& nx, int& ny, int& nz) const {
    dynamic_cast<const pGM_MultipoleForceImpl&>(getImplInContext(context)).getPMEParameters(alpha, nx, ny, nz);
}

int pGM_MultipoleForce::getMutualInducedMaxIterations() const {
    return mutualInducedMaxIterations;
}

void pGM_MultipoleForce::setMutualInducedMaxIterations(int inputMutualInducedMaxIterations) {
    mutualInducedMaxIterations = inputMutualInducedMaxIterations;
}

double pGM_MultipoleForce::getMutualInducedTargetEpsilon() const {
    return mutualInducedTargetEpsilon;
}

void pGM_MultipoleForce::setMutualInducedTargetEpsilon(double inputMutualInducedTargetEpsilon) {
    mutualInducedTargetEpsilon = inputMutualInducedTargetEpsilon;
}

double pGM_MultipoleForce::getEwaldErrorTolerance() const {
    return ewaldErrorTol;
}

void pGM_MultipoleForce::setEwaldErrorTolerance(double tol) {
    ewaldErrorTol = tol;
}

int pGM_MultipoleForce::addMultipole(double charge, const std::vector<double>& molecularDipole, const std::vector<int>& covalentAtoms, double beta, double polarity) {
    multipoles.push_back(MultipoleInfo(charge, molecularDipole, polarity, covalentAtoms.size()));
    return multipoles.size()-1;
}

void pGM_MultipoleForce::getMultipoleParameters(int index, double& charge, std::vector<double>& molecularDipole, 
                                std::vector<int>& covalentAtoms, double beta,  double& polarity) const {
    charge                      = multipoles[index].charge;

    covalentAtoms               = multipoles[index].covalentInfo[0];

    beta                        = multipoles[index].beta;

    molecularDipole.resize(covalentAtoms.size());
    for (int ii=0; ii< covalentAtoms.size(); ii++){
        molecularDipole[ii]          = multipoles[index].molecularDipole[ii];
    }


    polarity                    = multipoles[index].polarity;
}

void pGM_MultipoleForce::setMultipoleParameters(int index, double& charge, std::vector<double>& molecularDipole, 
                                const std::vector<int>& covalentAtoms, double beta,  double& polarity) {

    multipoles[index].charge                      = charge;

    multipoles[index].molecularDipole[0]          = molecularDipole[0];
    multipoles[index].molecularDipole[1]          = molecularDipole[1];
    multipoles[index].molecularDipole[2]          = molecularDipole[2];


    multipoles[index].polarity                    = polarity;

}

void pGM_MultipoleForce::setCovalentMap(int index, CovalentType typeId, const std::vector<int>& covalentAtoms) {

    std::vector<int>& covalentList = multipoles[index].covalentInfo[typeId];
    covalentList.resize(covalentAtoms.size());
    for (unsigned int ii = 0; ii < covalentAtoms.size(); ii++) {
       covalentList[ii] = covalentAtoms[ii];
    }
}

void pGM_MultipoleForce::getCovalentMap(int index, CovalentType typeId, std::vector<int>& covalentAtoms) const {

    // load covalent atom index entries for atomId==index and covalentId==typeId into covalentAtoms

    std::vector<int> covalentList = multipoles[index].covalentInfo[typeId];
    covalentAtoms.resize(covalentList.size());
    for (unsigned int ii = 0; ii < covalentList.size(); ii++) {
       covalentAtoms[ii] = covalentList[ii];
    }
}

void pGM_MultipoleForce::getCovalentMaps(int index, std::vector< std::vector<int> >& covalentLists) const {


    for (unsigned int jj = 0; jj < covalentLists.size(); jj++) {
        std::vector<int> covalentList = multipoles[index].covalentInfo[jj];
        std::vector<int> covalentAtoms;
        covalentAtoms.resize(covalentList.size());
        for (unsigned int ii = 0; ii < covalentList.size(); ii++) {
           covalentAtoms[ii] = covalentList[ii];
        }
        covalentLists[jj] = covalentAtoms;
    }
}

void pGM_MultipoleForce::getInducedDipoles(Context& context, vector<Vec3>& dipoles) {
    dynamic_cast<pGM_MultipoleForceImpl&>(getImplInContext(context)).getInducedDipoles(getContextImpl(context), dipoles);
}

void pGM_MultipoleForce::getLabFramePermanentDipoles(Context& context, vector<Vec3>& dipoles) {
    dynamic_cast<pGM_MultipoleForceImpl&>(getImplInContext(context)).getLabFramePermanentDipoles(getContextImpl(context), dipoles);
}

void pGM_MultipoleForce::getTotalDipoles(Context& context, vector<Vec3>& dipoles) {
    dynamic_cast<pGM_MultipoleForceImpl&>(getImplInContext(context)).getTotalDipoles(getContextImpl(context), dipoles);
}

void pGM_MultipoleForce::getElectrostaticPotential(const std::vector< Vec3 >& inputGrid, Context& context, std::vector< double >& outputElectrostaticPotential) {
    dynamic_cast<pGM_MultipoleForceImpl&>(getImplInContext(context)).getElectrostaticPotential(getContextImpl(context), inputGrid, outputElectrostaticPotential);
}

void pGM_MultipoleForce::getSystemMultipoleMoments(Context& context, std::vector< double >& outputMultipoleMoments) {
    dynamic_cast<pGM_MultipoleForceImpl&>(getImplInContext(context)).getSystemMultipoleMoments(getContextImpl(context), outputMultipoleMoments);
}

ForceImpl* pGM_MultipoleForce::createImpl()  const {
    return new pGM_MultipoleForceImpl(*this);
}

void pGM_MultipoleForce::updateParametersInContext(Context& context) {
    dynamic_cast<pGM_MultipoleForceImpl&>(getImplInContext(context)).updateParametersInContext(getContextImpl(context));
}
