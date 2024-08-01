#ifndef OPENMM_pGM_MULTIPOLE_FORCEIMPL_H_
#define OPENMM_pGM_MULTIPOLE_FORCEIMPL_H_

/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2013 Stanford University and the Authors.           *
 * Authors: Peter Eastman                                                     *
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

#include "openmm/internal/ForceImpl.h"
#include "openmm/pGM_MultipoleForce.h"
#include "openmm/Kernel.h"
#include <utility>
#include <string>

namespace OpenMM {

class System;

/**
 * This is the internal implementation of pGM_MultipoleForce.
 */

class OPENMM_EXPORT_pGM pGM_MultipoleForceImpl : public ForceImpl {
public:
    pGM_MultipoleForceImpl(const pGM_MultipoleForce& owner);
    ~pGM_MultipoleForceImpl();
    void initialize(ContextImpl& context);
    const pGM_MultipoleForce& getOwner() const {
        return owner;
    }
    void updateContextState(ContextImpl& context, bool& forcesInvalid) {
        // This force field doesn't update the state directly.
    }
    double calcForcesAndEnergy(ContextImpl& context, bool includeForces, bool includeEnergy, int groups);
    std::map<std::string, double> getDefaultParameters() {
        return std::map<std::string, double>(); // This force field doesn't define any parameters.
    }
    std::vector<std::string> getKernelNames();

    /**
     * Get the CovalentMap for an atom
     *
     * @param force                pGM_MultipoleForce force reference
     * @param index                the index of the atom for which to set parameters
     * @param minCovalentIndex     minimum covalent index
     * @param maxCovalentIndex     maximum covalent index
     */
    static void getCovalentRange(const pGM_MultipoleForce& force, int index,
                                 const std::vector< pGM_MultipoleForce::CovalentType>& lists,
                                 int* minCovalentIndex, int* maxCovalentIndex);

    /**
     * Get the covalent degree for the  CovalentEnd lists
     *
     * @param force                pGM_MultipoleForce force reference
     * @param covalentDegree      covalent degrees for the CovalentEnd lists
     */
    static void getCovalentDegree(const pGM_MultipoleForce& force, std::vector<int>& covalentDegree);
    void getLabFramePermanentDipoles(ContextImpl& context, std::vector<Vec3>& dipoles);
    void getInducedDipoles(ContextImpl& context, std::vector<Vec3>& dipoles);
    void getTotalDipoles(ContextImpl& context, std::vector<Vec3>& dipoles);

    void getElectrostaticPotential(ContextImpl& context, const std::vector< Vec3 >& inputGrid,
                                   std::vector< double >& outputElectrostaticPotential);

    void getSystemMultipoleMoments(ContextImpl& context, std::vector< double >& outputMultipoleMoments);
    void updateParametersInContext(ContextImpl& context);
    void getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const;


private:
    const pGM_MultipoleForce& owner;
    Kernel kernel;

    static int CovalentDegrees[pGM_MultipoleForce::CovalentEnd];
    static bool initializedCovalentDegrees;
    static const int* getCovalentDegrees();
};

} // namespace OpenMM

#endif /*OPENMM_pGM_MULTIPOLE_FORCE_IMPL_H_*/
