/* -------------------------------------------------------------------------- *
 *                              OpenMMpGM                                     *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2008-2016 Stanford University and the Authors.      *
 * Authors: Mark Friedrichs, Peter Eastman                                    *
 * Contributors:                                                              *
 *                                                                            *
 * This program is free software: you can redistribute it and/or modify       *
 * it under the terms of the GNU Lesser General Public License as published   *
 * by the Free Software Foundation, either version 3 of the License, or       *
 * (at your option) any later version.                                        *
 *                                                                            *
 * This program is distributed in the hope that it will be useful,            *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of             *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the              *
 * GNU Lesser General Public License for more details.                        *
 *                                                                            *
 * You should have received a copy of the GNU Lesser General Public License   *
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.      *
 * -------------------------------------------------------------------------- */

#include "pGM_ReferenceKernelFactory.h"
#include "pGM_ReferenceKernels.h"
#include "ReferencePlatform.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/OpenMMException.h"

using namespace OpenMM;

#ifdef OPENMM_BUILDING_STATIC_LIBRARY
static void registerPlatforms() {
#else
extern "C" OPENMM_EXPORT void registerPlatforms() {
#endif
}

#ifdef OPENMM_BUILDING_STATIC_LIBRARY
static void registerKernelFactories() {
#else
extern "C" OPENMM_EXPORT void registerKernelFactories() {
#endif
    for (int i = 0; i < Platform::getNumPlatforms(); i++) {
        Platform& platform = Platform::getPlatform(i);
        if (dynamic_cast<ReferencePlatform*>(&platform) != NULL) {
             pGM_ReferenceKernelFactory* factory = new pGM_ReferenceKernelFactory();
             platform.registerKernelFactory(Calc_pGM_MultipoleForceKernel::Name(), factory);
        }
    }
}

extern "C" OPENMM_EXPORT void register_pGM_ReferenceKernelFactories() {
    registerKernelFactories();
}

KernelImpl* pGM_ReferenceKernelFactory::createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const {

    if (name == Calc_pGM_MultipoleForceKernel::Name())
        return new ReferenceCalc_pGM_MultipoleForceKernel(name, platform, context.getSystem());

    throw OpenMMException((std::string("Tried to create kernel with illegal kernel name '")+name+"'").c_str());
}