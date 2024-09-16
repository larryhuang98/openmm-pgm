/* -------------------------------------------------------------------------- *
 *                                OpenMMpGM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2010-2016 Stanford University and the Authors.      *
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

#include "openmm/serialization/pGM_MultipoleForceProxy.h"
#include "openmm/serialization/SerializationNode.h"
#include "openmm/Force.h"
#include "openmm/pGM_MultipoleForce.h"
#include <sstream>

using namespace OpenMM;
using namespace std;

pGM_MultipoleForceProxy::pGM_MultipoleForceProxy() : SerializationProxy("pGM_MultipoleForce") {
}

static void getCovalentTypes(std::vector<std::string>& covalentTypes) {

    covalentTypes.push_back("Covalent12");
    covalentTypes.push_back("Covalent13");
    covalentTypes.push_back("Covalent14");
    covalentTypes.push_back("Covalent15");

    covalentTypes.push_back("PolarizationCovalent11");
    covalentTypes.push_back("PolarizationCovalent12");
    covalentTypes.push_back("PolarizationCovalent13");
    covalentTypes.push_back("PolarizationCovalent14");
}

static void addCovalentMap(SerializationNode& particleExclusions, int particleIndex, std::string mapName, std::vector< int > covalentMap) {
    SerializationNode& map   = particleExclusions.createChildNode(mapName);
    for (unsigned int ii = 0; ii < covalentMap.size(); ii++) {
        map.createChildNode("Cv").setIntProperty("v", covalentMap[ii]);
    }
}

void loadCovalentMap(const SerializationNode& map, std::vector< int >& covalentMap) {
    for (unsigned int ii = 0; ii < map.getChildren().size(); ii++) {
        covalentMap.push_back(map.getChildren()[ii].getIntProperty("v"));
    }
}

void pGM_MultipoleForceProxy::serialize(const void* object, SerializationNode& node) const {
    node.setIntProperty("version", 4);
    const pGM_MultipoleForce& force = *reinterpret_cast<const pGM_MultipoleForce*>(object);

    node.setIntProperty("forceGroup", force.getForceGroup());
    node.setStringProperty("name", force.getName());
    node.setIntProperty("nonbondedMethod",                  force.getNonbondedMethod());
    node.setIntProperty("polarizationType",                 force.getPolarizationType());
    node.setIntProperty("mutualInducedMaxIterations",       force.getMutualInducedMaxIterations());

    node.setDoubleProperty("cutoffDistance",                force.getCutoffDistance());
    double alpha;
    int nx, ny, nz;
    force.getPMEParameters(alpha, nx, ny, nz);
    node.setDoubleProperty("aEwald",                        alpha);
    node.setDoubleProperty("mutualInducedTargetEpsilon",    force.getMutualInducedTargetEpsilon());
    node.setDoubleProperty("ewaldErrorTolerance",           force.getEwaldErrorTolerance());

    SerializationNode& gridDimensionsNode  = node.createChildNode("MultipoleParticleGridDimension");
    gridDimensionsNode.setIntProperty("d0", nx).setIntProperty("d1", ny).setIntProperty("d2", nz); 
    
    //SerializationNode& coefficients = node.createChildNode("ExtrapolationCoefficients");
    //vector<double> coeff = force.getExtrapolationCoefficients();
    //for (int i = 0; i < coeff.size(); i++) {
    //    stringstream key;
    //    key << "c" << i;
    //    coefficients.setDoubleProperty(key.str(), coeff[i]);
    //}

    std::vector<std::string> covalentTypes;
    getCovalentTypes(covalentTypes);

    SerializationNode& particles = node.createChildNode("MultipoleParticles");
    for (unsigned int ii = 0; ii < static_cast<unsigned int>(force.getNumMultipoles()); ii++) {

        double charge, polarity, beta;

        std::vector<double> molecularDipole;
        std::vector<double> molecularQuadrupole;

        force.getMultipoleParameters(ii, charge, molecularDipole, beta, polarity);

        SerializationNode& particle    = particles.createChildNode("Particle");
        
        particle.setDoubleProperty("charge", charge).setDoubleProperty("beta", beta).setDoubleProperty("polarity", polarity);

        SerializationNode& dipole      = particle.createChildNode("Dipole");
        dipole.setDoubleProperty("d0", molecularDipole[0]).setDoubleProperty("d1", molecularDipole[1]).setDoubleProperty("d2", molecularDipole[2]);

        for (unsigned int jj = 0; jj < covalentTypes.size(); jj++) {
            std::vector< int > covalentMap;
            force.getCovalentMap(ii, static_cast<pGM_MultipoleForce::CovalentType>(jj), covalentMap);
            addCovalentMap(particle, ii, covalentTypes[jj], covalentMap);
        }
    }
}

void* pGM_MultipoleForceProxy::deserialize(const SerializationNode& node) const {
    int version = node.getIntProperty("version");
    if (version < 0 || version > 4)
        throw OpenMMException("Unsupported version number");
    pGM_MultipoleForce* force = new pGM_MultipoleForce();

    try {
        force->setForceGroup(node.getIntProperty("forceGroup", 0));
        force->setName(node.getStringProperty("name", force->getName()));
        force->setNonbondedMethod(static_cast<pGM_MultipoleForce::NonbondedMethod>(node.getIntProperty("nonbondedMethod")));
        if (version >= 2)
            force->setPolarizationType(static_cast<pGM_MultipoleForce::PolarizationType>(node.getIntProperty("polarizationType")));
        force->setMutualInducedMaxIterations(node.getIntProperty("mutualInducedMaxIterations"));

        force->setCutoffDistance(node.getDoubleProperty("cutoffDistance"));
        force->setMutualInducedTargetEpsilon(node.getDoubleProperty("mutualInducedTargetEpsilon"));
        force->setEwaldErrorTolerance(node.getDoubleProperty("ewaldErrorTolerance"));

        const SerializationNode& gridDimensionsNode  = node.getChildNode("MultipoleParticleGridDimension");
        force->setPMEParameters(node.getDoubleProperty("aEwald"), gridDimensionsNode.getIntProperty("d0"), gridDimensionsNode.getIntProperty("d1"), gridDimensionsNode.getIntProperty("d2"));
    

        std::vector<std::string> covalentTypes;
        getCovalentTypes(covalentTypes);

        const SerializationNode& particles = node.getChildNode("MultipoleParticles");
        for (unsigned int ii = 0; ii < particles.getChildren().size(); ii++) {

            const SerializationNode& particle = particles.getChildren()[ii];

            std::vector<double> molecularDipole;
            const SerializationNode& dipole = particle.getChildNode("Dipole");
            molecularDipole.push_back(dipole.getDoubleProperty("d0"));
            molecularDipole.push_back(dipole.getDoubleProperty("d1"));
            molecularDipole.push_back(dipole.getDoubleProperty("d2"));

            std::vector< int > covalentMap;
            loadCovalentMap(particle.getChildNode(covalentTypes[0]), covalentMap);
            force->setCovalentMap(ii, static_cast<pGM_MultipoleForce::CovalentType>(0), covalentMap);


            force->addMultipole(particle.getDoubleProperty("charge"), molecularDipole, covalentMap, particle.getDoubleProperty("beta"),  particle.getDoubleProperty("polarity"));

            // covalent maps : currently only one type

            // for (unsigned int jj = 0; jj < covalentTypes.size(); jj++) {
            //     std::vector< int > covalentMap;
            //     loadCovalentMap(particle.getChildNode(covalentTypes[jj]), covalentMap);
            //     force->setCovalentMap(ii, static_cast<pGM_MultipoleForce::CovalentType>(jj), covalentMap);
            // }
        }

    }
    catch (...) {
        delete force;
        throw;
    }

    return force;
}
