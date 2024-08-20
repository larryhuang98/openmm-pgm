#ifndef OPENMM_pGM_MULTIPOLE_FORCE_H_
#define OPENMM_pGM_MULTIPOLE_FORCE_H_

/* -------------------------------------------------------------------------- *
 *                              OpenMMpGM_                                  *
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
#include "internal/windowsExportpGM.h"
#include "openmm/Vec3.h"

#include <sstream>
#include <vector>

namespace OpenMM {

/**
 * This class implements the pGM multipole interaction.
 *
 * To use it, create an pGM_MultipoleForce object then call addMultipole() once for each atom.  After
 * an entry has been added, you can modify its force field parameters by calling setMultipoleParameters().
 * This will have no effect on Contexts that already exist unless you call updateParametersInContext().
 */

class OPENMM_EXPORT_pGM pGM_MultipoleForce : public Force {

public:

    enum NonbondedMethod {

        /**
         * No cutoff is applied to nonbonded interactions.  The full set of N^2 interactions is computed exactly.
         * This necessarily means that periodic boundary conditions cannot be used.  This is the default.
         */
        // NoCutoff = 0, to be implemented

        /**
         * Periodic boundary conditions are used, and Particle-Mesh Ewald (PME) summation is used to compute the interaction of each particle
         * with all periodic copies of every other particle.
         */
        PME = 1

        /**
         * Isotropic periodic sum, will be implemented later
         * 
         */
        // IPS = 1
    };

    enum PolarizationType {
        // pGM model currently only have mutual induced dipoles
        /**
         * Full mutually induced polarization.  The dipoles are iterated until the converge to the accuracy specified
         * by getMutualInducedTargetEpsilon().
         */
        Mutual = 0,

        /**
         * Direct polarization approximation.  The induced dipoles depend only on the fixed multipoles, not on other
         * induced dipoles.
         */
        //Direct = 1,

        /**
         * Extrapolated perturbation theory approximation.  The dipoles are iterated a few times, and then an analytic
         * approximation is used to extrapolate to the fully converged values.  Call setExtrapolationCoefficients()
         * to set the coefficients used for the extrapolation.  The default coefficients used in this release are
         * [-0.154, 0.017, 0.658, 0.474], but be aware that those may change in a future release.
         */
        //Extrapolated = 2

    };

    // larry : CovalentType: Currently, we do not have covalent bonds here, we only have non-bonded interaction
    enum CovalentType { Covalent = 0, nonInteractingCovalent = 1 };

    /**
     * Create an pGM_MultipoleForce.
     */
    pGM_MultipoleForce();

    /**
     * Get the number of particles in the potential function
     */
    int getNumMultipoles() const {
        return multipoles.size();
    }

    /**
     * Get the method used for handling long-range nonbonded interactions.
     */
    NonbondedMethod getNonbondedMethod() const;

    /**
     * Set the method used for handling long-range nonbonded interactions.
     */
    void setNonbondedMethod(NonbondedMethod method);

    /**
     * Get polarization type
     */
    PolarizationType getPolarizationType() const;

    /**
     * Set the polarization type
     */
    void setPolarizationType(PolarizationType type);

    /**
     * Get the cutoff distance (in nm) being used for nonbonded interactions.  If the NonbondedMethod in use
     * is NoCutoff, this value will have no effect.
     *
     * @return the cutoff distance, measured in nm
     */
    double getCutoffDistance() const;

    /**
     * Set the cutoff distance (in nm) being used for nonbonded interactions.  If the NonbondedMethod in use
     * is NoCutoff, this value will have no effect.
     *
     * @param distance    the cutoff distance, measured in nm
     */
    void setCutoffDistance(double distance);

    /**
     * Get the parameters to use for PME calculations.  If alpha is 0 (the default), these parameters are
     * ignored and instead their values are chosen based on the Ewald error tolerance.
     *
     * @param[out] alpha   the separation parameter
     * @param[out] nx      the number of grid points along the X axis
     * @param[out] ny      the number of grid points along the Y axis
     * @param[out] nz      the number of grid points along the Z axis
     */
    void getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const;

    /**
     * Set the parameters to use for PME calculations.  If alpha is 0 (the default), these parameters are
     * ignored and instead their values are chosen based on the Ewald error tolerance.
     *
     * @param alpha   the separation parameter
     * @param nx      the number of grid points along the X axis
     * @param ny      the number of grid points along the Y axis
     * @param nz      the number of grid points along the Z axis
     */
    void setPMEParameters(double alpha, int nx, int ny, int nz);

    /**
     * Get the Ewald alpha parameter.  If this is 0 (the default), a value is chosen automatically
     * based on the Ewald error tolerance.
     *
     * @return the Ewald alpha parameter
     * @deprecated This method exists only for backward compatibility.  Use getPMEParameters() instead.
     */
    double getAEwald() const;

    /**
     * Set the Ewald alpha parameter.  If this is 0 (the default), a value is chosen automatically
     * based on the Ewald error tolerance.
     *
     * @param aewald alpha parameter
     * @deprecated This method exists only for backward compatibility.  Use setPMEParameters() instead.
     */
    void setAEwald(double aewald);

    /**
     * Get the B-spline order to use for PME charge spreading
     *
     * @return the B-spline order
     */
    int getPmeBSplineOrder() const;

    /**
     * Get the PME grid dimensions.  If Ewald alpha is 0 (the default), this is ignored and grid dimensions
     * are chosen automatically based on the Ewald error tolerance.
     *
     * @return the PME grid dimensions
     * @deprecated This method exists only for backward compatibility.  Use getPMEParameters() instead.
     */
    void getPmeGridDimensions(std::vector<int>& gridDimension) const;

   /**
     * Set the PME grid dimensions.  If Ewald alpha is 0 (the default), this is ignored and grid dimensions
     * are chosen automatically based on the Ewald error tolerance.
     *
     * @param gridDimension   the PME grid dimensions
     * @deprecated This method exists only for backward compatibility.  Use setPMEParameters() instead.
     */
    void setPmeGridDimensions(const std::vector<int>& gridDimension);

    /**
     * Get the parameters being used for PME in a particular Context.  Because some platforms have restrictions
     * on the allowed grid sizes, the values that are actually used may be slightly different from those
     * specified with setPmeGridDimensions(), or the standard values calculated based on the Ewald error tolerance.
     * See the manual for details.
     *
     * @param context      the Context for which to get the parameters
     * @param[out] alpha   the separation parameter
     * @param[out] nx      the number of grid points along the X axis
     * @param[out] ny      the number of grid points along the Y axis
     * @param[out] nz      the number of grid points along the Z axis
     */
    void getPMEParametersInContext(const Context& context, double& alpha, int& nx, int& ny, int& nz) const;

    /**
     * Add multipole-related info for a particle
     *
     * @param charge               the particle's charge
     * @param molecularDipole      the particle's molecular dipole (vector of size 3)
     * @param multipoleAtomZ       index of first atom used in constructing lab<->molecular frames
     * @param multipoleAtomX       index of second atom used in constructing lab<->molecular frames
     * @param multipoleAtomY       index of second atom used in constructing lab<->molecular frames
     * @param polarity             polarity parameter
     *
     * @return the index of the particle that was added
     */
    int addMultipole(double charge, const std::vector<double>& molecularDipole, const std::vector<int>& covalentAtoms, double beta, double polarity);
    // larry: note! need to add the beta value for each multipole
    /**
     * Get the multipole parameters for a particle.
     *
     * @param index                     the index of the atom for which to get parameters
     * @param[out] charge               the particle's charge
     * @param[out] molecularDipole      the particle's molecular dipole (vector of size of total covalent bonds for an atom)
     * @param[out] polarity             polarity parameter
     */
    void getMultipoleParameters(int index, double& charge, std::vector<double>& molecularDipole, 
                                std::vector<int>& covalentAtoms, double beta,  double& polarity) const;

    /**
     * Set the multipole parameters for a particle.
     *
     * @param index                the index of the atom for which to set parameters
     * @param charge               the particle's charge
     * @param molecularDipole      the particle's molecular dipole (vector of size 3)
     * @param covalentAtoms        index of atoms for local dipoles
     * @param beta                 the pGM Gaussian radius
     * @param polarity             polarity parameter
     */
    void setMultipoleParameters(int index, double& charge, std::vector<double>& molecularDipole, 
                                const std::vector<int>& covalentAtoms, double beta,  double& polarity);

    /**
     * Set the CovalentMap for an atom
     *
     * @param index                the index of the atom for which to set parameters
     * @param typeId               CovalentTypes type
     * @param covalentAtoms        vector of covalent atoms associated w/ the specfied CovalentType
     */
    void setCovalentMap(int index, CovalentType typeId, const std::vector<int>& covalentAtoms);

    /**
     * Get the CovalentMap for an atom
     *
     * @param index                the index of the atom for which to set parameters
     * @param typeId               CovalentTypes type
     * @param[out] covalentAtoms   output vector of covalent atoms associated w/ the specfied CovalentType
     */
    void getCovalentMap(int index, CovalentType typeId, std::vector<int>& covalentAtoms) const;

    /**
     * Get the CovalentMap for an atom
     *
     * @param index                the index of the atom for which to set parameters
     * @param[out] covalentLists   output vector of covalent lists of atoms
     */
    void getCovalentMaps(int index, std::vector < std::vector<int> >& covalentLists) const;

    /**
     * Get the max number of iterations to be used in calculating the mutual induced dipoles
     *
     * @return max number of iterations
     */
    int getMutualInducedMaxIterations(void) const;

    /**
     * Set the max number of iterations to be used in calculating the mutual induced dipoles
     *
     * @param inputMutualInducedMaxIterations   number of iterations
     */
    void setMutualInducedMaxIterations(int inputMutualInducedMaxIterations);

    /**
     * Get the target epsilon to be used to test for convergence of iterative method used in calculating the mutual induced dipoles
     *
     * @return target epsilon
     */
    double getMutualInducedTargetEpsilon(void) const;

    /**
     * Set the target epsilon to be used to test for convergence of iterative method used in calculating the mutual induced dipoles
     *
     * @param inputMutualInducedTargetEpsilon   target epsilon
     */
    void setMutualInducedTargetEpsilon(double inputMutualInducedTargetEpsilon);

    /**
     * Get the error tolerance for Ewald summation.  This corresponds to the fractional error in the forces
     * which is acceptable.  This value is used to select the grid dimensions and separation (alpha)
     * parameter so that the average error level will be less than the tolerance.  There is not a
     * rigorous guarantee that all forces on all atoms will be less than the tolerance, however.
     *
     * This can be overridden by explicitly setting an alpha parameter and grid dimensions to use.
     */
    double getEwaldErrorTolerance() const;
    /**
     * Get the error tolerance for Ewald summation.  This corresponds to the fractional error in the forces
     * which is acceptable.  This value is used to select the grid dimensions and separation (alpha)
     * parameter so that the average error level will be less than the tolerance.  There is not a
     * rigorous guarantee that all forces on all atoms will be less than the tolerance, however.
     *
     * This can be overridden by explicitly setting an alpha parameter and grid dimensions to use.
     */
    void setEwaldErrorTolerance(double tol);
    /**
     * Get the fixed dipole moments of all particles in the global reference frame.
     *
     * @param context         the Context for which to get the fixed dipoles
     * @param[out] dipoles    the fixed dipole moment of particle i is stored into the i'th element
     */
    void getLabFramePermanentDipoles(Context& context, std::vector<Vec3>& dipoles);
    /**
     * Get the induced dipole moments of all particles.
     *
     * @param context         the Context for which to get the induced dipoles
     * @param[out] dipoles    the induced dipole moment of particle i is stored into the i'th element
     */
    void getInducedDipoles(Context& context, std::vector<Vec3>& dipoles);

    /**
     * Get the total dipole moments (fixed plus induced) of all particles.
     *
     * @param context         the Context for which to get the total dipoles
     * @param[out] dipoles    the total dipole moment of particle i is stored into the i'th element
     */
    void getTotalDipoles(Context& context, std::vector<Vec3>& dipoles);

    /**
     * Get the electrostatic potential.
     *
     * @param inputGrid    input grid points over which the potential is to be evaluated
     * @param context      context
     * @param[out] outputElectrostaticPotential output potential
     */

    void getElectrostaticPotential(const std::vector< Vec3 >& inputGrid,
                                    Context& context, std::vector< double >& outputElectrostaticPotential);

    /**
     * Get the system multipole moments.
     *
     * This method is most useful for non-periodic systems.  When called for a periodic system, only the
     * <i>lowest nonvanishing moment</i> has a well defined value.  This means that if the system has a net
     * nonzero charge, the dipole and quadrupole moments are not well defined and should be ignored.  If the
     * net charge is zero, the dipole moment is well defined (and really represents a dipole density), but
     * the quadrupole moment is still undefined and should be ignored.
     *
     * @param context      context
     * @param[out] outputMultipoleMoments (charge,
                                           dipole_x, dipole_y, dipole_z)
     */
    void getSystemMultipoleMoments(Context& context, std::vector< double >& outputMultipoleMoments);
    /**
     * Update the multipole parameters in a Context to match those stored in this Force object.  This method
     * provides an efficient method to update certain parameters in an existing Context without needing to reinitialize it.
     * Simply call setMultipoleParameters() to modify this object's parameters, then call updateParametersInContext() to
     * copy them over to the Context.
     *
     * This method has several limitations.  The only information it updates is the parameters of multipoles.
     * All other aspects of the Force (the nonbonded method, the cutoff distance, etc.) are unaffected and can only be
     * changed by reinitializing the Context.  Furthermore, this method cannot be used to add new multipoles,
     * only to change the parameters of existing ones.
     */
    void updateParametersInContext(Context& context);
    /**
     * Returns whether or not this force makes use of periodic boundary
     * conditions.
     *
     * @returns true if nonbondedMethod uses PBC and false otherwise
     */
    bool usesPeriodicBoundaryConditions() const {
        return nonbondedMethod == pGM_MultipoleForce::PME;
    }
protected:
    ForceImpl* createImpl() const;
private:
    NonbondedMethod nonbondedMethod;
    PolarizationType polarizationType;
    double cutoffDistance;
    double alpha;
    int pmeBSplineOrder, nx, ny, nz;
    int mutualInducedMaxIterations;

    double mutualInducedTargetEpsilon;
    double scalingDistanceCutoff;
    double electricConstant;
    double ewaldErrorTol;
    class MultipoleInfo;
    std::vector<MultipoleInfo> multipoles;
};

/**
 * This is an internal class used to record information about a multipole.
 * @private
 */
class pGM_MultipoleForce::MultipoleInfo {
public:

    int numCovalentBonds;

    double charge,  polarity, beta;

    std::vector<double> molecularDipole;
    
    std::vector< std::vector<int> > covalentInfo;

    MultipoleInfo() {

        numCovalentBonds = 0;
        
        charge   =  0.0;

        beta     =  1.0;

        molecularDipole.resize(numCovalentBonds);

    }

    MultipoleInfo(double charge, const std::vector<double>& inputMolecularDipole, double polarity, int numCovalentBonds) : charge(charge),  polarity(polarity) {

       covalentInfo.resize(numCovalentBonds);

       molecularDipole.resize(3);
       for (int ii; ii < numCovalentBonds; ii++) {
            molecularDipole[ii]    = inputMolecularDipole[ii];
       }

    }
};

} // namespace OpenMM

#endif /*OPENMM_pGM_MULTIPOLE_FORCE_H_*/
