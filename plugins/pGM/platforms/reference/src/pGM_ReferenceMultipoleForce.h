/* Portions copyright (c) 2006-2022 Stanford University and Simbios.
 * Contributors: Pande Group
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject
 * to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE
 * LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 * OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef __pGM_ReferenceMultipoleForce_H__
#define __pGM_ReferenceMultipoleForce_H__

#include "openmm/pGM_MultipoleForce.h"
#include "openmm/Vec3.h"
#include <map>
#include <complex>

namespace OpenMM {

typedef std::map< unsigned int, double> MapIntRealOpenMM;
typedef MapIntRealOpenMM::iterator MapIntRealOpenMMI;
typedef MapIntRealOpenMM::const_iterator MapIntRealOpenMMCI;


// A few useful constants for the spherical harmonic multipole code.
const double oneThird = 1.0/3.0;
const double twoThirds = 2.0/3.0;
const double fourThirds = 4.0/3.0;
const double fourSqrtOneThird = 4.0/sqrt(3.0);
const double sqrtFourThirds = 2.0/sqrt(3.0);
const double sqrtOneThird = 1.0/sqrt(3.0);
const double sqrtThree = sqrt(3.0);
const double oneNinth = 1.0/9.0;
const double fourOverFortyFive = 4.0/45.0;
const double fourOverFifteen = 4.0/15.0;


/**
 * 2-dimensional int vector
 */
class int2 {
public:
    /**
     * Create a int2 whose elements are all 0.
     */
    int2() {
        data[0] = data[1] = 0;
    }
    /**
     * Create a int2 with specified x, y components.
     */
    int2(int x, int y) {
        data[0] = x;
        data[1] = y;
    }
    int operator[](int index) const {
        assert(index >= 0 && index < 2);
        return data[index];
    }
    int& operator[](int index) {
        assert(index >= 0 && index < 2);
        return data[index];
    }

    // Arithmetic operators

    // unary plus
    int2 operator+() const {
        return int2(*this);
    }

    // plus
    int2 operator+(const int2& rhs) const {
        const int2& lhs = *this;
        return int2(lhs[0] + rhs[0], lhs[1] + rhs[1]);
    }

    int2& operator+=(const int2& rhs) {
        data[0] += rhs[0];
        data[1] += rhs[1];
        return *this;
    }

    int2& operator-=(const int2& rhs) {
        data[0] -= rhs[0];
        data[1] -= rhs[1];
        return *this;
    }

private:
    int data[2];
};

/**
 * 3-dimensional int vector
 */
class IntVec {
public:
    /**
     * Create a IntVec whose elements are all 0.
     */
    IntVec() {
        data[0] = data[1] = data[2] = 0;
    }
    /**
     * Create a IntVec with specified x, y, z, w components.
     */
    IntVec(int x, int y, int z) {
        data[0] = x;
        data[1] = y;
        data[2] = z;
    }
    int operator[](int index) const {
        assert(index >= 0 && index < 3);
        return data[index];
    }
    int& operator[](int index) {
        assert(index >= 0 && index < 3);
        return data[index];
    }

    // Arithmetic operators

    // unary plus
    IntVec operator+() const {
        return IntVec(*this);
    }

    // plus
    IntVec operator+(const IntVec& rhs) const {
        const IntVec& lhs = *this;
        return IntVec(lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2]);
    }

    IntVec& operator+=(const IntVec& rhs) {
        data[0] += rhs[0];
        data[1] += rhs[1];
        data[2] += rhs[2];
        return *this;
    }

    IntVec& operator-=(const IntVec& rhs) {
        data[0] -= rhs[0];
        data[1] -= rhs[1];
        data[2] -= rhs[2];
        return *this;
    }

private:
    int data[3];
};

/**
 * 4-dimensional double vector
 */
class double4 {
public:
    /**
     * Create a double4 whose elements are all 0.
     */
    double4() {
        data[0] = data[1] = data[2] = data[3] = 0.0;
    }
    /**
     * Create a double4 with specified x, y, z, w components.
     */
    double4(double x, double y, double z, double w) {
        data[0] = x;
        data[1] = y;
        data[2] = z;
        data[3] = w;
    }
    double operator[](int index) const {
        assert(index >= 0 && index < 4);
        return data[index];
    }
    double& operator[](int index) {
        assert(index >= 0 && index < 4);
        return data[index];
    }

    // Arithmetic operators

    // unary plus
    double4 operator+() const {
        return double4(*this);
    }

    // plus
    double4 operator+(const double4& rhs) const {
        const double4& lhs = *this;
        return double4(lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2],lhs[3] + rhs[3]);
    }

    double4& operator+=(const double4& rhs) {
        data[0] += rhs[0];
        data[1] += rhs[1];
        data[2] += rhs[2];
        data[3] += rhs[3];
        return *this;
    }

    double4& operator-=(const double4& rhs) {
        data[0] -= rhs[0];
        data[1] -= rhs[1];
        data[2] -= rhs[2];
        data[3] -= rhs[3];
        return *this;
    }

private:
    double data[4];
};

using namespace OpenMM;

class pGM_ReferenceMultipoleForce {

   /**
    * pGM_ReferenceMultipoleForce is base class for MultipoleForce calculations
    *
    * Below is a outline of the sequence of methods called to evaluate the force and energy 
    * for each scenario: Generalized Kirkwood (GK) and PME.
    *
    * If 'virtual' appears before the method name, the method is overridden in one or more of the derived classes.
    *
    * calculateForceAndEnergy()                            calculate forces  and energy
    *
    *    setup()                                           rotate molecular multipole moments to lab frame
    *                                                      setup scaling maps and calculate induced dipoles (see calculateInducedDipoles below)
    *
    *    virtual calculateElectrostatic()                  calculate forces and torques
    *
    *                                                       
    *                                                      PME case includes the following calls:
    *
    *                                                          reciprocal [computeReciprocalSpaceInducedDipoleForceAndEnergy(),
    *                                                                      computeReciprocalSpaceFixedMultipoleForceAndEnergy]
    *
    *                                                          direct space calculations [calculatePmeDirectElectrostaticPairIxn()]
    *
    *                                                          self-energy [calculatePmeSelfEnergy()]
    *
    * 
    * setup()
    *    loadParticleData()                                load particle data   
    *    calculateInducedDipoles()                         calculate induced dipoles
    * 
    * 
    * virtual calculateInducedDipoles()                    calculate induced dipoles:
    *                                                          field at each site due to fixed multipoles first calculated
    *                                                          if polarization type == Direct,
    *                                                          initial induced dipoles are calculated, but are not converged.
    *                                                          if polarization type == Mutual, then loop until
    *                                                          induce dipoles converge.
    *                                                       For GK, include gkField in setup
    *                                                       For PME, base class method is used
    *
    * 
    *     virtual zeroFixedMultipoleFields()                zero fixed multipole vectors; for GK includes zeroing of gkField vector
    * 
    *     virtual calculateFixedMultipoleField()            calculate fixed multipole field -- particle pair loop 
    *                                                       gkField also calculated for GK
    *                                                       for PME, reciprocal, direct space (particle pair loop) and self terms calculated
    *                                                       
    * 
    *         virtual calculateFixedMultipoleFieldPairIxn() pair ixn for fixed multipole
    *                                                       gkField also calculated for GK
    *                                                       for PME, direct space ixn calculated here
    * 
    *     virtual initializeInducedDipoles()                initialize induced dipoles; for PME, calculateReciprocalSpaceInducedDipoleField()
    *                                                       called in case polarization type == Direct
    *
    *     convergeInduceDipoles()                           loop until induced dipoles converge
    *
    *         updateInducedDipoleFields()                   update fields at each site due other induced dipoles
    *
    *           virtual calculateInducedDipoleFields()      calculate induced dipole field at each site by looping over particle pairs 
    *                                                       for PME includes reciprocal space calculation calculateReciprocalSpaceInducedDipoleField(), 
    *                                                       direct space calculateDirectInducedDipolePairIxns() and self terms
    *
    *              virtual calculateInducedDipolePairIxns() field at particle i due particle j's induced dipole and vice versa; for GK includes GK field
    */

public:

    /** 
     * This is an enumeration of the different methods that may be used for handling long range Multipole forces.
     */
    enum NonbondedMethod {

        /**
         * No cutoff is applied to the interactions.  The full set of N^2 interactions is computed exactly.
         * This necessarily means that periodic boundary conditions cannot be used.  This is the default.
         */

       /** 
         * Periodic boundary conditions are used, and Particle-Mesh Ewald (PME) summation is used to compute the interaction of each particle
         * with all periodic copies of every other particle.
         */
        PME = 1, IPS = 2
    };

    enum PolarizationType {

        /** 
         * Mutual polarization
         */
        Mutual = 0,


    };

    /**
     * Constructor
     * 
     */
    pGM_ReferenceMultipoleForce();
 
    /**
     * Constructor
     * 
     * @param nonbondedMethod nonbonded method
     */
    pGM_ReferenceMultipoleForce(NonbondedMethod nonbondedMethod);
 
    /**
     * Destructor
     * 
     */
    virtual ~pGM_ReferenceMultipoleForce() {};
 
    /**
     * Get nonbonded method.
     * 
     * @return nonbonded method
     */
    NonbondedMethod getNonbondedMethod() const;

    /**
     * Set nonbonded method.
     * 
     * @param nonbondedMethod nonbonded method
     */
    void setNonbondedMethod(NonbondedMethod nonbondedMethod);

    /**
     * Get polarization type.
     * 
     * @return polarization type
     */
    PolarizationType getPolarizationType() const;

    /**
     * Set polarization type.
     * 
     * @param  polarizationType polarization type
     */
    void setPolarizationType(PolarizationType polarizationType);

    /**
     * Get flag indicating if mutual induced dipoles are converged.
     *
     * @return nonzero if converged
     *
     */
    int getMutualInducedDipoleConverged() const;

    /**
     * Get the number of iterations used in computing mutual induced dipoles.
     *
     * @return number of iterations
     *
     */
    int getMutualInducedDipoleIterations() const;

    /**
     * Get the final epsilon for mutual induced dipoles.
     *
     *  @return epsilon
     *
     */
    double getMutualInducedDipoleEpsilon() const;

    /**
     * Set the target epsilon for converging mutual induced dipoles.
     *
     * @param targetEpsilon target epsilon for converging mutual induced dipoles
     *
     */
    void setMutualInducedDipoleTargetEpsilon(double targetEpsilon);

    /**
     * Get the target epsilon for converging mutual induced dipoles.
     *
     * @return target epsilon for converging mutual induced dipoles
     *
     */
    double getMutualInducedDipoleTargetEpsilon() const;

    /**
     * Set the maximum number of iterations to be executed in converging mutual induced dipoles.
     *
     * @param maximumMutualInducedDipoleIterations maximum number of iterations to be executed in converging mutual induced dipoles
     *
     */
    void setMaximumMutualInducedDipoleIterations(int maximumMutualInducedDipoleIterations);

    /**
     * Get the maximum number of iterations to be executed in converging mutual induced dipoles.
     *
     * @return maximum number of iterations to be executed in converging mutual induced dipoles
     * 
     */
    int getMaximumMutualInducedDipoleIterations() const;

    /**
     * Calculate force and energy.
     *
     * @param particlePositions         Cartesian coordinates of particles
     * @param charges                   scalar charges for each particle
     * @param dipoles                   molecular frame dipoles for each particle
     * @param polarity                  polarity for each particle
     * @param multipoleAtomCovalentInfo covalent info needed to set scaling factors
     * @param forces                    add forces to this vector
     *
     * @return energy
     */
    double calculateForceAndEnergy(const std::vector<OpenMM::Vec3>& particlePositions,
                                   const std::vector<double>& charges,
                                   const std::vector<double>& dipoles,
                                   const std::vector<double>& beta,
                                   const std::vector<double>& polarity,
                                   const std::vector< std::vector< std::vector<int> > >& multipoleAtomCovalentInfo,
                                   std::vector<OpenMM::Vec3>& forces);

    /**
     * Calculate particle induced dipoles.
     *
     * @param masses                    particle masses
     * @param particlePositions         Cartesian coordinates of particles
     * @param charges                   scalar charges for each particle
     * @param dipoles                   molecular frame dipoles for each particle
     * @param polarity                  polarity for each particle
     * @param multipoleAtomCovalentInfo covalent info needed to set scaling factors
     * @param outputMultipoleMoments    output multipole moments
     */
    void calculateInducedDipoles(const std::vector<OpenMM::Vec3>& particlePositions,
                                 const std::vector<double>& charges,
                                 const std::vector<double>& dipoles,
                                 const std::vector<double>& beta,
                                 const std::vector<double>& polarity,
                                 const std::vector< std::vector< std::vector<int> > >& multipoleAtomCovalentInfo,
                                 std::vector<Vec3>& outputInducedDipoles);

    /**
     * Calculate particle permanent dipoles rotated in the lab frame.
     *
     * @param masses                    particle masses
     * @param particlePositions         Cartesian coordinates of particles
     * @param charges                   scalar charges for each particle
     * @param dipoles                   molecular frame dipoles for each particle
     * @param polarity                  polarity for each particle
     * @param multipoleAtomCovalentInfo covalent info needed to set scaling factors
     * @param outputMultipoleMoments    output multipole moments
     */

    void calculateLabFramePermanentDipoles(const std::vector<Vec3>& particlePositions,
                                           const std::vector<double>& charges,
                                           const std::vector<double>& dipoles,
                                           const std::vector<double>& beta,
                                           const std::vector<double>& polarity,
                                           const std::vector< std::vector< std::vector<int> > >& multipoleAtomCovalentInfo,
                                           std::vector<Vec3>& outputPermanentDipoles);

    /**
     * Calculate particle total dipoles.
     *
     * @param masses                    particle masses
     * @param particlePositions         Cartesian coordinates of particles
     * @param charges                   scalar charges for each particle
     * @param dipoles                   molecular frame dipoles for each particle
     * @param polarity                  polarity for each particle
     * @param multipoleAtomCovalentInfo covalent info needed to set scaling factors
     * @param outputTotalDipoles    output multipole moments
     */


    void calculateTotalDipoles(const std::vector<Vec3>& particlePositions,
                               const std::vector<double>& charges,
                               const std::vector<double>& dipoles,
                               const std::vector<double>& beta,
                               const std::vector<double>& polarity,
                               const std::vector< std::vector< std::vector<int> > >& multipoleAtomCovalentInfo,
                               std::vector<Vec3>& outputTotalDipoles);



    /**
     * Calculate system multipole moments.
     *
     * @param masses                    particle masses
     * @param particlePositions         Cartesian coordinates of particles
     * @param charges                   scalar charges for each particle
     * @param dipoles                   molecular frame dipoles for each particle
     * @param polarity                  polarity for each particle
     * @param multipoleAtomCovalentInfo covalent info needed to set scaling factors
     * @param outputMultipoleMoments    output multipole moments
     */
    void calculateSystemMultipoleMoments(const std::vector<double>& masses,
                                               const std::vector<OpenMM::Vec3>& particlePositions,
                                               const std::vector<double>& charges,
                                               const std::vector<double>& dipoles,
                                               const std::vector<double>& beta,
                                               const std::vector<double>& polarity,
                                               const std::vector< std::vector< std::vector<int> > >& multipoleAtomCovalentInfo,
                                               std::vector<double>& outputMultipoleMoments);



protected:

    enum MultipoleParticleDataEnum { PARTICLE_POSITION, PARTICLE_CHARGE, PARTICLE_DIPOLE, PARTICLE_POLARITY, PARTICLE_FIELD, 
                                     PARTICLE_FIELD_POLAR, GK_FIELD, PARTICLE_INDUCED_DIPOLE, PARTICLE_INDUCED_DIPOLE_POLAR };


    /* 
     * Particle parameters and coordinates
     */
    class MultipoleParticleData {
        public:
            unsigned int particleIndex;    
            Vec3 position;
            double charge;
            Vec3 dipole;
            double polarity;
            double beta;
    };
    
    /**
     * Particle parameters transformed into fractional coordinates
     */
    class TransformedMultipole {
    public:
        double charge;
        Vec3 dipole;
    };

    /* 
     * Helper class used in calculating induced dipoles
     */
    struct UpdateInducedDipoleFieldStruct {
            UpdateInducedDipoleFieldStruct(std::vector<OpenMM::Vec3>& inputFixed_E_Field, std::vector<OpenMM::Vec3>& inputInducedDipoles, std::vector<std::vector<Vec3> >& extrapolatedDipoles, std::vector<std::vector<double> >& extrapolatedDipoleFieldGradient);
            std::vector<OpenMM::Vec3>* fixedMultipoleField;
            std::vector<OpenMM::Vec3>* inducedDipoles;
            std::vector<OpenMM::Vec3> inducedDipoleField;
            std::vector<std::vector<double> > inducedDipoleFieldGradient;
    };

    unsigned int _numParticles;

    NonbondedMethod _nonbondedMethod;
    PolarizationType _polarizationType;

    double _electric;
    double _dielectric;

    enum ScaleType { D_SCALE, P_SCALE, M_SCALE, U_SCALE, LAST_SCALE_TYPE_INDEX };
    std::vector<  std::vector< MapIntRealOpenMM > > _scaleMaps;
    std::vector<unsigned int> _maxScaleIndex;
    double _dScale[5];
    double _pScale[5];
    double _mScale[5];
    double _uScale[5];

    std::vector<TransformedMultipole> _transformed;
    std::vector<Vec3> _fixedMultipoleField;
    std::vector<Vec3> _fixedMultipoleFieldPolar;
    std::vector<Vec3> _inducedDipole;
    std::vector<Vec3> _inducedDipolePolar;
    std::vector<std::vector<Vec3> > _ptDipoleP;
    std::vector<std::vector<Vec3> > _ptDipoleD;
    std::vector<std::vector<double> > _ptDipoleFieldGradientP;
    std::vector<std::vector<double> > _ptDipoleFieldGradientD;

    int _mutualInducedDipoleConverged;
    int _mutualInducedDipoleIterations;
    int _maximumMutualInducedDipoleIterations;
    int _maxPTOrder;
    std::vector<double>  _extrapolationCoefficients;
    std::vector<double>  _extPartCoefficients;
    double  _mutualInducedDipoleEpsilon;
    double  _mutualInducedDipoleTargetEpsilon;
    double  _debye;

    /**
     * Helper constructor method to centralize initialization of objects.
     *
     */
    void initialize();

    /**
     * Load particle data.
     *
     * @param particlePositions   particle coordinates
     * @param charges             charges
     * @param dipoles             dipoles
     * @param polarity            polarity
     * @param particleData        output data struct
     *
     */
    void loadParticleData(const std::vector<OpenMM::Vec3>& particlePositions, 
                          const std::vector<double>& charges,
                          const std::vector<double>& dipoles,
                          const std::vector<double>& beta,
                          const std::vector<double>& polarity,
                          const std::vector< std::vector< std::vector<int> > >& multipoleAtomCovalentInfo,
                          std::vector<MultipoleParticleData>& particleData) const;

    /**
     * Calculate fixed multipole fields.
     *
     * @param particleData vector of particle data
     * 
     */
    virtual void calculateFixedMultipoleField(const std::vector<MultipoleParticleData>& particleData);

    /**
     * Set flag indicating if mutual induced dipoles are converged.
     * 
     * @param converged nonzero if converged
     *
     */
    void setMutualInducedDipoleConverged(int converged);

    /**
     * Set number of iterations used in computing mutual induced dipoles.
     * 
     * @param  number of iterations
     * 
     */
    void setMutualInducedDipoleIterations(int iterations);

    /**
     * Set the final epsilon for mutual induced dipoles.
     * 
     * @param epsilon
     *
     */
    void setMutualInducedDipoleEpsilon(double epsilon);


    /**
     * Zero fixed multipole fields.
     */
    virtual void zeroFixedMultipoleFields();

    /**
     * Calculate electric field at particle I due fixed multipoles at particle J and vice versa
     * (field at particle J due fixed multipoles at particle I).
     * 
     * @param particleI               positions and parameters (charge, labFrame dipoles, quadrupoles, ...) for particle I
     * @param particleJ               positions and parameters (charge, labFrame dipoles, quadrupoles, ...) for particle J
     */
    virtual void calculateFixedMultipoleFieldPairIxn(const MultipoleParticleData& particleI, const MultipoleParticleData& particleJ);

    /**
     * Initialize induced dipoles
     *
     * @param updateInducedDipoleFields vector of UpdateInducedDipoleFieldStruct containing input induced dipoles and output fields
     */
    virtual void initializeInducedDipoles(std::vector<UpdateInducedDipoleFieldStruct>& updateInducedDipoleFields); 

    /**
     * Calculate field at particle I due induced dipole at particle J and vice versa
     * (field at particle J due induced dipole at particle I).
     * 
     * @param particleI               index of particle I
     * @param particleJ               index of particle J
     * @param rr3                     damped 1/r^3 factor
     * @param rr5                     damped 1/r^5 factor
     * @param delta                   delta of particle positions: particleJ.x - particleI.x, ...
     * @param inducedDipole           vector of induced dipoles
     * @param field                   vector of induced dipole fields
     */
    void calculateInducedDipolePairIxn(unsigned int particleI, unsigned int particleJ,
                                       double rr3, double rr5, const Vec3& delta,
                                       const std::vector<Vec3>& inducedDipole,
                                       std::vector<Vec3>& field) const;

    /**
     * Calculate fields due induced dipoles at each site.
     *
     * @param particleI                 positions and parameters (charge, labFrame dipoles, quadrupoles, ...) for particle I
     * @param particleJ                 positions and parameters (charge, labFrame dipoles, quadrupoles, ...) for particle J
     * @param updateInducedDipoleFields vector of UpdateInducedDipoleFieldStruct containing input induced dipoles and output fields
     */
    virtual void calculateInducedDipolePairIxns(const MultipoleParticleData& particleI, const MultipoleParticleData& particleJ,
                                                std::vector<UpdateInducedDipoleFieldStruct>& updateInducedDipoleFields);

    /**
     * Calculate induced dipole fields.
     * 
     * @param particleData              vector of particle positions and parameters (charge, labFrame dipoles, quadrupoles, ...)
     * @param updateInducedDipoleFields vector of UpdateInducedDipoleFieldStruct containing input induced dipoles and output fields
     */
    virtual void calculateInducedDipoleFields(const std::vector<MultipoleParticleData>& particleData,
                                              std::vector<UpdateInducedDipoleFieldStruct>& updateInducedDipoleFields);

    void convergeInduceDipolesByCG(const std::vector<MultipoleParticleData>& particleData, std::vector<UpdateInducedDipoleFieldStruct>& updateInducedDipoleField);
    /**
     * Converge induced dipoles.
     * 
     * @param particleData              vector of particle positions and parameters (charge, labFrame dipoles, quadrupoles, ...)
     * @param updateInducedDipoleFields vector of UpdateInducedDipoleFieldStruct containing input induced dipoles and output fields
     */
    void convergeInduceDipolesByDIIS(const std::vector<MultipoleParticleData>& particleData,
                                     std::vector<UpdateInducedDipoleFieldStruct>& calculateInducedDipoleField);
    
    /**
     * Use DIIS to compute the weighting coefficients for the new induced dipoles.
     * 
     * @param prevErrors    the vector of errors from previous iterations
     * @param coefficients  the coefficients will be stored into this
     */
    void computeDIISCoefficients(const std::vector<std::vector<Vec3> >& prevErrors, std::vector<double>& coefficients) const;

    /**
     * Calculate induced dipoles.
     * 
     * @param particleData      vector of particle positions and parameters (charge, labFrame dipoles, quadrupoles, ...)
     */
    virtual void calculateInducedDipoles(const std::vector<MultipoleParticleData>& particleData);

    /**
     * Setup: 
     *        if needed invert multipole moments at chiral centers
     *        rotate molecular multipole moments to lab frame 
     *        setup scaling maps and 
     *        calculate induced dipoles (see calculateInducedDipoles below)
     *
     * @param particlePositions         Cartesian coordinates of particles
     * @param charges                   scalar charges for each particle
     * @param dipoles                   molecular frame dipoles for each particle
     * @param beta                      Gaussian radius
     * @param polarity                  polarity for each particle
     * @param multipoleAtomCovalentInfo covalent info needed to set scaling factors
     * @param particleData              output vector of parameters (charge, labFrame dipoles, quadrupoles, ...) for particles
     *
     */
    void setup(const std::vector<OpenMM::Vec3>& particlePositions,
               const std::vector<double>& charges,
               const std::vector<double>& dipoles,
               const std::vector<double>& beta,
               const std::vector<double>& polarity,
               const std::vector< std::vector< std::vector<int> > >& multipoleAtomCovalentInfo,
               std::vector<MultipoleParticleData>& particleData);

    /**
     * Calculate electrostatic interaction between particles I and K.
     * 
     * @param particleI         positions and parameters (charge, labFrame dipoles, quadrupoles, ...) for particle I
     * @param particleK         positions and parameters (charge, labFrame dipoles, quadrupoles, ...) for particle K
     * @param scalingFactors    scaling factors for interaction
     * @param forces            vector of particle forces to be updated
     * @param torque            vector of particle torques to be updated
     */
    double calculateElectrostaticPairIxn(const MultipoleParticleData& particleI, const MultipoleParticleData& particleK,
                                         std::vector<OpenMM::Vec3>& forces) const;


    /**
     * Calculate electrostatic forces
     * 
     * @param particleData            vector of parameters (charge, labFrame dipoles, quadrupoles, ...) for particles
     * @param torques                 output torques
     * @param forces                  output forces 
     *
     * @return energy
     */
    virtual double calculateElectrostatic(const std::vector<MultipoleParticleData>& particleData, 
                                          std::vector<OpenMM::Vec3>& forces);

    /**
     * Normalize a Vec3
     *
     * @param vectorToNormalize vector to normalize
     *
     * @return norm of vector on input
     * 
     */
    double normalizeVec3(Vec3& vectorToNormalize) const;

    /**
     * Initialize vector of double (size=numParticles)
     *
     * @param vectorToInitialize vector to initialize
     * 
     */
    void initializeRealOpenMMVector(std::vector<double>& vectorToInitialize) const;

    /**
     * Initialize vector of Vec3 (size=numParticles)
     *
     * @param vectorToInitialize vector to initialize
     * 
     */
    void initializeVec3Vector(std::vector<Vec3>& vectorToInitialize) const;

    /**
     * Copy vector of Vec3
     *
     * @param inputVector  vector to copy
     * @param outputVector output vector
     * 
     */
    void copyVec3Vector(const std::vector<OpenMM::Vec3>& inputVector, std::vector<OpenMM::Vec3>& outputVector) const;


    /**
     * Apply periodic boundary conditions to difference in positions
     * 
     * @param deltaR  difference in particle positions; modified on output after applying PBC
     * 
     */
    virtual void getPeriodicDelta(Vec3& deltaR) const {};
};

class pGM_ReferenceIPSMultipoleForce : public pGM_ReferenceMultipoleForce {

public:

    /**
     * Constructor
     * 
     */
    pGM_ReferenceIPSMultipoleForce();
 
    /**
     * Destructor
     * 
     */
    ~pGM_ReferenceIPSMultipoleForce();
 
    /**
     * Get cutoff distance.
     *
     * @return cutoff distance
     *
     */
    double getCutoffDistance() const;

    /**
     * Set cutoff distance.
     *
     * @return cutoff distance
     *
     */
    void setCutoffDistance(double cutoffDistance);


    /**
     * Calculate electrostatic forces.
     * 
     * @param particleData            vector of parameters (charge, labFrame dipoles, quadrupoles, ...) for particles
     * @param forces                  output forces 
     *
     * @return energy
     */
    double calculateElectrostatic(const std::vector<MultipoleParticleData>& particleData, 
                                  std::vector<OpenMM::Vec3>& forces);

};

class pGM_ReferencePmeMultipoleForce : public pGM_ReferenceMultipoleForce {

public:

    /**
     * Constructor
     * 
     */
    pGM_ReferencePmeMultipoleForce();
 
    /**
     * Destructor
     * 
     */
    ~pGM_ReferencePmeMultipoleForce();
 
    /**
     * Get cutoff distance.
     *
     * @return cutoff distance
     *
     */
    double getCutoffDistance() const;

    /**
     * Set cutoff distance.
     *
     * @return cutoff distance
     *
     */
    void setCutoffDistance(double cutoffDistance);

    /**
     * Get alpha used in Ewald summation.
     *
     * @return alpha
     *
     */
    double getAlphaEwald() const;

    /**
     * Set alpha used in Ewald summation.
     *
     * @return alpha
     *
     */
    void setAlphaEwald(double alphaEwald);

    /**
     * Get PME grid dimensions.
     *
     * @param pmeGridDimensions contains PME grid dimensions upon return

     *
     */
    void getPmeGridDimensions(std::vector<int>& pmeGridDimensions) const;

    /**
     * Set PME grid dimensions.
     *
     * @param pmeGridDimensions input PME grid dimensions 
     *
     */
    void setPmeGridDimensions(std::vector<int>& pmeGridDimensions);

    /**
     * Set periodic box size.
     *
     * @param vectors    the vectors defining the periodic box
     */
     void setPeriodicBoxSize(OpenMM::Vec3* vectors);

private:

    static const int pGM_PME_ORDER;
    static const double SQRT_PI;

    double _alphaEwald;
    double _cutoffDistance;
    double _cutoffDistanceSquared;

    Vec3 _recipBoxVectors[3];
    Vec3 _periodicBoxVectors[3];

    int _totalGridSize;
    IntVec _pmeGridDimensions;

    unsigned int _pmeGridSize;
    std::complex<double>* _pmeGrid;
 
    std::vector<double> _pmeBsplineModuli[3];
    std::vector<double4> _thetai[3];
    std::vector<IntVec> _iGrid;
    std::vector<double> _phi;
    std::vector<double> _phid;
    std::vector<double> _phip;
    std::vector<double> _phidp;
    std::vector<double4> _pmeBsplineTheta;
    std::vector<double4> _pmeBsplineDtheta;

    /**
     * Resize PME arrays.
     * 
     */
    void resizePmeArrays();

    /**
     * Zero Pme grid.
     */
    void initializePmeGrid();

    /**
     * Modify input vector of differences in particle positions for periodic boundary conditions.
     * 
     * @param delta                   input vector of difference in particle positions; on output adjusted for
     *                                periodic boundary conditions
     */
    void getPeriodicDelta(Vec3& deltaR) const;

    /**
     * Calculate damped inverse distances.
     * 
     * @param particleI               positions and parameters (charge, labFrame dipoles, quadrupoles, ...) for particle I
     * @param particleJ               positions and parameters (charge, labFrame dipoles, quadrupoles, ...) for particle J
     */
    void getDampedInverseDistances(const MultipoleParticleData& particleI, const MultipoleParticleData& particleJ, double r) const;
    
    /**
     * Initialize B-spline moduli.
     * 
     */
    void initializeBSplineModuli();

    /**
     * Calculate direct-space field at site I due fixed multipoles at site J and vice versa.
     * 
     * @param particleI               positions and parameters (charge, labFrame dipoles, quadrupoles, ...) for particle I
     * @param particleJ               positions and parameters (charge, labFrame dipoles, quadrupoles, ...) for particle J
     * @param dScale                  d-scale value for i-j interaction
     * @param pScale                  p-scale value for i-j interaction
     */
    void calculateFixedMultipoleFieldPairIxn(const MultipoleParticleData& particleI, const MultipoleParticleData& particleJ,
                                             double dscale, double pscale);
    
    /**
     * Calculate fixed multipole fields.
     *
     * @param particleData vector particle data
     * 
     */
    void calculateFixedMultipoleField(const std::vector<MultipoleParticleData>& particleData);

    /**
     * This is called from computeAmoebaBsplines().  It calculates the spline coefficients for a single atom along a single axis.
     * 
     * @param thetai output spline coefficients
     * @param w offset from grid point
     */
    void computeBSplinePoint(std::vector<double4>& thetai, double w);
    
    /**
     * Compute bspline coefficients.
     *
     * @param particleData   vector of particle positions and parameters (charge, labFrame dipoles, quadrupoles, ...)
     */
    void computeBsplines(const std::vector<MultipoleParticleData>& particleData);

    /**
     * Transform multipoles from cartesian coordinates to fractional coordinates.
     */
    void transformMultipolesToFractionalCoordinates(const std::vector<MultipoleParticleData>& particleData);

    /**
     * Transform potential from fractional coordinates to cartesian coordinates.
     */
    void transformPotentialToCartesianCoordinates(const std::vector<double>& fphi, std::vector<double>& cphi) const;

    /**
     * Spread fixed multipoles onto PME grid.
     * 
     * @param particleData vector of particle positions and parameters (charge, labFrame dipoles, quadrupoles, ...)
     */
    void spreadFixedMultipolesOntoGrid(const std::vector<MultipoleParticleData>& particleData);

    /**
     * Perform reciprocal convolution.
     * 
     */
    void performReciprocalConvolution();

    /**
     * Compute reciprocal potential due fixed multipoles at each particle site.
     * 
     */
    void computeFixedPotentialFromGrid(void);

    /**
     * Compute reciprocal potential due fixed multipoles at each particle site.
     * 
     */
    void computeInducedPotentialFromGrid();

    /**
     * Calculate reciprocal space energy and force due to fixed multipoles.
     * 
     * @param particleData    vector of particle positions and parameters (charge, labFrame dipoles, quadrupoles, ...)
     * @param forces          upon return updated vector of forces
     * @param torques         upon return updated vector of torques
     *
     * @return energy
     */
    double computeReciprocalSpaceFixedMultipoleForceAndEnergy(const std::vector<MultipoleParticleData>& particleData,
                                                              std::vector<Vec3>& forces, std::vector<Vec3>& torques) const;

    /**
     * Set reciprocal space fixed multipole fields.
     * 
     */
    void recordFixedMultipoleField();

    /**
     * Compute the potential due to the reciprocal space PME calculation for induced dipoles.
     *
     * @param updateInducedDipoleFields vector of UpdateInducedDipoleFieldStruct containing input induced dipoles and output fields
     */
    void calculateReciprocalSpaceInducedDipoleField(std::vector<UpdateInducedDipoleFieldStruct>& updateInducedDipoleFields);

    /**
     * Calculate field at particleI due to induced dipole at particle J and vice versa.
     *
     * @param iIndex        particle I index
     * @param jIndex        particle J index
     * @param preFactor1    first factor used in calculating field
     * @param preFactor2    second factor used in calculating field
     * @param delta         delta in particle positions after adjusting for periodic boundary conditions
     * @param inducedDipole vector of induced dipoles
     * @param field         vector of field at each particle due induced dipole of other particles
     */
    void calculateDirectInducedDipolePairIxn(unsigned int iIndex, unsigned int jIndex,
                                             double preFactor1, double preFactor2, const Vec3& delta,
                                             const std::vector<Vec3>& inducedDipole,
                                             std::vector<Vec3>& field) const;

    /**
     * Calculate direct space field at particleI due to induced dipole at particle J and vice versa for
     * inducedDipole and inducedDipolePolar.
     * 
     * @param particleI                 positions and parameters (charge, labFrame dipoles, quadrupoles, ...) for particle I
     * @param particleJ                 positions and parameters (charge, labFrame dipoles, quadrupoles, ...) for particle J
     * @param updateInducedDipoleFields vector of UpdateInducedDipoleFieldStruct containing input induced dipoles and output fields
     */
    void calculateDirectInducedDipolePairIxns(const MultipoleParticleData& particleI,
                                              const MultipoleParticleData& particleJ,
                                              std::vector<UpdateInducedDipoleFieldStruct>& updateInducedDipoleFields);

    /**
     * Initialize induced dipoles
     *
     * @param updateInducedDipoleFields vector of UpdateInducedDipoleFieldStruct containing input induced dipoles and output fields
     */
    void initializeInducedDipoles(std::vector<UpdateInducedDipoleFieldStruct>& updateInducedDipoleFields); 

    /**
     * Spread induced dipoles onto grid.
     *
     * @param inputInducedDipole      induced dipole value
     * @param inputInducedDipolePolar induced dipole polar value
     */
    void spreadInducedDipolesOnGrid(const std::vector<Vec3>& inputInducedDipole,
                                    const std::vector<Vec3>& inputInducedDipolePolar);

    /**
     * Calculate induced dipole fields.
     * 
     * @param particleData              vector of particle positions and parameters (charge, labFrame dipoles, quadrupoles, ...)
     * @param updateInducedDipoleFields vector of UpdateInducedDipoleFieldStruct containing input induced dipoles and output fields
     */
    void calculateInducedDipoleFields(const std::vector<MultipoleParticleData>& particleData,
                                      std::vector<UpdateInducedDipoleFieldStruct>& updateInducedDipoleFields);

    /**
     * Set reciprocal space induced dipole fields. 
     *
     * @param field       reciprocal space output induced dipole field value at each site
     * @param fieldPolar  reciprocal space output induced dipole polar field value at each site
     * 
     */
    void recordInducedDipoleField(std::vector<Vec3>& field, std::vector<Vec3>& fieldPolar);

    /**
     * Compute Pme self energy.
     *
     * @param particleData            vector of parameters (charge, labFrame dipoles, quadrupoles, ...) for particles
     */
    double calculatePmeSelfEnergy(const std::vector<MultipoleParticleData>& particleData) const;


    /**
     * Calculate direct space electrostatic interaction between particles I and J.
     * 
     * @param particleI         positions and parameters (charge, labFrame dipoles, quadrupoles, ...) for particle I
     * @param particleJ         positions and parameters (charge, labFrame dipoles, quadrupoles, ...) for particle J
     * @param forces            vector of particle forces to be updated
     */
    double calculatePmeDirectElectrostaticPairIxn(const MultipoleParticleData& particleI, const MultipoleParticleData& particleJ,
                                                  std::vector<Vec3>& forces) const;

    /**
     * Calculate reciprocal space energy/force/torque for dipole interaction.
     * 
     * @param polarizationType  if 'Direct' polariztion, only initial induced dipoles calculated
     *                          if 'Mutual' polariztion, induced dipoles converged to specified tolerance
     * @param particleData      vector of particle positions and parameters (charge, labFrame dipoles, quadrupoles, ...)
     * @param forces            vector of particle forces to be updated
     * @param torques           vector of particle torques to be updated
     */
     double computeReciprocalSpaceInducedDipoleForceAndEnergy(pGM_ReferenceMultipoleForce::PolarizationType polarizationType,
                                                              const std::vector<MultipoleParticleData>& particleData,
                                                              std::vector<Vec3>& forces) const;

    /**
     * Calculate electrostatic forces.
     * 
     * @param particleData            vector of parameters (charge, labFrame dipoles, quadrupoles, ...) for particles
     * @param forces                  output forces 
     *
     * @return energy
     */
    double calculateElectrostatic(const std::vector<MultipoleParticleData>& particleData, 
                                  std::vector<OpenMM::Vec3>& forces);

};

} // namespace OpenMM

#endif // _AmoebaReferenceMultipoleForce___
